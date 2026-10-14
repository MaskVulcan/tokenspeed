# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""GPU D2H/L3/H2D byte tests, including independent native Mooncake clients.

The default test uses MemoryKvStore. Set MOONCAKE_GPU_TEST_CONFIG to a JSON
Mooncake configuration with global_segment_size=0 to also run native tests.
Start a master and an independent storage owner first. Each native writer
exits before its reader starts, so neither GPU nor Host allocations survive.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from test.runtime.test_host_executor import _synthetic_arena

from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")


class L3FlatKvRoundTripTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch

            import tokenspeed.runtime.cache.l2.executor as executor_module
            from tokenspeed.runtime.cache.l3.backend import MemoryKvStore
            from tokenspeed.runtime.cache.transfer.layout import (
                CacheField,
                CacheGroupLayout,
                CacheTransferLayout,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            if os.environ.get("MOONCAKE_GPU_TEST_CONFIG"):
                raise
            self.skipTest(f"needs runtime dependencies: {exc}")
        if not torch.cuda.is_available():
            if os.environ.get("MOONCAKE_GPU_TEST_CONFIG"):
                self.fail("native Mooncake GPU test requires a CUDA device")
            self.skipTest("needs a CUDA device")
        self.torch = torch
        self.executor_module = executor_module
        self.MemoryKvStore = MemoryKvStore
        self.CacheField = CacheField
        self.CacheGroupLayout = CacheGroupLayout
        self.CacheTransferLayout = CacheTransferLayout

    def test_d2h_backup_host_wipe_prefetch_h2d_restores_bytes(self):
        self._exercise(
            backend="memory",
            phase="roundtrip",
            io_backend="direct",
            namespace=f"e2e-{uuid.uuid4().hex}",
            native_config=None,
        )

    def test_native_mooncake_survives_writer_process_exit(self):
        raw = os.environ.get("MOONCAKE_GPU_TEST_CONFIG")
        if not raw:
            self.skipTest("set MOONCAKE_GPU_TEST_CONFIG for a live storage owner")
        config = json.loads(raw)
        self.assertEqual(config["global_segment_size"], 0)
        for io_backend in ("direct", "kernel"):
            with self.subTest(io_backend=io_backend):
                namespace = f"gpu-native-{uuid.uuid4().hex}"
                for phase in ("put", "get"):
                    result = subprocess.run(
                        [
                            sys.executable,
                            os.path.abspath(__file__),
                            "--native-worker",
                            phase,
                            "--io-backend",
                            io_backend,
                            "--namespace",
                            namespace,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=120,
                        check=False,
                    )
                    self.assertEqual(
                        result.returncode, 0, f"{result.stdout}\n{result.stderr}"
                    )
                    print(f"native Mooncake {io_backend} {phase}: passed", flush=True)

    def _complete(self, executor, *, op_id, expected):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            completed = executor.poll().completions
            if completed:
                self.assertEqual(len(completed), 1)
                self.assertEqual(completed[0].op_id, op_id)
                self.assertEqual(completed[0].success, expected)
                self.assertTrue(completed[0].retired)
                return
            time.sleep(0.001)
        self.fail(f"storage operation {op_id} did not complete")

    def _exercise(self, *, backend, phase, io_backend, namespace, native_config):
        torch = self.torch
        first = torch.full((128,), 0xCC, dtype=torch.uint8, device="cuda")
        second = torch.full((128,), 0xCC, dtype=torch.uint8, device="cuda")
        layout = self.CacheTransferLayout(
            num_lcm_blocks=4,
            groups=(
                self.CacheGroupLayout(
                    group_id="full",
                    cache_blocks_per_lcm_block=2,
                    fields=(
                        self.CacheField("layer.0.k", 0, 8, 8, 4),
                        self.CacheField("layer.0.v", 1, 16, 12, 6),
                    ),
                ),
                self.CacheGroupLayout(
                    group_id="state",
                    cache_blocks_per_lcm_block=1,
                    fields=(self.CacheField("layer.1.state", 0, 64, 10, 5),),
                ),
            ),
            buffers=(first, second),
            consumers=(("layer.0.k", "layer.0.v"), ("layer.1.state",)),
        )

        class SyntheticPool:
            def __init__(self):
                self.load_tracker = None
                self.arena = _synthetic_arena(layout)

            def cache_transfer_layout(self):
                return layout

            def register_layerwise_load_tracker(self, tracker):
                self.load_tracker = tracker

        pool = SyntheticPool()
        from tokenspeed.runtime.cache.l2.sizing import RetractionPoolRequest

        executor = self.executor_module.HostCacheExecutor(
            pool,
            l2_tier=True,
            snapshot_pool=RetractionPoolRequest(
                host_gb=0.0,
                ratio=0.0,
                max_retracted_requests=0,
                tail_lcm_blocks_per_request=0,
            ),
            slot_state_exporters=None,
            kvp_rank=0,
            host_ratio=1.0,
            host_size_gb=0,
            io_backend=io_backend,
            attn_tp_rank=0,
            host_memory_check=lambda size: None,
        )
        if backend == "memory":
            store = self.MemoryKvStore()
        else:
            self.assertEqual(backend, "mooncake")
            from tokenspeed.runtime.cache.l3.mooncake import MooncakeKvStore

            store = MooncakeKvStore(
                native_config,
                host_buffer=executor.host_storage.host_buffer,
                tp_size=1,
                pp_size=1,
            )
        from tokenspeed_scheduler import StorageKind

        from tokenspeed.runtime.cache.l3.async_executor import AsyncL3Executor
        from tokenspeed.runtime.cache.l3.config import L3AsyncConfig
        from tokenspeed.runtime.cache.l3.executor import L3HostStore

        l3 = AsyncL3Executor(
            L3HostStore(store, executor.host_storage, key_prefix=namespace, rank=0),
            prefix_for_weight_version=lambda version: f"{namespace}-{version}",
            config=L3AsyncConfig.from_extra_config(
                None, host_blocks=4, prefix_granularity=2
            ),
        )
        self.addCleanup(l3.close, timeout_s=5.0)
        self.addCleanup(executor.shutdown)

        keys = [(0, "h0", 0), (0, "h1", 0), (1, "h2", 0)]
        if phase in ("put", "roundtrip"):
            l3.submit_operations(
                [SimpleNamespace(op_id=1, kind=StorageKind.Query, keys=keys)]
            )
            self._complete(l3, op_id=1, expected=(False,) * 3)
            first[16:20].fill_(0x11)
            second[28:34].fill_(0x12)
            first[40:44].fill_(0x41)
            second[64:70].fill_(0x42)
            first[94:99].fill_(0x73)
            torch.cuda.synchronize()

            executor._start_writing(  # pylint: disable=protected-access
                [7],
                [(0, 1, 1), (0, 4, 4), (1, 3, 3)],
                lane=executor._pinned_write_lane,
                prerequisite_stream=torch.cuda.current_stream(),
            )
            executor.write_stream.synchronize()
            write_results = executor.poll_results()
            self.assertEqual([int(event.op_id) for event in write_results], [7])
            l3.submit_operations(
                [
                    SimpleNamespace(
                        op_id=2,
                        kind=StorageKind.Put,
                        keys=keys,
                        host_blocks=[1, 4, 3],
                    )
                ]
            )
            self._complete(l3, op_id=2, expected=(True,) * 3)
            if phase == "put":
                return

        # Prove restore comes from L3, not leftover Host bytes.
        executor.host_storage.host_buffer.fill_(0)
        first.fill_(0xEE)
        second.fill_(0xEE)
        torch.cuda.synchronize()
        self.assertFalse(bool(executor.host_storage.host_buffer.any().item()))

        l3.submit_operations(
            [
                SimpleNamespace(
                    op_id=3, kind=StorageKind.Get, keys=keys, host_blocks=[2, 5, 4]
                )
            ]
        )
        self._complete(l3, op_id=3, expected=(True,) * 3)
        self.assertTrue(bool(executor.host_storage.host_buffer.any().item()))

        load_index = executor._start_loading(  # pylint: disable=protected-access
            [9],
            [(0, 2, 2), (0, 5, 5), (1, 4, 4)],
            prerequisite_stream=torch.cuda.current_stream(),
        )
        self.assertIsNotNone(load_index)
        pool.load_tracker.set_consumers(load_index)
        pool.load_tracker.wait_for_layer(0)
        pool.load_tracker.wait_for_layer(1)
        torch.cuda.synchronize()
        load_results = executor.poll_results()
        self.assertEqual([int(event.op_id) for event in load_results], [9])

        # Check entire allocations, including padding and untouched blocks.
        expected_first = torch.full((128,), 0xEE, dtype=torch.uint8)
        expected_second = torch.full((128,), 0xEE, dtype=torch.uint8)
        expected_first[24:28].fill_(0x11)
        expected_second[40:46].fill_(0x12)
        expected_first[48:52].fill_(0x41)
        expected_second[76:82].fill_(0x42)
        expected_first[104:109].fill_(0x73)
        self.assertTrue(torch.equal(first.cpu(), expected_first))
        self.assertTrue(torch.equal(second.cpu(), expected_second))

        self.assertTrue(l3.store.backend.remove_by_prefix(namespace))
        executor.host_storage.host_buffer.fill_(0)
        l3.submit_operations(
            [
                SimpleNamespace(
                    op_id=4, kind=StorageKind.Get, keys=keys, host_blocks=[2, 5, 4]
                )
            ]
        )
        self._complete(l3, op_id=4, expected=(False,) * 3)
        self.assertFalse(bool(executor.host_storage.host_buffer.any().item()))


if __name__ == "__main__":
    if "--native-worker" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--native-worker", choices=("put", "get"), required=True)
        parser.add_argument("--io-backend", choices=("direct", "kernel"), required=True)
        parser.add_argument("--namespace", required=True)
        args = parser.parse_args()
        case = L3FlatKvRoundTripTest()
        try:
            case.setUp()
            case._exercise(
                backend="mooncake",
                phase=args.native_worker,
                io_backend=args.io_backend,
                namespace=args.namespace,
                native_config=json.loads(os.environ["MOONCAKE_GPU_TEST_CONFIG"]),
            )
        finally:
            if not case.doCleanups():
                raise RuntimeError("native Mooncake client cleanup failed")
    else:
        unittest.main()
