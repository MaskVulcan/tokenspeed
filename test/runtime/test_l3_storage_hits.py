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

"""Async L3 control boundary and immutable Host operation snapshots."""

import inspect
from test.runtime.test_host_executor import _load_executor_module_without_triton
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from tokenspeed_scheduler import StorageKind

from tokenspeed.runtime.cache.l3.async_executor import AsyncL3Executor
from tokenspeed.runtime.cache.l3.backend import MemoryKvStore
from tokenspeed.runtime.cache.l3.config import L3AsyncConfig
from tokenspeed.runtime.cache.l3.executor import L3HostStore
from tokenspeed.runtime.cache.l3.io_types import StorageCompletion, StoragePoll
from tokenspeed.runtime.engine.l3_cache_hooks import L3CacheHooks


def hooks(device):
    return L3CacheHooks(
        SimpleNamespace(submit_requests=Mock()),
        device,
        config=L3AsyncConfig.from_extra_config(
            None, host_blocks=16, prefix_granularity=2
        ),
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        pp_size=1,
        pp_cpu_group=None,
        metrics=None,
    )


def test_local_completion_is_reported_once_and_does_not_select_a_prefix():
    queue = [StorageCompletion(7, (True, False), True, (False, True))]
    h = hooks(SimpleNamespace(poll=lambda: StoragePoll(tuple(queue), 0, 0, 0)))
    events = h.poll_ready_events(allow_forward=True, allow_storage=True)
    assert events[1].op_id == 7 and list(events[1].success) == [1, 0]
    queue.clear()
    assert len(h.poll_ready_events(allow_forward=False, allow_storage=False)) == 1
    assert not h._reports


def test_storage_ranges_and_namespace_are_captured_before_queueing():
    host = SimpleNamespace(
        host_buffer=bytearray(64), host_block_range=lambda g, b: ((b - 1) * 8, 8)
    )
    store = L3HostStore(MemoryKvStore(), host, key_prefix="old", rank=0)
    executor = AsyncL3Executor(
        store,
        config=L3AsyncConfig.from_extra_config(
            None, host_blocks=16, prefix_granularity=2
        ),
        prefix_for_weight_version=lambda version: f"new-{version}",
    )
    try:
        with patch.object(executor, "submit") as submit, patch.object(
            executor, "cancel"
        ) as cancel:
            executor.submit_operations(
                [
                    SimpleNamespace(
                        op_id=1,
                        kind=StorageKind.Get,
                        keys=[(0, "hash", 0)],
                        host_blocks=[2],
                    )
                ]
            )
            job = submit.call_args.args[0]
            executor.set_weight_version("v2")
            assert job.keys == ("old_hash|g0|o0|r0|c0",)
            assert job.offsets == (8,) and job.sizes == (8,)
            executor.submit_operations(
                [SimpleNamespace(op_id=1, kind=StorageKind.Cancel)]
            )
            cancel.assert_called_once_with(1)
    finally:
        executor.close(timeout_s=2)


def test_namespace_mutation_rejects_inflight_io_without_waiting():
    store = L3HostStore(
        MemoryKvStore(),
        SimpleNamespace(host_buffer=bytearray(64)),
        key_prefix="old",
        rank=0,
    )
    executor = AsyncL3Executor(
        store,
        config=L3AsyncConfig.from_extra_config(
            None, host_blocks=16, prefix_granularity=2
        ),
        prefix_for_weight_version=lambda version: f"new-{version}",
    )
    try:
        with patch.object(executor, "is_idle", return_value=False), patch.object(
            store, "rotate_namespace"
        ) as purge:
            assert executor.delete_namespace() is False
            with pytest.raises(RuntimeError, match="must retire"):
                executor.set_weight_version("v2")
            assert store.key_prefix == "old"
            purge.assert_not_called()
    finally:
        executor.close(timeout_s=2)


def test_metrics_sample_pending_transitions_without_extra_feedback(monkeypatch):
    import tokenspeed.runtime.engine.l3_cache_hooks as module

    now = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    snapshot, stats = object(), object()
    device = SimpleNamespace(snapshot=Mock(return_value=snapshot))
    h = hooks(device)
    h._metrics = Mock()
    h._scheduler = SimpleNamespace(
        storage_pending=Mock(return_value=False), storage_stats=Mock(return_value=stats)
    )
    h.record_metrics()
    h.record_metrics()
    assert device.snapshot.call_count == 1
    h._scheduler.storage_pending.return_value = True
    h.record_metrics()
    h._scheduler.storage_pending.return_value = False
    h.record_metrics()
    assert device.snapshot.call_count == 3
    h._metrics.record.assert_called_with(snapshot, stats)
    now[0] += 1
    h.record_metrics()
    assert device.snapshot.call_count == 4


@pytest.mark.parametrize("reject", ["health", "queue"])
def test_rejected_get_retries_same_remote_keys_after_recovery(reject):
    import queue
    import time
    from unittest.mock import patch

    import tokenspeed_scheduler as ts

    from tokenspeed.runtime.cache.l3.async_executor import AsyncL3Executor, StorageJob

    config = L3AsyncConfig.from_extra_config(None, host_blocks=31, prefix_granularity=2)
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = 2
    cfg.num_snapshot_pages = 1
    cfg.num_device_pages = 32
    cfg.num_host_pages = 32
    cfg.max_batch_size = 4
    cfg.max_scheduled_tokens = 16
    cfg.enable_l3_storage = True
    cfg.storage_config = config.scheduler_config()
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id="kv",
            block_granularity=2,
            total_pages=32,
            attn_kind=ts.AttnKind.Full,
        )
    ]
    scheduler = ts.Scheduler(cfg)
    host = SimpleNamespace(host_buffer=bytearray(256))
    backend = MemoryKvStore()
    store = L3HostStore(backend, host, key_prefix="retry", rank=0)
    now = [1.0]
    executor = AsyncL3Executor(
        store, config=config, clock=lambda: now[0], prefix_for_weight_version=None
    )
    h = L3CacheHooks(
        scheduler,
        SimpleNamespace(poll=executor.poll),
        config=config,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        pp_size=1,
        pp_cpu_group=None,
        metrics=None,
    )

    def advance():
        events = h.poll_ready_events(allow_forward=True, allow_storage=True)
        message = ts.ExecutionEvent()
        for event in events:
            message.add_event(event)
        scheduler.advance(message)
        return len(events) > 1

    def wait():
        deadline = time.monotonic() + 3
        while not advance():
            assert time.monotonic() < deadline
            time.sleep(0.001)

    def job(op):
        keys = tuple(
            store.object_key(content_hash, group, offset)
            for group, content_hash, offset in op.keys
        )
        return StorageJob(
            op.op_id,
            "query" if op.kind == ts.StorageKind.Query else "get",
            keys,
            (
                ()
                if op.kind == ts.StorageKind.Query
                else tuple(8 * i for i in range(len(keys)))
            ),
            () if op.kind == ts.StorageKind.Query else (8,) * len(keys),
        )

    def submit(rid):
        spec = ts.RequestSpec()
        spec.request_id = rid
        spec.tokens = list(range(8))
        spec.max_new_tokens = 2
        scheduler.submit_requests([spec])

    try:
        advance()
        submit("first")
        query = job(scheduler.next_execution_plan().storage[0])
        backend.batch_put_from(
            query.keys,
            bytearray(b"abcdefgh" * len(query.keys)),
            tuple(8 * i for i in range(len(query.keys))),
            (8,) * len(query.keys),
        )
        executor.submit(query)
        wait()
        read = job(scheduler.next_execution_plan().storage[0])
        if reject == "health":
            executor._health.report_failure(executor._health.begin_operation())
            executor.submit(read)
        else:
            with patch.object(
                executor._queues["get"], "put_nowait", side_effect=queue.Full
            ):
                executor.submit(read)
        wait()
        assert scheduler.storage_negative_keys() == 0
        get = next(row for row in executor.snapshot().operations if row.kind == "get")
        assert dict(get.events)["backend_calls"] == 0
        plan = scheduler.next_execution_plan()
        assert list(plan.forward[0].extend_prefix_lens) == [0]  # Ordinary fallback.
        finished = ts.ForwardEvent.Abort()
        finished.request_id = "first"
        feedback = ts.ExecutionEvent()
        feedback.add_event(finished)
        scheduler.advance(feedback)
        now[0] += 100  # Allow the monitor's one recovery probe.
        submit("after-recovery")
        query2 = job(scheduler.next_execution_plan().storage[0])
        assert query2.keys == query.keys
        executor.submit(query2)
        wait()
        assert executor.snapshot().health.state == "healthy"
        read2 = job(scheduler.next_execution_plan().storage[0])
        assert read2.keys == read.keys
        executor.submit(read2)
        wait()
        plan = scheduler.next_execution_plan()
        assert list(plan.forward[0].extend_prefix_lens) == [6]
        assert host.host_buffer[:24] == b"abcdefgh" * 3
        assert scheduler.storage_negative_keys() == 0
        assert scheduler.storage_quarantined() == 0
    finally:
        executor.shutdown(timeout_s=2)
