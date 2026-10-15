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

"""Live Mooncake Store L3 tests against a real ``mooncake_master``.

The test boots ``mooncake_master`` (or reuses ``MOONCAKE_MASTER``) and drives
``MooncakeKvStore`` / ``L3HostStore`` over TCP with ``P2PHANDSHAKE``. Skip when
the Mooncake package is not installed, unless ``MOONCAKE_REQUIRE_MASTER=1``.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import unittest
from contextlib import closing

from tokenspeed.runtime.cache.l3.executor import L3HostStore
from tokenspeed.runtime.cache.l3.factory import create_kvstore_storage_backend
from tokenspeed.runtime.cache.l3.mooncake import MooncakeKvStore


def _require_or_skip(reason: str) -> None:
    flag = os.environ.get("MOONCAKE_REQUIRE_MASTER", "").strip().lower()
    if flag in {"1", "true", "yes"}:
        raise RuntimeError(reason)
    raise unittest.SkipTest(reason)


def _libcudart_dir() -> str | None:
    try:
        import nvidia.cu13 as cuda_runtime
    except ImportError:
        return None
    for location in getattr(cuda_runtime, "__path__", []):
        candidate = os.path.join(location, "lib", "libcudart.so.13")
        if os.path.isfile(candidate):
            return os.path.dirname(candidate)
    return None


def _ensure_libcudart() -> str | None:
    """Load ``libcudart`` into this process and export it for child binaries.

    ``tokenspeed-mooncake`` links against CUDA 13 even for Host-only TCP
    transfers. Changing
    ``LD_LIBRARY_PATH`` after Python starts does not affect this process's
    ``dlopen``, so preload via ``ctypes`` as well.
    """

    lib_dir = _libcudart_dir()
    if lib_dir is None:
        return None
    ctypes.CDLL(
        os.path.join(lib_dir, "libcudart.so.13"),
        mode=ctypes.RTLD_GLOBAL,
    )
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [part for part in current.split(os.pathsep) if part]
    if lib_dir not in parts:
        os.environ["LD_LIBRARY_PATH"] = lib_dir + (
            os.pathsep + current if current else ""
        )
    return lib_dir


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_tcp(host: str, port: int, *, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            with closing(socket.create_connection((host, port), timeout=0.2)):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.1)
    raise TimeoutError(f"Mooncake master {host}:{port} did not come up: {last_error}")


class _PtrBuffer:
    """CPU byte buffer with a ``data_ptr()`` for Mooncake ``register_buffer``."""

    def __init__(self, nbytes: int):
        self._nbytes = int(nbytes)
        self._raw = bytearray(self._nbytes)
        self._ctype = (ctypes.c_char * self._nbytes).from_buffer(self._raw)

    def data_ptr(self) -> int:
        return ctypes.addressof(self._ctype)

    def numel(self) -> int:
        return self._nbytes

    def element_size(self) -> int:
        return 1

    def __setitem__(self, key, value) -> None:
        self._raw[key] = bytes(value)

    def __getitem__(self, key):
        return self._raw[key]


class _Host:
    def __init__(self, *, nbytes: int, page: int):
        self.host_buffer = _PtrBuffer(nbytes)
        self._page = int(page)

    def host_block_range(self, group_index: int, block_id: int) -> tuple[int, int]:
        del group_index
        return (int(block_id) - 1) * self._page, self._page


class MooncakeMasterLiveTest(unittest.TestCase):
    master_proc: subprocess.Popen[bytes] | None = None
    master_log = None
    master_addr = ""
    started_master = False
    _lib_dir: str | None = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._lib_dir = _ensure_libcudart()
        try:
            import mooncake
            from mooncake.store import MooncakeDistributedStore  # noqa: F401
        except ImportError as exc:
            _require_or_skip(f"mooncake package is not installed: {exc}")

        existing = os.environ.get("MOONCAKE_MASTER", "").strip()
        if existing:
            host, _, port_text = existing.rpartition(":")
            _wait_tcp(host or "127.0.0.1", int(port_text), timeout_s=20.0)
            cls.master_addr = existing
            cls.started_master = False
            return

        package_bin = os.path.join(
            os.path.dirname(mooncake.__file__), "mooncake_master"
        )
        exe = (
            package_bin
            if os.path.isfile(package_bin)
            else shutil.which("mooncake_master")
        )
        if not exe or not os.path.isfile(exe):
            _require_or_skip("mooncake_master is not installed")
        port = _free_port()
        metrics_port = _free_port()
        log_dir = os.environ.get("TEST_MOONCAKE_LOG_DIR", "/tmp/mooncake-master-test")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"master-{port}.log")
        env = os.environ.copy()
        if cls._lib_dir:
            env["LD_LIBRARY_PATH"] = (
                cls._lib_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
            )
        cls.master_log = open(log_path, "wb")
        cls.master_proc = subprocess.Popen(
            [
                exe,
                f"--port={port}",
                f"--metrics_port={metrics_port}",
                "--enable_http_metadata_server=false",
                "--enable_metric_reporting=false",
                f"--log_dir={log_dir}",
            ],
            stdout=cls.master_log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        cls.started_master = True
        cls.master_addr = f"127.0.0.1:{port}"
        try:
            _wait_tcp("127.0.0.1", port, timeout_s=20.0)
        except TimeoutError:
            cls._stop_master()
            output = b""
            if os.path.isfile(log_path):
                with open(log_path, "rb") as log_file:
                    output = log_file.read(2000)
            _require_or_skip(
                f"mooncake_master failed to bind {cls.master_addr}: {output!r}"
            )

    @classmethod
    def _stop_master(cls) -> None:
        proc = cls.master_proc
        cls.master_proc = None
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    proc.kill()
                proc.wait(timeout=5)
        log_file = cls.master_log
        cls.master_log = None
        if log_file is not None:
            log_file.close()

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.started_master:
            cls._stop_master()

    def _extra_config(self) -> dict:
        return {
            "master_server_address": self.master_addr,
            "local_hostname": "localhost",
            "metadata_server": "P2PHANDSHAKE",
            "protocol": "tcp",
            "global_segment_size": 64 * 1024 * 1024,
            "device_name": "",
        }

    def test_global_objects_survive_async_writer_shutdown(self) -> None:
        from tokenspeed.runtime.cache.l3.async_executor import (
            AsyncL3Executor,
            StorageJob,
        )
        from tokenspeed.runtime.cache.l3.config import L3AsyncConfig

        owner_host = _Host(nbytes=4096, page=64)
        owner = MooncakeKvStore(
            self._extra_config(),
            host_buffer=owner_host.host_buffer,
            tp_size=1,
            pp_size=1,
        )
        self.addCleanup(owner.close)
        clients = dict(self._extra_config(), global_segment_size=0)
        config = L3AsyncConfig.from_extra_config(
            None, host_blocks=8, prefix_granularity=64
        )
        key = f"global-async-{time.monotonic_ns()}"
        payload = bytes(range(64))

        def client():
            host = _Host(nbytes=4096, page=64)
            backend = MooncakeKvStore(
                clients, host_buffer=host.host_buffer, tp_size=1, pp_size=1
            )
            return AsyncL3Executor(
                L3HostStore(
                    backend, host, key_prefix="global-async", rank=0, ownership=None
                ),
                config=config,
                prefix_for_weight_version=None,
            )

        def complete(executor):
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                results = executor.poll().completions
                if results:
                    self.assertEqual(len(results), 1)
                    self.assertTrue(results[0].retired)
                    self.assertEqual(results[0].success, (True,))
                    return
                time.sleep(0.01)
            self.fail("native asynchronous storage did not complete")

        writer = client()
        try:
            writer.store.host_storage.host_buffer[:64] = payload
            writer.submit(StorageJob(1, "put", (key,), (0,), (64,)))
            complete(writer)
        finally:
            writer.shutdown(timeout_s=10)
            writer.store.close()
        reader = client()
        try:
            reader.submit(StorageJob(1, "get", (key,), (128,), (64,)))
            complete(reader)
            self.assertEqual(
                bytes(reader.store.host_storage.host_buffer[128:192]), payload
            )
            self.assertTrue(reader.store.backend.remove_by_prefix(key))
        finally:
            reader.shutdown(timeout_s=10)
            reader.store.close()

    def test_l3_host_store_round_trips_through_mooncake_master(self) -> None:
        host = _Host(nbytes=4096, page=64)
        payload = b"mooncake-l3-bytes"
        host.host_buffer[0 : len(payload)] = payload
        backend = create_kvstore_storage_backend(
            "mooncake",
            json.dumps(self._extra_config()),
            host_buffer=host.host_buffer,
            tp_size=1,
            pp_size=1,
        )
        self.assertIsInstance(backend, MooncakeKvStore)
        l3 = L3HostStore(backend, host, key_prefix="live", rank=0, ownership=None)
        pages = [(0, 1, "h0", 0)]
        self.assertEqual(l3.backup(pages), [True])
        self.assertEqual(l3.exists(pages), [True])
        host.host_buffer[0 : len(payload)] = b"\x00" * len(payload)
        self.assertEqual(
            bytes(host.host_buffer[0 : len(payload)]), b"\x00" * len(payload)
        )
        self.assertEqual(l3.prefetch(pages), [True])
        self.assertEqual(bytes(host.host_buffer[0 : len(payload)]), payload)
        l3.rotate_namespace()
        self.assertEqual(l3.exists(pages), [False])
        # Exercise the deployed wheel's retirement contract, not just a mock
        # return code: eviction after EXISTS must be a safely retired miss.
        result = backend.batch_get_into_result(
            [l3.object_key("h0", 0, 0)], host.host_buffer, [0], [64]
        )
        self.assertEqual(result.success, (False,))
        self.assertTrue(result.retired)
        l3.close()

    def test_monitor_recovers_after_owned_master_restart(self) -> None:
        """An actual metadata outage opens the circuit; a new owner can serve it.

        Only the master created by this fixture may be stopped. This checks
        recovery of cache service, not persistence of the old in-memory objects.
        """
        from dataclasses import replace

        from tokenspeed.runtime.cache.l3.async_executor import (
            AsyncL3Executor,
            StorageJob,
        )
        from tokenspeed.runtime.cache.l3.config import L3AsyncConfig

        cls = type(self)
        if not cls.started_master:
            self.skipTest("restart testing requires a fixture-owned master")
        command = cls.master_proc.args
        restarted = False

        def restart():
            nonlocal restarted
            cls.master_log = open(f"/tmp/mooncake-restart-{os.getpid()}.log", "wb")
            cls.master_proc = subprocess.Popen(
                command,
                stdout=cls.master_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            host, _, port = cls.master_addr.rpartition(":")
            _wait_tcp(host, int(port), timeout_s=20)
            restarted = True

        def complete(executor):
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                results = executor.poll().completions
                if results:
                    self.assertEqual(len(results), 1)
                    self.assertTrue(results[0].retired)
                    return results[0]
                time.sleep(0.01)
            self.fail("native storage operation timed out")

        host = _Host(nbytes=4096, page=64)
        config = dict(self._extra_config(), global_segment_size=0)
        backend = MooncakeKvStore(
            config, host_buffer=host.host_buffer, tp_size=1, pp_size=1
        )
        executor = AsyncL3Executor(
            L3HostStore(
                backend, host, key_prefix="master-restart", rank=0, ownership=None
            ),
            config=replace(
                L3AsyncConfig.from_extra_config(
                    None, host_blocks=8, prefix_granularity=64
                ),
                recovery_interval_ms=50,
            ),
            prefix_for_weight_version=None,
        )
        owner = None
        key = f"recovered-{time.monotonic_ns()}"
        try:
            cls._stop_master()
            executor.submit(StorageJob(1, "query", (key,), (), ()))
            self.assertEqual(complete(executor).success, (False,))
            self.assertEqual(executor.snapshot().health.state, "open")
            restart()
            owner_host = _Host(nbytes=4096, page=64)
            owner = MooncakeKvStore(
                self._extra_config(),
                host_buffer=owner_host.host_buffer,
                tp_size=1,
                pp_size=1,
            )
            time.sleep(0.06)
            executor.submit(StorageJob(2, "query", (key,), (), ()))
            self.assertEqual(complete(executor).success, (False,))
            self.assertEqual(executor.snapshot().health.state, "healthy")
            payload = bytes(range(64))
            host.host_buffer[:64] = payload
            executor.submit(StorageJob(3, "put", (key,), (0,), (64,)))
            self.assertEqual(complete(executor).success, (True,))
            host.host_buffer[:64] = bytes(64)
            executor.submit(StorageJob(4, "get", (key,), (128,), (64,)))
            self.assertEqual(complete(executor).success, (True,))
            self.assertEqual(bytes(host.host_buffer[128:192]), payload)
            self.assertTrue(backend.remove_by_prefix(key))
        finally:
            if not restarted:
                restart()
            executor.shutdown(timeout_s=10)
            backend.close()
            if owner is not None:
                owner.close()


if __name__ == "__main__":
    unittest.main()
