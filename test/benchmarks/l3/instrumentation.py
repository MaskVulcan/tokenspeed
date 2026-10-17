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

"""Opt-in benchmark observations and idle-only local cache reset.

Never imported by runtime code. The benchmark launcher installs this in an
isolated candidate. Local reset does not invoke the production namespace purge.
"""

from __future__ import annotations

import functools
import json
import os
import re
import threading
import time
from dataclasses import asdict
from pathlib import Path

import psutil
import torch.distributed as dist


class Observer:
    def __init__(self, *, control: Path, output: Path, role: str):
        self.control = control
        self.output = output
        self.role = role
        self.lock = threading.Lock()
        self.client = None
        self.owned = set()
        self.counters = {}
        self.spans = []
        self.seq = 0
        self.next_poll = 0.0
        self.next_sample = 0.0
        self.settings = {}
        self.profile_active = False
        self.output.mkdir(parents=True, exist_ok=True)

    def native(self, function, kind):
        @functools.wraps(function)
        def call(client, *args, **kwargs):
            begin = time.time_ns()
            results = function(client, *args, **kwargs)
            end = time.time_ns()
            with self.lock:
                self.client = client
                row = self.counters.setdefault(
                    kind,
                    dict(
                        calls=0, keys=0, success_keys=0, bytes=0, seconds=0.0, errors=0
                    ),
                )
                row["calls"] += 1
                row["keys"] += len(results)
                row["seconds"] += (end - begin) / 1e9
                if kind == "get":
                    row["success_keys"] += sum(
                        int(code) == size for code, size in zip(results, args[2])
                    )
                    row["bytes"] += sum(max(0, int(code)) for code in results)
                    row["errors"] += sum(int(code) < 0 for code in results)
                elif kind == "put":
                    row["success_keys"] += sum(
                        int(code) in (0, -705) for code in results
                    )
                    row["bytes"] += sum(
                        size for size, code in zip(args[2], results) if int(code) == 0
                    )
                    row["errors"] += sum(int(code) not in (0, -705) for code in results)
                    # Some SDK releases return 0 for an existing object too.
                    # These bytes are successful submitted payload, not wire
                    # traffic or proof of a newly created object.
                    successful = [
                        key for key, code in zip(args[0], results) if int(code) == 0
                    ]
                    self.owned.update(successful)
                    if successful:
                        with (self.output / f"owned-{os.getpid()}.jsonl").open(
                            "a"
                        ) as f:
                            f.write(json.dumps(successful) + "\n")
                else:
                    row["success_keys"] += sum(int(code) == 1 for code in results)
                    row["errors"] += sum(int(code) not in (0, 1) for code in results)
                if self.settings.get("spans") and len(self.spans) < 8192:
                    self.spans.append(dict(kind=kind, start_ns=begin, end_ns=end))
            return results

        return call

    def delayed(self, function, kind):
        @functools.wraps(function)
        def call(*args, **kwargs):
            with self.lock:
                settings = dict(self.settings)
            delay = settings.get(f"delay_{kind}_s", 0)
            rank = dist.get_rank() if dist.is_initialized() else 0
            if delay and rank == settings.get("delay_rank", 0):
                time.sleep(delay)
            return function(*args, **kwargs)

        return call

    def state(self, hooks):
        s = hooks._scheduler
        stats = s.storage_stats()
        rss = psutil.Process().memory_info().rss
        pss = None
        rollup = Path("/proc/self/smaps_rollup")
        if rollup.exists():
            for line in rollup.read_text().splitlines():
                if line.startswith("Pss:"):
                    pss = int(line.split()[1]) * 1024
        return dict(
            executor=asdict(hooks._l3.snapshot()) if hooks._l3 is not None else None,
            namespace=hooks._l3.store.key_prefix if hooks._l3 is not None else None,
            pending=s.storage_pending(),
            held=stats.held_blocks,
            quarantine=stats.quarantined,
            candidates=stats.backup_candidates,
            waiting=stats.waiting_requests,
            ready=stats.ready_requests,
            gate=stats.pressure_gate_closed,
            bypasses=stats.pressure_bypasses,
            dropped=stats.pressure_dropped_backup_keys,
            backup_drops=dict(
                pressure=stats.pressure_dropped_backup_keys,
                overflow=stats.overflow_dropped_backup_keys,
                host_unavailable=stats.unavailable_dropped_backup_keys,
                write_disabled=stats.disabled_dropped_backup_keys,
                cleanup=stats.cleanup_dropped_backup_keys,
            ),
            timeouts=stats.wait_timeouts,
            host_cached=s.host_pool_cached_blocks(),
            host_pinned=s.host_pool_pinned_blocks(),
            rss=rss,
            pss=pss,
        )

    def poll(self, hooks):
        now = time.monotonic()
        if now < self.next_poll:
            return
        self.next_poll = now + 0.1
        command = json.loads(self.control.read_text())
        with self.lock:
            self.settings = command.get("settings", {})
        seq = command.get("seq", 0)
        rank = dist.get_rank() if dist.is_initialized() else 0
        tag = f"{self.role}-{rank}"
        if self.settings.get("sample") and now >= self.next_sample:
            self.next_sample = now + 1
            row = dict(time=time.time(), **self.state(hooks))
            with (self.output / f"{tag}-samples.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
        if seq == self.seq:
            return
        action = command.get("action", "snapshot")
        s = hooks._scheduler
        if action != "observe":
            state = self.state(hooks)
            if state["quarantine"]:
                self.reply(
                    tag, dict(seq=seq, error="native retirement unproven", state=state)
                )
                return
            if not s.can_clear_cache() or state["pending"] or state["candidates"]:
                self.reply(tag, dict(seq=seq, ready=False, state=state))
                return
            if hooks._l3 is not None and not hooks._l3.is_idle():
                return
        if action == "clear_local":
            assert s.clear_cache(), "local cache reset was refused"
        elif action == "clear_device":
            assert s.clear_l1_cache(), "device cache reset was refused"
        elif action == "purge_owned":
            with self.lock:
                keys = sorted(self.owned)
            if keys:
                assert self.client is not None
                assert hooks._l3 is not None
                prefix = f"{hooks._l3.store.key_prefix}_"
                assert all(
                    key.startswith(prefix) for key in keys
                ), "Cleanup key is outside the isolated benchmark namespace"
                pattern = "^(?:" + "|".join(re.escape(key) for key in keys) + ")$"
                removed = self.client.remove_by_regex(pattern, True)
                assert removed >= 0, f"Native cleanup failed: {removed}"
                exists = self.client.batch_is_exist(keys)
                assert len(exists) == len(keys) and all(
                    int(code) in (0, 1) for code in exists
                )
                # Force still skips replicas while the store offloads them.
                # Retry on a later idle poll, outside all measured requests.
                remaining = {key for key, code in zip(keys, exists) if int(code) == 1}
                with self.lock:
                    self.owned.difference_update(set(keys) - remaining)
                if remaining:
                    self.reply(
                        tag,
                        dict(seq=seq, ready=False, cleanup_remaining=len(remaining)),
                    )
                    return
            (self.output / f"{tag}-purged-{seq}.json").write_text(json.dumps(keys))
        elif action == "reset_counters":
            with self.lock:
                self.counters.clear()
                self.spans.clear()
        elif action not in ("snapshot", "observe"):
            raise ValueError(f"Unknown benchmark action {action}")
        state = self.state(hooks)
        with self.lock:
            result = dict(
                seq=seq,
                ready=True,
                pid=os.getpid(),
                role=self.role,
                rank=rank,
                time=time.time(),
                state=state,
                counters={kind: dict(row) for kind, row in self.counters.items()},
                spans=list(self.spans),
                owned_keys=len(self.owned),
            )
        self.reply(tag, result)
        self.seq = seq

    def reply(self, tag, data):
        temporary = self.output / f"{tag}.new"
        temporary.write_text(json.dumps(data))
        temporary.replace(self.output / f"{tag}.json")


def install(*, control: Path, output: Path, role: str) -> Observer:
    from mooncake.store import MooncakeDistributedStore

    from tokenspeed.runtime.cache.l3.mooncake import MooncakeKvStore
    from tokenspeed.runtime.engine.l3_cache_hooks import L3CacheHooks

    observer = Observer(control=control, output=output, role=role)
    for name, kind in (
        ("batch_get_into", "get"),
        ("batch_put_from", "put"),
        ("batch_is_exist", "query"),
    ):
        method = getattr(MooncakeDistributedStore, name)
        setattr(MooncakeDistributedStore, name, observer.native(method, kind))
    MooncakeKvStore.batch_get_into_result = observer.delayed(
        MooncakeKvStore.batch_get_into_result, "get"
    )
    MooncakeKvStore.batch_put_from_result = observer.delayed(
        MooncakeKvStore.batch_put_from_result, "put"
    )
    original = L3CacheHooks.record_metrics

    @functools.wraps(original)
    def record(hooks):
        original(hooks)
        observer.poll(hooks)

    L3CacheHooks.record_metrics = record
    return observer


def install_profile(observer: Observer) -> None:
    import nvtx
    import torch
    from mooncake.store import MooncakeDistributedStore

    from tokenspeed.runtime.cache.l2.executor import HostCacheExecutor
    from tokenspeed.runtime.execution.model_executor import ModelExecutor

    original = ModelExecutor.execute_forward_op

    @functools.wraps(original)
    def forward(*args, **kwargs):
        with observer.lock:
            enabled = observer.settings.get("profile", False)
        if dist.get_rank() == 0 and enabled != observer.profile_active:
            if enabled:
                torch.cuda.cudart().cudaProfilerStart()
            else:
                torch.cuda.cudart().cudaProfilerStop()
            observer.profile_active = enabled
        with nvtx.annotate("model_forward_launch"):
            return original(*args, **kwargs)

    ModelExecutor.execute_forward_op = forward

    # NVTX is a CPU annotation; GPU profiler APIs above run on ForwardThread.
    def mark(function, name):
        @functools.wraps(function)
        def call(*args, **kwargs):
            with nvtx.annotate(name):
                return function(*args, **kwargs)

        return call

    MooncakeDistributedStore.batch_get_into = mark(
        MooncakeDistributedStore.batch_get_into, "l3_native_get"
    )
    MooncakeDistributedStore.batch_put_from = mark(
        MooncakeDistributedStore.batch_put_from, "l3_native_put"
    )
    HostCacheExecutor._start_loading = mark(
        HostCacheExecutor._start_loading, "l2_h2d_launch"
    )
    HostCacheExecutor._start_writing = mark(
        HostCacheExecutor._start_writing, "l2_d2h_launch"
    )
