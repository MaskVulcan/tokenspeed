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

from __future__ import annotations

import tokenspeed_scheduler as ts
from conftest import _advance, _spec


def scheduler(*, groups=1, host=32, timeout=100):
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = 2
    cfg.num_snapshot_pages = 1
    cfg.num_device_pages = 64
    cfg.num_host_pages = host
    cfg.max_scheduled_tokens = 64
    cfg.max_batch_size = 8
    cfg.enable_l3_storage = True
    cfg.storage_config = ts.StorageConfig(
        8,
        32,
        max(1, (host - 1) // 2),
        4,
        32,
        2,
        timeout,
        100,
        ts.StoragePressure(8, 1048576, 1000),
        ts.StoragePressure(4, 524288, 500),
    )
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id=f"g{i}",
            block_granularity=2,
            total_pages=64,
            attn_kind=ts.AttnKind.Full,
        )
        for i in range(groups)
    ]
    result = ts.Scheduler(cfg)
    tick(result, 1, forward=True, storage=True)
    return result


def event(s, item):
    e = ts.ExecutionEvent()
    e.add_event(item)
    s.advance(e)


def tick(s, now, *, forward, storage):
    event(s, ts.Cache.StorageTick(now, forward, storage, ts.StoragePressure(0, 0, 0)))


def done(s, op, flags, *, retired):
    event(
        s,
        ts.Cache.StorageDone(
            op.op_id,
            flags,
            retired,
            [
                int(not flag and retired and op.kind == ts.StorageKind.Get)
                for flag in flags
            ],
        ),
    )


def requests(plan):
    return [rid for batch in plan.forward for rid in batch.request_ids]


def read_pending(s, rid="r"):
    s.submit_requests([_spec(rid, list(range(8)))])
    q = s.next_execution_plan().storage[0]
    assert q.kind == ts.StorageKind.Query
    done(s, q, [1] * len(q.keys), retired=True)
    read = s.next_execution_plan().storage[0]
    assert read.kind == ts.StorageKind.Get
    return read


def test_get_holds_only_host_then_uses_normal_loadback():
    s = scheduler()
    before = s.available_lcm_blocks()
    read = read_pending(s)
    assert s.available_lcm_blocks() == before
    assert s.storage_held_blocks() == len(read.keys)
    done(s, read, [1] * len(read.keys), retired=True)
    plan = s.next_execution_plan()
    assert requests(plan) == ["r"]
    assert list(plan.forward[0].extend_prefix_lens) == [6]
    load = next(op for op in plan.cache if isinstance(op, ts.Cache.LoadBackOp))
    assert sum(len(row) for row in load.src_pages) == len(read.keys)
    assert s.storage_held_blocks() == 0
    assert s.host_pool_pinned_blocks() > 0  # ordinary H2D owns the handoff
    event(s, ts.Cache.LoadBackDoneEvent(load.op_ids[0]))


def test_soft_timeout_does_not_free_native_destination():
    s = scheduler()
    read = read_pending(s)
    held = s.storage_held_blocks()
    tick(s, 102, forward=True, storage=True)
    plan = s.next_execution_plan()
    assert requests(plan) == ["r"]
    assert list(plan.forward[0].extend_prefix_lens) == [0]
    assert any(op.kind == ts.StorageKind.Cancel for op in plan.storage)
    assert s.storage_held_blocks() == held
    done(s, read, [1] * len(read.keys), retired=True)
    assert s.storage_held_blocks() == 0
    assert (
        s.host_pool_cached_blocks() == 0
    )  # detached data never resurrects the request


def test_unknown_retirement_quarantines_instead_of_reusing_blocks():
    s = scheduler()
    read = read_pending(s)
    done(s, read, [0] * len(read.keys), retired=False)
    plan = s.next_execution_plan()
    assert requests(plan) == ["r"]
    assert s.storage_quarantined() == 1
    assert s.storage_held_blocks() == len(read.keys)
    assert not s.can_clear_cache()


def test_short_read_is_not_published_and_negative_metadata_is_bounded():
    s = scheduler(groups=2)
    read = read_pending(s)
    done(s, read, [0] * len(read.keys), retired=True)
    assert s.storage_negative_keys() == 4
    plan = s.next_execution_plan()
    assert requests(plan) == ["r"]
    assert list(plan.forward[0].extend_prefix_lens) == [0]
    assert s.storage_held_blocks() == 0


def test_later_query_can_progress_while_earlier_query_is_pending():
    s = scheduler()
    s.submit_requests([_spec("a", list(range(8))), _spec("b", list(range(10, 18)))])
    queries = s.next_execution_plan().storage
    assert len(queries) == 2
    done(s, queries[1], [0] * len(queries[1].keys), retired=True)
    assert requests(s.next_execution_plan()) == ["b"]
    assert s.storage_pending()  # a is still querying


def test_zero_logprob_bound_never_queries_storage():
    s = scheduler()
    spec = _spec("r", list(range(8)))
    spec.max_cached_prefix_tokens = 0
    s.submit_requests([spec])
    plan = s.next_execution_plan()
    assert requests(plan) == ["r"] and not plan.storage


def test_ready_result_is_protected_while_forward_is_frozen():
    s = scheduler()
    read = read_pending(s)
    tick(s, 2, forward=False, storage=False)
    done(s, read, [1] * len(read.keys), retired=True)
    plan = s.next_execution_plan()
    assert not requests(plan)
    assert s.storage_held_blocks() > 0
    tick(s, 103, forward=False, storage=False)
    assert not requests(s.next_execution_plan())
    assert s.storage_held_blocks() == 0
    tick(s, 104, forward=True, storage=True)
    plan = s.next_execution_plan()
    assert requests(plan) == ["r"] and not plan.storage


def test_d2h_publishes_before_put_and_freeze_drops_unowned_candidates():
    s = scheduler()
    s.submit_requests([_spec("r", list(range(8)))])
    q = s.next_execution_plan().storage[0]
    done(s, q, [0] * len(q.keys), retired=True)
    s.next_execution_plan()
    _advance(s, "r", [99])
    plan = s.next_execution_plan()
    wb = next(op for op in plan.cache if isinstance(op, ts.Cache.WriteBackOp))
    ack = ts.Cache.WriteBackDoneEvent()
    ack.op_id = wb.op_ids[0]
    event(s, ack)
    assert s.host_pool_cached_blocks() > 0
    assert s.storage_held_blocks() == 0
    candidates = s.storage_stats().backup_candidates
    assert candidates > 0
    tick(s, 3, forward=False, storage=False)
    frozen = s.next_execution_plan()
    assert not requests(frozen) and not frozen.storage
    assert s.storage_held_blocks() == 0
    assert s.storage_stats().backup_candidates == 0
    assert s.storage_stats().cleanup_dropped_backup_keys == candidates


def test_query_credits_never_split_a_prefix_boundary_across_groups():
    s = scheduler(groups=6)
    s.submit_requests([_spec("bounded", list(range(16)))])
    op = s.next_execution_plan().storage[0]
    assert op.kind == ts.StorageKind.Query
    # 32 keys fit five whole six-group boundaries. A partial sixth boundary
    # cannot establish a usable prefix and must not consume query credits.
    assert len(op.keys) == 30
    for group in range(6):
        assert sum(key[0] == group for key in op.keys) == 5


def test_storage_statistics_separate_request_expiry_from_native_retirement():
    s = scheduler()
    read = read_pending(s)
    stats = s.storage_stats()
    assert stats.waiting_requests == stats.operations == 1
    assert stats.held_blocks == len(read.keys)
    tick(s, 102, forward=True, storage=True)
    stats = s.storage_stats()
    assert stats.wait_timeouts == 1 and stats.ready_expirations == 0
    assert stats.waiting_requests == 0 and stats.operations == 1
    assert stats.held_blocks == len(read.keys)
    tick(s, 103, forward=True, storage=True)
    assert s.storage_stats().wait_timeouts == 1
    done(s, read, [1] * len(read.keys), retired=True)
    assert s.storage_stats().operations == s.storage_stats().held_blocks == 0


def test_ready_expiration_and_detach_have_distinct_statistics():
    s = scheduler()
    read = read_pending(s)
    tick(s, 2, forward=False, storage=False)
    done(s, read, [1] * len(read.keys), retired=True)
    s.next_execution_plan()
    assert s.storage_stats().ready_requests == 1
    tick(s, 103, forward=False, storage=False)
    stats = s.storage_stats()
    assert stats.wait_timeouts == 0 and stats.ready_expirations == 1
    assert stats.operations == stats.ready_requests == stats.held_blocks == 0

    s = scheduler()
    read = read_pending(s)
    event(s, ts.Cache.StorageDetach("r"))
    done(s, read, [0] * len(read.keys), retired=False)
    stats = s.storage_stats()
    assert stats.wait_timeouts == stats.ready_expirations == 0
    assert stats.quarantined == stats.operations == 1
    assert stats.held_blocks == len(read.keys)


def pressure_tick(s, now, *, queued, byte_count, age):
    event(
        s,
        ts.Cache.StorageTick(
            now, True, True, ts.StoragePressure(queued, byte_count, age)
        ),
    )


def test_pressure_bypasses_new_requests_and_reopens_only_below_low_watermarks():
    s = scheduler(timeout=10000)
    pressure_tick(s, 2, queued=8, byte_count=0, age=0)
    s.submit_requests([_spec("bypass", list(range(8)))])
    plan = s.next_execution_plan()
    assert requests(plan) == ["bypass"]
    assert not plan.storage and s.storage_stats().pressure_bypasses == 1
    assert s.storage_stats().pressure_gate_closed
    pressure_tick(s, 3, queued=5, byte_count=0, age=0)
    assert s.storage_stats().pressure_gate_closed
    pressure_tick(s, 4, queued=4, byte_count=524288, age=500)
    assert not s.storage_stats().pressure_gate_closed
    s.submit_requests([_spec("recovered", list(range(40, 48)))])
    plan = s.next_execution_plan()
    assert any(op.kind == ts.StorageKind.Query for op in plan.storage)


def test_pressure_detaches_query_without_waiting_for_deadline_or_issuing_get():
    for completed in (False, True):
        s = scheduler(timeout=10000)
        s.submit_requests([_spec("r", list(range(8)))])
        query = s.next_execution_plan().storage[0]
        pressure_tick(s, 2, queued=0, byte_count=0, age=1000)
        if completed:
            done(s, query, [1] * len(query.keys), retired=True)
        plan = s.next_execution_plan()
        assert requests(plan) == ["r"]
        assert not any(
            op.kind in (ts.StorageKind.Query, ts.StorageKind.Get) for op in plan.storage
        )
        assert s.storage_stats().pressure_bypasses == 1
        assert s.storage_stats().wait_timeouts == 0
        assert s.storage_held_blocks() == 0
        if not completed:
            assert any(op.kind == ts.StorageKind.Cancel for op in plan.storage)
            done(s, query, [1] * len(query.keys), retired=True)
        assert not s.storage_pending()


def test_active_get_and_ready_data_progress_while_pressure_blocks_new_work():
    s = scheduler(timeout=10000)
    read = read_pending(s)
    held = s.storage_held_blocks()
    pressure_tick(s, 2, queued=8, byte_count=0, age=0)
    s.submit_requests([_spec("other", list(range(40, 48)))])
    plan = s.next_execution_plan()
    assert requests(plan) == ["other"]
    assert not plan.storage
    assert s.storage_held_blocks() == held
    done(s, read, [1] * len(read.keys), retired=True)
    plan = s.next_execution_plan()
    assert "r" in requests(plan)
    restored = next(batch for batch in plan.forward if "r" in batch.request_ids)
    assert restored.extend_prefix_lens[list(restored.request_ids).index("r")] == 6
    assert any(isinstance(op, ts.Cache.LoadBackOp) for op in plan.cache)
    assert s.storage_held_blocks() == 0
    assert s.storage_stats().pressure_gate_closed


def test_pressure_does_not_pause_soft_timeout_or_release_detached_native_get():
    s = scheduler(timeout=100)
    read = read_pending(s)
    held = s.storage_held_blocks()
    pressure_tick(s, 2, queued=8, byte_count=0, age=0)
    pressure_tick(s, 102, queued=8, byte_count=0, age=1000)
    plan = s.next_execution_plan()
    assert requests(plan) == ["r"]
    assert s.storage_held_blocks() == held
    assert s.storage_stats().wait_timeouts == 1
    done(s, read, [1] * len(read.keys), retired=True)
    assert s.storage_held_blocks() == 0
    assert s.host_pool_cached_blocks() == 0
