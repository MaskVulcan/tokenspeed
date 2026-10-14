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

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tokenspeed.runtime.cache.host_memory import (
    HostMemoryRequest,
    MemoryConstraint,
    agree_host_memory,
    cgroup_constraints,
    validate_host_memory,
)
from tokenspeed.runtime.cache.l3.mooncake import mooncake_host_memory


def request(pid, *, node, available, planned):
    return HostMemoryRequest(
        node,
        pid,
        1,
        (("host_cache_arena", planned),),
        (MemoryConstraint("physical", available),),
    )


def test_aggregate_ranks_fail_before_individually_fitting_allocations():
    ranks = [request(i, node="a", available=100, planned=20) for i in range(8)]
    for rank in ranks:
        validate_host_memory([rank], headroom_bytes=10)
    with pytest.raises(ValueError, match="aggregate Host memory"):
        validate_host_memory(ranks, headroom_bytes=10)


def test_nodes_and_cgroup_ancestors_have_separate_budgets():
    a = request(1, node="a", available=100, planned=20)
    b = request(2, node="b", available=100, planned=20)
    reports = validate_host_memory([a, b], headroom_bytes=10)
    assert [x["remaining_bytes"] for x in reports] == [70, 70]
    a = replace(a, constraints=(*a.constraints, MemoryConstraint("parent", 35)))
    peer = replace(a, pid=3)
    with pytest.raises(ValueError, match="parent"):
        validate_host_memory([a, peer, b], headroom_bytes=0)


def test_shared_arena_is_counted_once_with_sdk_buffers():
    memory = mooncake_host_memory(
        {"master_server_address": "test", "global_segment_size": 1024},
        tp_size=4,
        pp_size=2,
    )
    assert dict(memory) == {"mooncake_segment": 128, "mooncake_staging": 16 * 1024**2}
    client = mooncake_host_memory(
        {"master_server_address": "test", "global_segment_size": 0},
        tp_size=8,
        pp_size=1,
    )
    assert dict(client)["mooncake_segment"] == 0
    rank = HostMemoryRequest(
        "n",
        1,
        1,
        (("host_cache_arena", 1024), *client),
        (MemoryConstraint("physical", 10**9),),
    )
    report = validate_host_memory([rank], headroom_bytes=100)[0]
    assert report["planned_bytes"] == 1024 + 16 * 1024**2
    assert report["headroom_bytes"] == 100


def test_cgroup_v2_uses_visible_parent_limit(tmp_path):
    root = tmp_path / "cg"
    child = root / "user" / "job"
    child.mkdir(parents=True)
    for directory, limit, used in [
        (root, "max", 0),
        (root / "user", 100, 80),
        (child, 100, 40),
    ]:
        (directory / "memory.max").write_text(str(limit))
        (directory / "memory.current").write_text(str(used))
    constraints = cgroup_constraints(
        membership="0::/user/job", mountinfo=f"1 0 0:1 / {root} rw - cgroup2 cgroup rw"
    )
    assert sorted(x.available_bytes for x in constraints) == [20, 60]
    req = HostMemoryRequest("n", 1, 1, (("host_cache_arena", 25),), constraints)
    with pytest.raises(ValueError, match="aggregate Host memory"):
        validate_host_memory([req], headroom_bytes=0)


def test_pid_namespaces_share_node_budget_without_owner_collision():
    a = request(1, node="a", available=100, planned=20)
    b = replace(a, pid_namespace=2)
    assert validate_host_memory([a, b], headroom_bytes=10)[0]["planned_bytes"] == 40
    with pytest.raises(ValueError, match="Duplicate Host memory owner"):
        validate_host_memory([a, a], headroom_bytes=10)


def test_disabled_child_controller_still_checks_limited_ancestor(tmp_path):
    child = tmp_path / "job"
    child.mkdir()
    (tmp_path / "memory.max").write_text("100")
    (tmp_path / "memory.current").write_text("80")
    values = cgroup_constraints(
        membership="0::/job", mountinfo=f"1 0 0:1 / {tmp_path} rw - cgroup2 cgroup rw"
    )
    assert [v.available_bytes for v in values] == [20]


def test_cgroup_outside_namespace_cannot_escape_visible_mount(tmp_path):
    with pytest.raises(RuntimeError, match="outside the visible mount"):
        cgroup_constraints(
            membership="0::/../job",
            mountinfo=f"1 0 0:1 / {tmp_path} rw - cgroup2 cgroup rw",
        )


def test_cgroup_v1_mount_root_and_escaped_space(tmp_path):
    root = tmp_path / "memory space"
    root.mkdir()
    (root / "memory.limit_in_bytes").write_text("100")
    (root / "memory.usage_in_bytes").write_text("70")
    mount = str(root).replace(" ", r"\040")
    values = cgroup_constraints(
        membership="4:memory:/container/a",
        mountinfo=f"1 0 0:1 /container/a {mount} rw - cgroup cgroup rw,memory",
    )
    assert [v.available_bytes for v in values] == [30]


def test_sampling_failure_joins_agreement_before_raising():
    gathered = []

    def components():
        raise OSError("unreadable cgroup")

    def gather(local):
        gathered.append(local)
        return [local, {"request": None, "error": "peer sampling failure"}]

    with pytest.raises(RuntimeError, match="peer sampling failure"):
        agree_host_memory(components, headroom_bytes=0, gather=gather)
    assert gathered[0]["request"] is None
    assert "unreadable cgroup" in gathered[0]["error"]


def _memory_worker(rank, path, sampling_error):
    from datetime import timedelta
    from unittest.mock import patch

    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    try:

        def gather(local):
            rows = [None] * 2
            dist.all_gather_object(rows, local)
            return rows

        def components():
            if sampling_error and rank == 1:
                raise OSError("rank-local cgroup failure")
            return (("arena", 20),)

        sample = request(rank + 1, node="same-host", available=30, planned=20)
        with patch(
            "tokenspeed.runtime.cache.host_memory.sample_host_memory",
            return_value=sample,
        ):
            with pytest.raises(RuntimeError if sampling_error else ValueError):
                agree_host_memory(components, headroom_bytes=0, gather=gather)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("sampling_error", [False, True])
def test_real_ranks_converge_budget_and_sampling_failures(
    tmp_path, monkeypatch, sampling_error
):
    import torch.multiprocessing as mp

    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    mp.spawn(
        _memory_worker,
        args=(str(tmp_path / "memory-rendezvous"), sampling_error),
        nprocs=2,
        join=True,
    )
