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

"""Startup-only Host allocation accounting across ranks and visible cgroups."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import psutil

HOST_MEMORY_HEADROOM_BYTES = 10 * 1024**3


@dataclass(frozen=True)
class MemoryConstraint:
    domain: str
    available_bytes: int


@dataclass(frozen=True)
class HostMemoryRequest:
    node: str
    pid: int
    pid_namespace: int
    components: tuple[tuple[str, int], ...]
    constraints: tuple[MemoryConstraint, ...]


def _unescape_mount(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)


def cgroup_constraints(
    *, membership: str, mountinfo: str
) -> tuple[MemoryConstraint, ...]:
    """Read the process's visible v1/v2 memory limits, including ancestors.

    Other allocations are already charged in current usage. Missing controllers
    impose no local limit; unreadable configured limits fail the preflight.
    """
    groups = {}
    for row in membership.splitlines():
        _, controllers, path = row.split(":", 2)
        groups[controllers] = path
    result = {}
    for row in mountinfo.splitlines():
        before, sep, after = row.partition(" - ")
        if not sep:
            continue
        fields, tail = before.split(), after.split()
        kind = tail[0]
        if kind == "cgroup2" and "" in groups:
            group_path, limit_file, usage_file = (
                groups[""],
                "memory.max",
                "memory.current",
            )
        elif kind == "cgroup" and "memory" in tail[2].split(","):
            matches = [
                path for names, path in groups.items() if "memory" in names.split(",")
            ]
            if not matches:
                continue
            group_path, limit_file, usage_file = (
                matches[0],
                "memory.limit_in_bytes",
                "memory.usage_in_bytes",
            )
        else:
            continue
        mount_root = Path(_unescape_mount(fields[3]))
        mountpoint = Path(_unescape_mount(fields[4]))
        group = Path(group_path)
        if group.is_relative_to(mount_root):
            relative = group.relative_to(mount_root)
        elif group == Path("/"):
            relative = Path(".")  # A cgroup namespace exposes its root as /.
        else:
            continue
        if ".." in relative.parts:
            raise RuntimeError("Host memory cgroup lies outside the visible mount")
        current = mountpoint / relative
        while True:
            limit_path = current / limit_file
            # A subtree may leave the memory controller disabled; ancestors
            # can still account for it and impose a shared limit.
            try:
                raw = limit_path.read_text().strip()
            except FileNotFoundError:
                pass
            else:
                if raw != "max":
                    limit = int(raw)
                    usage = int((current / usage_file).read_text().strip())
                    stat = current.stat()
                    domain = f"{kind}:{stat.st_dev}:{stat.st_ino}"
                    result[domain] = MemoryConstraint(domain, max(0, limit - usage))
            if current == mountpoint:
                break
            current = current.parent
    return tuple(result.values())


def sample_host_memory(components: Sequence[tuple[str, int]]) -> HostMemoryRequest:
    """Describe future allocations; resident model/state memory is already used."""
    memory = psutil.virtual_memory()
    boot_id = Path("/proc/sys/kernel/random/boot_id")
    node = boot_id.read_text().strip() if boot_id.exists() else socket.gethostname()
    membership, mounts = Path("/proc/self/cgroup"), Path("/proc/self/mountinfo")
    constraints = [MemoryConstraint("physical", int(memory.available))]
    if membership.exists() and mounts.exists():
        constraints.extend(
            cgroup_constraints(
                membership=membership.read_text(), mountinfo=mounts.read_text()
            )
        )
    return HostMemoryRequest(
        node,
        os.getpid(),
        Path("/proc/self/ns/pid").stat().st_ino,
        tuple(components),
        tuple(constraints),
    )


def validate_host_memory(
    requests: Sequence[HostMemoryRequest], *, headroom_bytes: int
) -> list[dict]:
    """Check aggregate future bytes against each node and cgroup's free budget.

    A shared Host arena is one component, not one allocation per L2/L3 user.
    Headroom is reserved once per memory domain. Independent engine worlds
    need a combined deployment preflight before concurrent startup.
    """
    if headroom_bytes < 0 or not requests:
        raise ValueError(
            "Host memory validation needs requests and nonnegative headroom"
        )
    domains = {}
    seen = set()
    for request in requests:
        identity = (request.node, request.pid_namespace, request.pid)
        if identity in seen:
            raise ValueError(f"Duplicate Host memory owner {identity}")
        seen.add(identity)
        if len(dict(request.components)) != len(request.components) or any(
            type(size) is not int or size < 0 for _, size in request.components
        ):
            raise ValueError(
                "Host memory components must be unique nonnegative byte counts"
            )
        if not request.constraints or len(
            {c.domain for c in request.constraints}
        ) != len(request.constraints):
            raise ValueError("Host memory constraints must be present and unique")
        for constraint in request.constraints:
            if constraint.available_bytes < 0:
                raise ValueError("Host memory availability cannot be negative")
            key = (request.node, constraint.domain)
            if key not in domains:
                domains[key] = dict(
                    node=request.node,
                    domain=constraint.domain,
                    available_bytes=constraint.available_bytes,
                    planned_bytes=0,
                    components={},
                    processes=0,
                )
            domain = domains[key]
            domain["available_bytes"] = min(
                domain["available_bytes"], constraint.available_bytes
            )
            domain["planned_bytes"] += sum(size for _, size in request.components)
            domain["processes"] += 1
            for name, size in request.components:
                domain["components"][name] = domain["components"].get(name, 0) + size
    reports = list(domains.values())
    for report in reports:
        report["headroom_bytes"] = headroom_bytes
        report["remaining_bytes"] = (
            report["available_bytes"] - report["planned_bytes"] - headroom_bytes
        )
    failures = [row for row in reports if row["remaining_bytes"] < 0]
    if failures:
        raise ValueError(
            f"Insufficient aggregate Host memory: {json.dumps(failures, sort_keys=True)}"
        )
    return reports


def agree_host_memory(
    components: Callable[[], Sequence[tuple[str, int]]],
    *,
    headroom_bytes: int,
    gather: Callable[[dict], Sequence[dict]],
) -> list[dict]:
    """Converge sampling errors before any rank allocates its Host arena."""
    try:
        request = sample_host_memory(components())
        local = {"request": request, "error": None}
    except Exception as exc:
        local = {"request": None, "error": f"{type(exc).__name__}: {exc}"}
    reports = gather(local)
    errors = [row["error"] for row in reports if row["error"] is not None]
    if errors:
        raise RuntimeError(f"Host memory sampling failed across ranks: {errors}")
    return validate_host_memory(
        [row["request"] for row in reports], headroom_bytes=headroom_bytes
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="JSON with allocations (name, bytes) and headroom_bytes for this host",
    )
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    request = sample_host_memory(
        tuple((entry["name"], entry["bytes"]) for entry in plan["allocations"])
    )
    report = validate_host_memory([request], headroom_bytes=plan["headroom_bytes"])
    print(json.dumps({"request": asdict(request), "domains": report}, indent=2))


if __name__ == "__main__":
    main()
