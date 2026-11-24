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

"""The Disk tier's startup settings (``docs/rfcs/0006-disk-tier.md``): the
scheduler's limits and the engine host's file and executor settings, resolved
once from the server arguments and the Host pool's geometry.

The Disk pool packs like the Host pool, so a Disk block id maps to a file
offset as a Host block id maps to a Host pool offset; the Host layout is
then aligned to ``DIRECT_IO_ALIGNMENT`` (``host_storage.py``).
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenspeed.runtime.cache.host_storage import HostCacheStorage
    from tokenspeed.runtime.cache.transfer.ownership import BlockOwnerTranslation


# The direct-I/O alignment of every offset, address and length (``ts_disk::ALIGNMENT``).
DIRECT_IO_ALIGNMENT = 4096


def host_block_alignment(server_args) -> int:
    """The alignment the Host layout keeps: the direct-I/O one with a Disk
    tier, 1 without."""
    return DIRECT_IO_ALIGNMENT if server_args.disk_tier_paths is not None else 1


@dataclass(frozen=True, slots=True)
class DiskTierTuning:
    """``--disk-tier-config``: the Disk tier's tuning, each key optional.

    Attributes:
        queue_depth: Requests in flight on the drive; with psync, threads.
        write_queue_depth: Of those, at most this many cleaning writes.
        max_io_bytes: Longer ranges are issued as several requests.
        max_operations: Disk operations in flight, reads and writes.
        max_blocks_per_operation: Blocks one operation moves.
        host_blocks_fraction: The share of Host L2 blocks Disk operations
            may hold at once, counted in the densest group's blocks.
        write_host_blocks_fraction: Of those, the share cleaning writes may
            hold.
        max_queued_cleanings: Host entries waiting to be cleaned.
        min_prefetch_tokens: The fewest tokens past the local hit worth a
            prefetch.
        wait_timeout_ms: How long a request waits for its prefetch.
        ready_timeout_ms: How long a finished prefetch is held for it.
        push_ring_mb: With peer sharing, the staging ring a peer push reads
            Disk blocks into, in MB, rounded up to whole blocks.
    """

    queue_depth: int = 32
    write_queue_depth: int = 8
    max_io_bytes: int = 4 << 20
    max_operations: int = 16
    max_blocks_per_operation: int = 4096
    host_blocks_fraction: float = 0.25
    write_host_blocks_fraction: float = 0.5
    max_queued_cleanings: int = 1 << 16
    min_prefetch_tokens: int = 256
    wait_timeout_ms: int = 10000
    ready_timeout_ms: int = 5000
    push_ring_mb: int = 512
    min_clean_tokens: int = 0
    require_reuse_for_cleaning: bool = False

    @classmethod
    def from_json(cls, text: str | None) -> DiskTierTuning:
        """Parse ``--disk-tier-config``; an unknown key is refused."""
        if text is None:
            return cls()
        values = json.loads(text)
        if not isinstance(values, dict):
            raise ValueError("--disk-tier-config must be a JSON object")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown --disk-tier-config keys: {sorted(unknown)}")
        tuning = cls(**values)
        if not 0 < tuning.write_queue_depth <= tuning.queue_depth:
            raise ValueError("write_queue_depth must be in [1, queue_depth]")
        if tuning.max_io_bytes <= 0 or tuning.max_io_bytes % DIRECT_IO_ALIGNMENT:
            raise ValueError(
                f"max_io_bytes must be a positive multiple of {DIRECT_IO_ALIGNMENT}"
            )
        if not 0 < tuning.host_blocks_fraction <= 1:
            raise ValueError("host_blocks_fraction must be in (0, 1]")
        if not 0 < tuning.write_host_blocks_fraction <= 1:
            raise ValueError("write_host_blocks_fraction must be in (0, 1]")
        if tuning.push_ring_mb <= 0:
            raise ValueError("push_ring_mb must be positive")
        if tuning.min_clean_tokens < 0:
            raise ValueError("min_clean_tokens must be non-negative")
        return tuning


def num_disk_lcm_blocks(server_args, host_lcm_block_bytes: int) -> int:
    """The Disk pool's parents per rank: the size over the Host parent."""
    return int(server_args.disk_tier_size_gb * 1e9) // host_lcm_block_bytes


def disk_scheduler_config(tuning: DiskTierTuning, *, num_host_blocks: int):
    """The scheduler's ``DiskConfig``.

    Args:
        tuning: ``--disk-tier-config``.
        num_host_blocks: The Host pool's blocks in its densest group: its
            parents, the null block excluded, times that group's packing
            (``SchedulerConfig::host_group_packing``). A hybrid model's
            prefetch must hold every KV page under the State checkpoint it
            reaches, so the shares are of blocks, not parents.
    """
    from tokenspeed_engine.scheduler import DiskConfig

    max_host_blocks = max(1, int(num_host_blocks * tuning.host_blocks_fraction))
    return DiskConfig(
        max_operations=tuning.max_operations,
        max_blocks_per_operation=tuning.max_blocks_per_operation,
        max_host_blocks=max_host_blocks,
        max_write_host_blocks=max(
            1, int(max_host_blocks * tuning.write_host_blocks_fraction)
        ),
        max_queued_cleanings=tuning.max_queued_cleanings,
        min_prefetch_tokens=tuning.min_prefetch_tokens,
        wait_timeout_us=tuning.wait_timeout_ms * 1000,
        ready_timeout_us=tuning.ready_timeout_ms * 1000,
        min_clean_tokens=tuning.min_clean_tokens,
        require_reuse_for_cleaning=tuning.require_reuse_for_cleaning,
    )


@dataclass(frozen=True, slots=True)
class DiskSettings:
    """What the engine host starts this rank's Disk tier from.

    Attributes:
        file: This rank's file.
        file_bytes: Its size, the Disk pool's.
        backend: ``"io_uring"`` or ``"psync"``.
        queue_depth: Requests in flight; with psync, threads.
        write_queue_depth: Of those, at most this many writes.
        max_io_bytes: Longer ranges are issued as several requests.
        write_bytes_per_second: The write budget.
        host_buffer: The Host pool's address.
        host_buffer_bytes: Its size.
        groups: Per cache group, ``(packing, block_bytes, shard_count)``.
        lcm_block_bytes: One Host (and Disk) parent's bytes.
        num_host_lcm_blocks: Host parents, the null block excluded.
        num_disk_lcm_blocks: Disk parents, the null block excluded.
        owner_rank: This rank among a sharded group's owners.
        tp_rank: This rank's attention-TP rank.
        max_operations: The scheduler's Disk operation credits.
        push_ring_bytes: With peer sharing, the staging ring peer pushes
            read Disk blocks into; 0 without.
    """

    file: str
    file_bytes: int
    backend: str
    queue_depth: int
    write_queue_depth: int
    max_io_bytes: int
    write_bytes_per_second: int
    host_buffer: int
    host_buffer_bytes: int
    groups: tuple[tuple[int, int, int], ...]
    lcm_block_bytes: int
    num_host_lcm_blocks: int
    num_disk_lcm_blocks: int
    owner_rank: int
    tp_rank: int
    max_operations: int
    push_ring_bytes: int


def is_raw_block_device_path(path: str) -> bool:
    """Whether ``path`` names a raw block device (e.g. ``/dev/nvme0n1`` or a
    ``/dev/disk/by-id/...`` symlink) rather than a directory."""
    try:
        mode = os.stat(path).st_mode
    except OSError:
        normalized = os.path.normpath(path)
        return normalized.startswith("/dev/") and not path.endswith(os.sep)
    if stat.S_ISBLK(mode):
        return True
    if stat.S_ISDIR(mode):
        return False
    return os.path.normpath(path).startswith("/dev/")


def _local_world_size(server_args) -> int:
    """Number of ranks sharing this node's ``/dev`` namespace."""
    nnodes = max(1, int(getattr(server_args, "nnodes", 1) or 1))
    mapping = getattr(server_args, "mapping", None)
    if mapping is not None:
        per_node = max(1, int(mapping.world_size) // nnodes)
        nprocs_per_node = getattr(mapping, "nprocs_per_node", None)
        if isinstance(nprocs_per_node, int) and nprocs_per_node > 0:
            return min(per_node, nprocs_per_node)
        return per_node
    tp_size = max(1, int(getattr(server_args, "tp_size", 1) or 1))
    dp_size = max(1, int(getattr(server_args, "dp_size", 1) or 1))
    return max(1, (tp_size * dp_size) // nnodes)


def _mapping_rank(mapping) -> int | None:
    """Return ``mapping``'s initialized global rank, or ``None`` before rank
    assignment (``Mapping.rank`` asserts when ``_rank`` is still ``None``)."""
    if mapping is None:
        return None
    if hasattr(mapping, "_rank"):
        return mapping._rank
    try:
        return getattr(mapping, "rank", None)
    except AssertionError:
        return None


def _node_global_ranks(
    server_args, *, global_rank: int | None = None
) -> range:
    local_ranks = _local_world_size(server_args)
    if global_rank is not None:
        base = (int(global_rank) // local_ranks) * local_ranks
    else:
        mapping_rank = _mapping_rank(getattr(server_args, "mapping", None))
        if mapping_rank is not None:
            base = (max(0, int(mapping_rank)) // local_ranks) * local_ranks
        else:
            node_rank = max(0, int(getattr(server_args, "node_rank", 0) or 0))
            base = node_rank * local_ranks
    return range(base, base + local_ranks)


def validate_disk_tier_paths(
    server_args, *, global_rank: int | None = None
) -> None:
    """Validate ``--disk-tier-paths``: directories may be shared across local
    ranks (each rank opens its own per-port/per-rank file), while raw block
    devices (``/dev/...``) must map exclusively to one local rank so two ranks
    never overwrite offset 0 of the same device."""
    paths = server_args.disk_tier_paths
    if not paths:
        raise ValueError("--disk-tier-paths needs at least one directory")
    seen_raw: dict[str, int] = {}
    for rank in _node_global_ranks(server_args, global_rank=global_rank):
        target = paths[rank % len(paths)]
        if not is_raw_block_device_path(target):
            continue
        canonical = os.path.realpath(target)
        if canonical in seen_raw:
            raise ValueError(
                f"raw block device {target!r} in --disk-tier-paths cannot be "
                f"shared by multiple local ranks (ranks {seen_raw[canonical]} "
                f"and {rank}); supply one distinct block device or "
                f"partition per local rank, or use mounted directories"
            )
        seen_raw[canonical] = rank


def disk_file_path(server_args, *, global_rank: int) -> str:
    """This rank's file or raw block device: one directory or block device per
    drive, ranks spread over them; for directories the port and rank keep
    engines apart, while raw block devices (``/dev/...``) are opened directly."""
    validate_disk_tier_paths(server_args, global_rank=global_rank)
    target = server_args.disk_tier_paths[
        global_rank % len(server_args.disk_tier_paths)
    ]
    if is_raw_block_device_path(target):
        return target
    return os.path.join(
        target, f"tokenspeed-disk-p{server_args.port}-r{global_rank}.bin"
    )


def disk_settings(
    server_args,
    *,
    tuning: DiskTierTuning,
    host_storage: HostCacheStorage,
    ownership: BlockOwnerTranslation,
    global_rank: int,
) -> DiskSettings:
    """The engine host's settings for this rank's Disk tier.

    Args:
        server_args: The resolved server arguments.
        tuning: ``--disk-tier-config``.
        host_storage: The Host pool, laid out with ``DIRECT_IO_ALIGNMENT``;
            the caller keeps it for the process, since the executor
            addresses it by pointer.
        ownership: The Host pool's block owners: each group's shard count
            and this rank among a sharded group's owners.
        global_rank: This rank's global rank, which names its file.
    """
    disk_blocks = num_disk_lcm_blocks(server_args, host_storage.host_lcm_block_bytes)
    if disk_blocks <= 0:
        raise ValueError(
            f"--disk-tier-size-gb {server_args.disk_tier_size_gb} holds no "
            f"{host_storage.host_lcm_block_bytes}-byte Host parent"
        )
    return DiskSettings(
        file=disk_file_path(server_args, global_rank=global_rank),
        file_bytes=disk_blocks * host_storage.host_lcm_block_bytes,
        backend=server_args.disk_tier_io,
        queue_depth=tuning.queue_depth,
        write_queue_depth=tuning.write_queue_depth,
        max_io_bytes=tuning.max_io_bytes,
        write_bytes_per_second=server_args.disk_tier_write_mbps * 1_000_000,
        host_buffer=int(host_storage.host_buffer.data_ptr()),
        host_buffer_bytes=int(host_storage.host_buffer_bytes),
        groups=tuple(
            (int(packing), int(block_bytes), int(shards))
            for packing, block_bytes, shards in zip(
                host_storage.group_packing,
                host_storage.host_cache_block_bytes,
                ownership.shard_counts,
                strict=True,
            )
        ),
        lcm_block_bytes=int(host_storage.host_lcm_block_bytes),
        num_host_lcm_blocks=int(host_storage.num_host_lcm_blocks),
        num_disk_lcm_blocks=disk_blocks,
        owner_rank=int(ownership.rank),
        tp_rank=int(server_args.mapping.attn.tp_rank),
        max_operations=tuning.max_operations,
        # A holder pushes its Disk pages to peers through the ring.
        push_ring_bytes=(
            tuning.push_ring_mb * 1_000_000
            if server_args.peer_prefix_port is not None
            else 0
        ),
    )
