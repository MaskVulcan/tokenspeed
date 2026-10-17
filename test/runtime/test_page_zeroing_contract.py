"""Contract for physical cache-page sanitization.

``ModelExecutor.zero_cache_pages`` runs the scheduler's page-reuse list through
the target pool and any stateful draft pool. Only pools that alias recurrent
state and KV bytes need clearing. Every GPU pool still returns a page-reuse
fence so external RDMA writes cannot race a previous owner's GPU work.
"""

from __future__ import annotations

import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=15, suite="runtime-1gpu")


class _DefaultStream:
    """Records the streams it was asked to wait on, in call order."""

    def __init__(self, trace):
        self._trace = trace

    def wait_stream(self, stream):
        self._trace.append(("wait", stream))


def _as_lists(pages):
    """Translated pages arrive as per-group arrays; compare them as lists."""
    return {group: list(map(int, ids)) for group, ids in pages.items()}


class TestZeroCachePagesContractTest:
    # Group-keyed requests are translated from scheduler (virtual) IDs before
    # any pool sees them; every group here is replicated, so the translation
    # only drops the reserved null block 0.
    _GROUPS = ("full", "state", "history", "target_only")

    @classmethod
    def _fake(cls, pool, draft_pool, trace):
        return types.SimpleNamespace(
            token_to_kv_pool=pool,
            draft_token_to_kv_pool=draft_pool,
            device="cpu",
            default_stream=_DefaultStream(trace),
            execution_stream="execution-stream",
            _cache_runtime_contract=types.SimpleNamespace(
                group_specs=tuple(
                    types.SimpleNamespace(group_id=group_id, shard_count=1)
                    for group_id in cls._GROUPS
                ),
                virtual_block_counts={group_id: 64 for group_id in cls._GROUPS},
            ),
            _cache_kvp_rank=0,
        )

    @classmethod
    def _call(cls, pool, page_ids, draft_pool=None):
        from tokenspeed.runtime.execution.model_executor import ModelExecutor

        fake = cls._fake(pool, draft_pool, trace=[])
        return ModelExecutor.zero_cache_pages(fake, page_ids)

    def test_empty_page_list_is_a_noop(self):
        from tokenspeed.runtime.execution.model_executor import ModelExecutor

        trace = []
        fake = self._fake(types.SimpleNamespace(), None, trace)
        assert (ModelExecutor.zero_cache_pages(fake, [])) is None
        assert (trace) == ([]), "an empty list places no fence either"

    def test_zeroing_orders_itself_behind_the_execution_stream(self):
        # The pages' previous owner may still be writing them from a forward
        # in flight on the execution stream: the default stream waits on it
        # before the pool touches a byte, and the function places that wait
        # itself rather than relying on a caller-side fence.
        from tokenspeed.runtime.execution.model_executor import ModelExecutor

        trace = []
        pool = types.SimpleNamespace(
            requires_page_zeroing=True,
            zero_pages=lambda page_ids: trace.append(("zero", list(page_ids))),
        )
        fake = self._fake(pool, None, trace)
        assert (ModelExecutor.zero_cache_pages(fake, [4, 5])) is None
        assert (trace) == ([("wait", "execution-stream"), ("zero", [4, 5])])

    def test_pure_attention_pool_ignores_page_reuse_list(self):
        # No zero_pages and not flagged as state-aliasing -> skip, no raise.
        pool = types.SimpleNamespace(requires_page_zeroing=False)
        assert (self._call(pool, [1, 2, 3])) is None

    def test_gpu_page_reuse_returns_a_fence_without_sanitization(self):
        from tokenspeed.runtime.execution.model_executor import ModelExecutor

        trace = []
        event = types.SimpleNamespace(
            record=lambda stream: trace.append(("record", stream))
        )
        fake = self._fake(
            types.SimpleNamespace(requires_page_zeroing=False), None, trace
        )
        fake.device = "cuda"
        fake.device_module = types.SimpleNamespace(Event=lambda: event)
        result = ModelExecutor.zero_cache_pages(fake, [4, 5])
        assert result is event
        assert trace == [("wait", "execution-stream"), ("record", fake.default_stream)]

    def test_pure_attention_fence_waits_for_prior_gpu_writes(self):
        import torch

        from tokenspeed.runtime.execution.model_executor import ModelExecutor

        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        execution = torch.cuda.Stream()
        default = torch.cuda.Stream()
        prior_done = torch.cuda.Event()
        value = torch.zeros(1, dtype=torch.int32, device="cuda")
        value.record_stream(execution)
        torch.cuda.synchronize()
        fake = self._fake(types.SimpleNamespace(requires_page_zeroing=False), None, [])
        fake.device = "cuda"
        fake.device_module = torch.cuda
        fake.default_stream = default
        fake.execution_stream = execution
        with torch.cuda.stream(execution):
            torch.cuda._sleep(100_000_000)
            value.fill_(7)
            prior_done.record()
        fence = ModelExecutor.zero_cache_pages(fake, [4])
        assert fence is not None, "RDMA must wait even when this pool needs no zeroing"
        fence.synchronize()
        assert prior_done.query()
        assert value.item() == 7

    def test_native_rdma_respects_pure_attention_page_reuse(self):
        import torch

        from tokenspeed.runtime.execution.model_executor import ModelExecutor
        from tokenspeed.runtime.pd.base.mooncake_engine import MooncakeTransferEngine

        raw = os.environ.get("MOONCAKE_GPU_TEST_CONFIG")
        if not raw:
            pytest.skip("set MOONCAKE_GPU_TEST_CONFIG for native RDMA validation")
        config = json.loads(raw)
        assert config["protocol"] == "rdma"
        assert torch.cuda.is_available()
        sender = MooncakeTransferEngine(
            config["local_hostname"], gpu_id=0, ib_device=config["device_name"]
        )
        receiver = MooncakeTransferEngine(
            config["local_hostname"], gpu_id=0, ib_device=config["device_name"]
        )
        source = torch.full((1 << 20,), 29, dtype=torch.uint8, pin_memory=True)
        destination = torch.zeros_like(source, device="cuda")
        execution = torch.cuda.Stream()
        default = torch.cuda.Stream()
        destination.record_stream(execution)
        fake = self._fake(types.SimpleNamespace(requires_page_zeroing=False), None, [])
        fake.device = "cuda"
        fake.device_module = torch.cuda
        fake.default_stream = default
        fake.execution_stream = execution
        torch.cuda.synchronize()
        assert sender.register(source.data_ptr(), source.numel()) == 0
        assert receiver.register(destination.data_ptr(), destination.numel()) == 0
        try:

            def transfer():
                assert (
                    sender.transfer_sync(
                        receiver.get_session_id(),
                        source.data_ptr(),
                        destination.data_ptr(),
                        source.numel(),
                    )
                    == 0
                )

            transfer()  # Warm transport setup before placing the delayed writer.
            torch.testing.assert_close(destination.cpu(), source, atol=0, rtol=0)
            with torch.cuda.stream(execution):
                torch.cuda._sleep(100_000_000)
                destination.fill_(7)
            fence = ModelExecutor.zero_cache_pages(fake, [4])
            # Match the receive submission: without a returned event, external
            # RDMA can finish before the previous owner's delayed GPU write.
            if fence is not None:
                fence.synchronize()
            transfer()
            torch.cuda.synchronize()
            torch.testing.assert_close(destination.cpu(), source, atol=0, rtol=0)
        finally:
            torch.cuda.synchronize()
            sender.deregister(source.data_ptr())
            receiver.deregister(destination.data_ptr())

    def test_missing_flag_defaults_to_skip(self):
        pool = types.SimpleNamespace()  # attribute absent entirely
        assert (self._call(pool, [1, 2, 3])) is None

    def test_state_aliasing_pool_without_impl_fails_loudly(self):
        # Declares it needs zeroing but forgot to implement it -> tripwire.
        pool = types.SimpleNamespace(requires_page_zeroing=True)
        with pytest.raises((RuntimeError)):
            self._call(pool, [1, 2, 3])

    def test_pool_with_impl_is_invoked(self):
        seen = []
        pool = types.SimpleNamespace(
            requires_page_zeroing=True,
            zero_pages=lambda page_ids: seen.append(list(page_ids)),
        )
        # device="cpu" -> returns None after invoking zero_pages.
        assert (self._call(pool, [4, 5])) is None
        assert (seen) == ([[4, 5]])

    def test_group_aware_pool_is_invoked(self):
        seen = []
        pool = types.SimpleNamespace(
            requires_page_zeroing=True,
            zero_new_blocks=lambda pages: seen.append(_as_lists(pages)),
        )
        pages = {"full": [4, 5], "state": [9]}
        assert (self._call(pool, pages)) is None
        assert (seen) == ([pages])

    def test_group_aware_pool_never_sees_the_null_block(self):
        seen = []
        pool = types.SimpleNamespace(
            requires_page_zeroing=True,
            zero_new_blocks=lambda pages: seen.append(_as_lists(pages)),
        )
        assert (self._call(pool, {"full": [0, 4], "state": [0]})) is None
        assert (seen) == ([{"full": [4], "state": []}])

    def test_group_aware_pool_rejects_out_of_range_virtual_blocks(self):
        pool = types.SimpleNamespace(
            requires_page_zeroing=True,
            zero_new_blocks=lambda pages: None,
        )
        with pytest.raises((IndexError)):
            self._call(pool, {"full": [64]})

    def test_stateful_draft_pool_zeros_its_group_subset(self):
        target_seen = []
        draft_seen = []
        target = types.SimpleNamespace(
            requires_page_zeroing=True,
            zero_new_blocks=lambda pages: target_seen.append(_as_lists(pages)),
        )
        draft = types.SimpleNamespace(
            requires_page_zeroing=True,
            arena=types.SimpleNamespace(
                cache_group_specs=(
                    types.SimpleNamespace(group_id="history"),
                    types.SimpleNamespace(group_id="state"),
                ),
            ),
            zero_new_blocks=lambda pages: draft_seen.append(_as_lists(pages)),
        )
        pages = {"history": [4], "state": [9], "target_only": [12]}

        assert (self._call(target, pages, draft)) is None
        assert (target_seen) == ([pages])
        assert (draft_seen) == ([{"history": [4], "state": [9]}])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
