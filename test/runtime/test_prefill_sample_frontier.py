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

"""Prefill sampling never mutates the live cache frontier before commit."""

import pytest
import torch
from tokenspeed_kernel.ops.metadata import advance_accepted_frontier

from tokenspeed.runtime.execution.model_executor import _with_prefill_sample_offsets
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "npu:0",
            marks=pytest.mark.skipif(
                torch._C._get_privateuse1_backend_name() != "npu",
                reason="requires torch_npu",
            ),
        ),
    ],
)
@pytest.mark.parametrize("prefix,chunk", [(0, 1024), (4096, 3968), (8064, 1)])
def test_sample_offsets_preserve_frontier_until_commit(device, prefix, chunk):
    live = torch.tensor([0, prefix, 41], dtype=torch.int32, device=device)
    rows = torch.tensor([1], dtype=torch.int64, device=device)
    lengths = torch.tensor([chunk], dtype=torch.int32, device=device)
    info = SamplingBatchInfo(
        req_pool_indices=rows, valid_cache_lengths=live, device=device
    )
    sampled = _with_prefill_sample_offsets(info, lengths)
    assert sampled.valid_cache_lengths.cpu().tolist() == [0, prefix + chunk - 1, 41]
    assert live.cpu().tolist() == [0, prefix, 41]
    advance_accepted_frontier(
        rows,
        lengths,
        torch.ones_like(lengths),
        live,
        1,
        0,
        ngram_tail=None,
        ngram_previous_tokens=None,
        ngram_token_mask=None,
        input_ids=None,
    )
    assert live.cpu().tolist() == [0, prefix + chunk, 41]
