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

"""Invalid cache rows must not contaminate live MLA decode requests."""

import pytest
import torch
from tokenspeed_kernel.ops.attention.mla import mla_decode_with_kvcache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads", [4, 8])
@pytest.mark.parametrize("batch", [4, 16])
@pytest.mark.parametrize("projected", [False, True])
def test_mla_decode_ignores_nan_cache_padding(heads, batch, projected):
    torch.manual_seed(17)
    page_size, rank, rope = 64, 512, 64
    lengths = [2018 if i % 2 == 0 else 6114 for i in range(batch - 1)] + [1]
    pages = [(length + page_size - 1) // page_size for length in lengths[:-1]]
    cache = torch.randn(
        (sum(pages) + 1, page_size, 1, rank + rope), dtype=torch.bfloat16, device="cuda"
    )
    table = torch.zeros((batch, 8192 // page_size), dtype=torch.int32, device="cuda")
    cursor = 1
    for row, count in enumerate(pages):
        table[row, :count] = torch.arange(cursor, cursor + count, device="cuda")
        cache[cursor + count - 1, lengths[row] % page_size :] = 0
        cursor += count
    cache[0] = 0
    poisoned = cache.clone()
    poisoned[0] = float("nan")
    cursor = 1
    for row, count in enumerate(pages):
        poisoned[cursor + count - 1, lengths[row] % page_size :] = float("nan")
        cursor += count
    q = torch.randn((batch, 1, heads, rank + rope), dtype=torch.bfloat16, device="cuda")
    seq_lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")

    weight = torch.randn((heads, rank, 128), dtype=torch.bfloat16, device="cuda")

    def decode(kv):
        return mla_decode_with_kvcache(
            q,
            kv,
            table,
            seq_lens,
            8192,
            128,
            rank,
            rope,
            softmax_scale=192**-0.5,
            logit_cap=0.0,
            window_left=-1,
            noncausal_block_size=1,
            return_lse=False,
            solution=None,
            value_weight=weight if projected else None,
            out=(
                torch.empty((batch, heads * 128), dtype=torch.bfloat16, device="cuda")
                if projected
                else None
            ),
        )

    reference = decode(cache)
    dirty = cache.clone().fill_(float("nan"))
    decode(dirty)
    actual = decode(poisoned)
    assert torch.isfinite(reference[:-1]).all()
    assert torch.isfinite(actual[:-1]).all()
    torch.testing.assert_close(actual[:-1], reference[:-1], atol=0, rtol=0)

    # Replays must overwrite their private scratch even after invalid inputs.
    working = cache.clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = decode(working)
    working.fill_(float("nan"))
    graph.replay()
    working.copy_(poisoned)
    graph.replay()
    assert torch.isfinite(captured[:-1]).all()
    torch.testing.assert_close(captured[:-1], reference[:-1], atol=0, rtol=0)
