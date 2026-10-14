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

"""Lossless coalescing when the consumer falls behind streaming producers."""

import copy

import pytest

from tokenspeed.runtime.engine.collector import RequestOutputCollector


@pytest.mark.parametrize("second", [[15], [15, 198]])
def test_repeated_token_deltas_are_not_cumulative_snapshots(second):
    collector = RequestOutputCollector()
    first = {"output_ids": [15], "meta_info": {"id": "r", "completion_tokens": 1}}
    last = {
        "output_ids": second,
        "meta_info": {"id": "r", "completion_tokens": 1 + len(second)},
    }
    originals = copy.deepcopy([first, last])
    collector.put(first, stream=True, cumulative=False)
    collector.put(last, stream=True, cumulative=False)
    result = collector.take()
    assert result["output_ids"] == [15, *second]
    assert result["meta_info"]["completion_tokens"] == len(result["output_ids"])
    assert [first, last] == originals


def test_equal_logprob_positions_are_preserved():
    collector = RequestOutputCollector()
    for count in (1, 2, 3):
        collector.put(
            {
                "text": "0" * count,
                "output_ids": [15],
                "meta_info": {
                    "id": "r",
                    "completion_tokens": count,
                    "logprobs": [{15: {"logprob": -0.25}}],
                    "output_token_logprobs": [(-0.25, 15, "0")],
                    "input_token_logprobs": [(None, 1, "prompt")] if count == 1 else [],
                    "cumulative_logprob": -0.25,
                },
            },
            stream=True,
            cumulative=False,
        )
    result = collector.take()
    assert result["text"] == "000"
    assert result["output_ids"] == [15, 15, 15]
    assert result["meta_info"]["logprobs"] == [{15: {"logprob": -0.25}}] * 3
    assert result["meta_info"]["output_token_logprobs"] == [(-0.25, 15, "0")] * 3
    assert result["meta_info"]["input_token_logprobs"] == [(None, 1, "prompt")]
    assert result["meta_info"]["cumulative_logprob"] == -0.75


def test_cumulative_tokens_replace_but_streaming_logprobs_append():
    collector = RequestOutputCollector()
    for count in (1, 2, 3):
        collector.put(
            {
                "output_ids": [15] * count,
                "output_multi_ids": [[15, 7]] * count,
                "meta_info": {
                    "id": "r",
                    "completion_tokens": count,
                    "output_token_logprobs": [(-0.25, 15, "0")],
                },
            },
            stream=True,
            cumulative=True,
        )
    result = collector.take()
    assert result["output_ids"] == [15] * 3
    assert result["output_multi_ids"] == [[15, 7]] * 3
    assert result["meta_info"]["output_token_logprobs"] == [(-0.25, 15, "0")] * 3


def test_taking_a_chunk_keeps_its_lists_independent_of_later_merges():
    collector = RequestOutputCollector()
    frames = [
        {"output_ids": [15], "output_multi_ids": [[15, 7]], "meta_info": {"id": "r"}}
        for _ in range(4)
    ]
    for frame in frames[:2]:
        collector.put(frame, stream=True, cumulative=False)
    first = collector.take()
    for frame in frames[2:]:
        collector.put(frame, stream=True, cumulative=False)
    second = collector.take()
    assert first["output_ids"] == second["output_ids"] == [15, 15]
    assert first["output_multi_ids"] == second["output_multi_ids"] == [[15, 7], [15, 7]]
    assert all(frame["output_ids"] == [15] for frame in frames)
    assert not collector.has_pending()
