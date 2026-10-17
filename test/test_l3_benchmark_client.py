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

import asyncio
import json
from test.benchmarks.l3.client import request, summarize
from test.benchmarks.l3.instrumentation import Observer
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


class Response:
    status = 200

    def __init__(self, *, done, reason):
        self.events = [
            {"choices": [{"delta": {"content": "123456"}, "finish_reason": None}]},
            {
                "choices": [{"delta": {}, "finish_reason": reason}],
                "usage": {"completion_tokens": 2, "prompt_tokens": 512},
            },
        ]
        self.done = done
        self.content = self.lines()

    async def lines(self):
        for event in self.events:
            yield f"data: {json.dumps(event)}\n".encode()
        if self.done:
            yield b"data: [DONE]\n"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class Session:
    def __init__(self, response):
        self.response = response

    def post(self, *args, **kwargs):
        return self.response


@pytest.mark.parametrize("done,reason", [(False, "stop"), (True, None)])
def test_truncated_stream_is_not_a_successful_measurement(done, reason):
    with pytest.raises(RuntimeError, match="Incomplete SSE"):
        asyncio.run(
            request(
                Session(Response(done=done, reason=reason)),
                url="test",
                model="test",
                prompt="test",
                max_tokens=8,
                expected="123456",
            )
        )


def test_completed_stream_and_small_sample_statistics():
    row = asyncio.run(
        request(
            Session(Response(done=True, reason="stop")),
            url="test",
            model="test",
            prompt="test",
            max_tokens=8,
            expected="123456",
        )
    )
    assert row["done"] and row["finish_reason"] == "stop"
    report = summarize([row])
    assert report["requests"] == 1
    assert report["ttft_s"]["p99"] == row["ttft_s"]
    assert "small cohorts" in report["p99_interpretation"]


def test_cleanup_waits_for_store_side_offload_without_forgetting_owned_keys(tmp_path):
    control = tmp_path / "control.json"
    output = tmp_path / "owners"
    control.write_text(json.dumps({"seq": 1, "action": "purge_owned"}))
    observer = Observer(control=control, output=output, role="test")
    observer.owned = {"namespace_one", "namespace_two"}
    observer.client = Mock()
    observer.client.remove_by_regex.side_effect = [1, 1]
    observer.client.batch_is_exist.side_effect = [[0, 1], [0]]
    observer.state = Mock(
        return_value={"quarantine": 0, "pending": False, "candidates": 0}
    )
    hooks = SimpleNamespace(
        _scheduler=Mock(),
        _l3=SimpleNamespace(
            store=SimpleNamespace(key_prefix="namespace"), is_idle=lambda: True
        ),
    )
    hooks._scheduler.can_clear_cache.return_value = True
    observer.poll(hooks)
    assert observer.owned == {"namespace_two"}
    assert not json.loads((output / "test-0.json").read_text())["ready"]
    assert observer.seq == 0
    observer.next_poll = 0
    observer.poll(hooks)
    assert not observer.owned
    assert json.loads((output / "test-0.json").read_text())["ready"]
    assert observer.seq == 1


def test_cleanup_refuses_keys_outside_benchmark_namespace(tmp_path):
    control = tmp_path / "control.json"
    control.write_text(json.dumps({"seq": 1, "action": "purge_owned"}))
    observer = Observer(control=control, output=tmp_path / "owners", role="test")
    observer.owned = {"another_namespace_key"}
    observer.client = Mock()
    observer.state = Mock(
        return_value={"quarantine": 0, "pending": False, "candidates": 0}
    )
    hooks = SimpleNamespace(
        _scheduler=Mock(),
        _l3=SimpleNamespace(
            store=SimpleNamespace(key_prefix="benchmark"), is_idle=lambda: True
        ),
    )
    with pytest.raises(AssertionError, match="outside the isolated"):
        observer.poll(hooks)
    observer.client.remove_by_regex.assert_not_called()
