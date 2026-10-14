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

"""SSE request measurements and out-of-window benchmark control."""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path


async def request(
    session, *, url: str, model: str, prompt: str, max_tokens: int, expected: str
) -> dict:
    payload = dict(
        model=model,
        messages=[dict(role="user", content=prompt)],
        temperature=0,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
        chat_template_kwargs={"enable_thinking": False},
    )
    begin = time.perf_counter()
    first = None
    chunks = []
    text = ""
    usage = None
    reason = None
    done = False
    started_ns = time.time_ns()
    async with session.post(url, json=payload) as response:
        if response.status != 200:
            raise RuntimeError(
                f"HTTP {response.status}: {(await response.text())[:1000]}"
            )
        async for raw in response.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content") or delta.get("reasoning_content") or ""
                if content:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    chunks.append(time.time_ns())
                    text += content
                reason = choice.get("finish_reason") or reason
    end = time.perf_counter()
    if not done or reason not in ("stop", "length") or usage is None or first is None:
        raise RuntimeError(
            f"Incomplete SSE response: done={done}, finish={reason}, usage={usage}"
        )
    if expected not in text:
        raise AssertionError(f"Expected label {expected!r} missing in {text[:200]!r}")
    count = usage["completion_tokens"]
    return dict(
        start_ns=started_ns,
        end_ns=time.time_ns(),
        ttft_s=first - begin,
        e2e_s=end - begin,
        tpot_s=(end - first) / (count - 1) if count > 1 else None,
        usage=usage,
        cached_tokens=usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
        finish_reason=reason,
        done=done,
        output=text,
        chunk_times_ns=chunks,
    )


def summarize(rows: list[dict]) -> dict:
    """Latency distributions only; independent reset/seed episodes are not QPS."""
    result = {"requests": len(rows)}
    for field in ("ttft_s", "e2e_s", "tpot_s"):
        values = sorted(r[field] for r in rows if r[field] is not None)
        if not values:
            continue
        result[field] = {
            f"p{percent}": values[max(0, math.ceil(len(values) * percent / 100) - 1)]
            for percent in (50, 95, 99)
        }
    result["cached_tokens"] = [r["cached_tokens"] for r in rows]
    result["p99_interpretation"] = (
        "empirical order statistic; small cohorts do not establish a stable tail"
    )
    return result


class Control:
    def __init__(self, path: Path, output: Path, *, owners: int):
        self.path = path
        self.output = output
        self.owners = owners
        self.seq = 0
        self.settings = {}

    async def command(self, action: str, *, timeout_s: float = 60) -> list[dict]:
        self.seq += 1
        temporary = self.path.with_suffix(".new")
        temporary.write_text(
            json.dumps(dict(seq=self.seq, action=action, settings=self.settings))
        )
        temporary.replace(self.path)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rows = [
                json.loads(p.read_text())
                for p in self.output.glob("*.json")
                if "purged" not in p.name
            ]
            rows = [r for r in rows if r.get("seq") == self.seq]
            if any(r.get("error") for r in rows):
                raise RuntimeError(f"Benchmark control failed: {rows}")
            if len(rows) == self.owners and all(r.get("ready") for r in rows):
                return rows
            await asyncio.sleep(0.05)
        raise TimeoutError(
            f"Benchmark {action} did not converge across {self.owners} owners"
        )
