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

"""Tests for the scheduler ready pipe when a child dies before it reports."""

import contextlib
import os
import signal
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.utils.process import (  # noqa: E402
    receive_ready_messages,
    start_process_with_ready_pipe,
)

# Long enough for a slow spawn; a blocked recv() would wait forever instead.
_TIMEOUT_S = 60


def _report_ready(value, writer):
    writer.send({"status": "ready", "value": value})
    # A scheduler keeps running after it reports.
    time.sleep(_TIMEOUT_S * 10)


def _die_before_ready(writer):
    os.kill(os.getpid(), signal.SIGKILL)


def _report_then_die(writer):
    writer.send({"status": "ready"})
    os.kill(os.getpid(), signal.SIGKILL)


def _never_ready(writer):
    # Like a rank blocked in a collective on a peer that died.
    time.sleep(_TIMEOUT_S * 10)


def _fail_with_background_peer(exit_kind, writer):
    from tokenspeed.runtime.engine import event_loop as event_loop_module
    from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs

    class FailedDeviceBuild:
        def __init__(self, *_args):
            # Stand in for model/device startup, which needs real GPU weights.
            threading.Thread(target=threading.Event().wait, daemon=False).start()
            if exit_kind == "system_exit":
                raise SystemExit(1)
            raise ValueError("L3 construction failed after starting the PD peer")

    class Parent:
        def parent(self):
            return self

        def send_signal(self, _signum):
            # Do not deliver the scheduler's failure signal to pytest itself.
            pass

    args = ServerArgs(model="unused", device="cpu", world_size=1, nprocs_per_node=1)
    args.mapping.rank = 0
    ports = PortArgs(
        tokenizer_ipc_name="unused",
        scheduler_input_ipc_name="unused",
        nccl_port=0,
        dist_init_addr="localhost:0",
        rpc_ipc_name="unused",
        metrics_ipc_name="unused",
        tokenizer_worker_ipc_name=None,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(event_loop_module, "EventLoop", FailedDeviceBuild)
        patch.setattr(event_loop_module.psutil, "Process", Parent)
        event_loop_module.run_event_loop(args, ports, writer)


class TestReadyPipe:
    def test_ready_message_reaches_parent(self, start_process) -> None:
        proc, reader = start_process(_report_ready, (7,))
        assert reader.poll(_TIMEOUT_S)
        assert reader.recv() == {"status": "ready", "value": 7}

    def test_killed_child_raises_eof_instead_of_blocking(self) -> None:
        proc, reader = start_process_with_ready_pipe(
            _die_before_ready, (), contextlib.nullcontext()
        )
        proc.join(_TIMEOUT_S)
        assert proc.exitcode == -signal.SIGKILL
        message = "recv() would block: the parent still holds the write end"
        assert reader.poll(_TIMEOUT_S), message
        with pytest.raises(EOFError):
            reader.recv()

    def _receive_or_fail(self, started):
        """receive_ready_messages, failing the test instead of hanging."""
        result = {}

        def run():
            try:
                result["messages"] = receive_ready_messages(
                    [reader for _, reader in started], [proc for proc, _ in started]
                )
            except RuntimeError as exc:
                result["error"] = exc

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(_TIMEOUT_S)
        assert not thread.is_alive(), "receive_ready_messages hung"
        return result

    @pytest.fixture
    def start_process(self, request):
        def start(target, args=()):
            proc, reader = start_process_with_ready_pipe(
                target, args, contextlib.nullcontext()
            )
            request.addfinalizer(proc.kill)
            return proc, reader

        return start

    def test_receive_returns_messages_in_reader_order(self, start_process) -> None:
        started = [start_process(_report_ready, (value,)) for value in (1, 2)]
        messages = self._receive_or_fail(started)["messages"]
        assert [m["value"] for m in messages] == [1, 2]

    def test_receive_fails_when_another_rank_dies(self, start_process) -> None:
        # Rank 0 stays alive without reporting; rank 1 is killed. Waiting on
        # rank 0's pipe alone would hang.
        started = [start_process(_never_ready), start_process(_die_before_ready)]
        assert "Scheduler process 1" in str(self._receive_or_fail(started)["error"])

    def test_receive_fails_when_a_reported_rank_dies(self, start_process) -> None:
        # Rank 1 reports and then dies while rank 0 is still loading: only
        # its process exit, not its pipe, shows it.
        started = [start_process(_never_ready), start_process(_report_then_die)]
        assert "Scheduler process 1" in str(self._receive_or_fail(started)["error"])

    @pytest.mark.parametrize("exit_kind", ["exception", "system_exit"])
    def test_failed_startup_reaps_background_peer_without_waiting_for_other_ranks(
        self, start_process, exit_kind
    ) -> None:
        started = [
            start_process(_never_ready),
            start_process(_fail_with_background_peer, (exit_kind,)),
        ]
        failed, _ = started[1]
        assert "Scheduler process 1" in str(self._receive_or_fail(started)["error"])
        assert not failed.is_alive()
        assert failed.exitcode == -signal.SIGKILL


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
