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

from types import SimpleNamespace

from prometheus_client import CollectorRegistry

from tokenspeed.runtime.cache.l3.metrics import L3Metrics
from tokenspeed.runtime.cache.l3.statistics import (
    ExecutorSnapshot,
    HealthSnapshot,
    OperationCounters,
)


def test_prometheus_snapshots_are_idempotent_and_keep_quarantine_visible():
    registry = CollectorRegistry()
    labels = {"model_name": "model", "dp_rank": "0", "tp_rank": "1"}
    metrics = L3Metrics(labels=labels, registry=registry)
    counters = OperationCounters(
        accepted=2,
        completed=2,
        backend_calls=1,
        cancelled_queued=1,
        unretired=1,
        failed_keys=2,
    )
    row = counters.snapshot(
        "get",
        operations=dict(queued=0, running=0, completed=0),
        byte_counts=dict(queued=0, running=0, completed=0),
    )
    snapshot = ExecutorSnapshot((row,), HealthSnapshot("open", 3, 1), True)
    scheduler = SimpleNamespace(
        operations=1,
        held_blocks=2,
        quarantined=1,
        negative_keys=1,
        backup_candidates=0,
        waiting_requests=0,
        ready_requests=0,
        wait_timeouts=1,
        ready_expirations=0,
        pressure_gate_closed=True,
        pressure_bypasses=3,
        pressure_dropped_backup_keys=7,
        overflow_dropped_backup_keys=2,
        unavailable_dropped_backup_keys=3,
        disabled_dropped_backup_keys=4,
        cleanup_dropped_backup_keys=5,
    )
    metrics.record(snapshot, scheduler)
    metrics.record(snapshot, scheduler)
    for reason, count in (
        ("pressure", 7),
        ("overflow", 2),
        ("host_unavailable", 3),
        ("write_disabled", 4),
        ("cleanup", 5),
    ):
        assert (
            registry.get_sample_value(
                "tokenspeed:l3_dropped_backup_keys_total", dict(labels, reason=reason)
            )
            == count
        )
    assert (
        registry.get_sample_value(
            "tokenspeed:l3_io_events_total", dict(labels, kind="get", event="completed")
        )
        == 2
    )
    assert (
        registry.get_sample_value(
            "tokenspeed:l3_preparation_expirations_total", dict(labels, phase="waiting")
        )
        == 1
    )
    assert (
        registry.get_sample_value(
            "tokenspeed:l3_scheduler_resources", dict(labels, resource="held_blocks")
        )
        == 2
    )
    assert registry.get_sample_value("tokenspeed:l3_io_quarantined", labels) == 1
    assert registry.get_sample_value("tokenspeed:l3_backend_state", labels) == 1
    assert (
        registry.get_sample_value("tokenspeed:l3_pressure_bypasses_total", labels) == 3
    )
    assert (
        registry.get_sample_value(
            "tokenspeed:l3_pressure_dropped_backup_keys_total", labels
        )
        == 7
    )
    assert (
        registry.get_sample_value(
            "tokenspeed:l3_scheduler_resources",
            dict(labels, resource="pressure_gate_closed"),
        )
        == 1
    )
    assert all(
        set(sample.labels)
        <= set(labels)
        | {"kind", "event", "result", "stage", "state", "resource", "phase", "reason"}
        for metric in registry.collect()
        for sample in metric.samples
    )
