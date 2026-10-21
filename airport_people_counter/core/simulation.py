"""Deterministic traffic simulation for the 10 required scenarios.

Drives a ``QueueMonitor`` with a fake clock and scripted sensor readings so
the fusion / state-machine / backlog / alert logic can be validated
end-to-end without hardware. Each scenario reports its expected outcome
and whether the engine produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .config import Settings
from .models import (QueueState, SensorHealth, SensorReading, ZoneSnapshot,
                     utc_now_iso)
from .monitor import QueueMonitor


class FakeClock:
    """Simulated UTC clock; start at a fixed epoch and advance manually."""

    def __init__(self, start: float = 1_700_000_000.0):
        self.t = start

    def __call__(self) -> str:
        return datetime.fromtimestamp(self.t, tz=timezone.utc).isoformat(
            timespec="milliseconds").replace("+00:00", "Z")

    def advance(self, seconds: float) -> None:
        self.t += seconds


def make_reading(clock, sensor_id: str, sensor_type: str, people: int,
                 density: float, flow: float = 0.0,
                 health: SensorHealth = SensorHealth.ONLINE,
                 confidence: float = 0.95) -> SensorReading:
    ts = clock()
    return SensorReading(
        sensor_id=sensor_id,
        sensor_type=sensor_type,
        event_time=ts,
        server_received_time=ts,
        people_count=people,
        density=density,
        flow_rate=flow,
        confidence=confidence,
        health=health,
    )


@dataclass
class ScenarioResult:
    name: str
    description: str
    expected: QueueState
    actual: QueueState
    passed: bool
    notes: str = ""

    def line(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}: expected={self.expected.value} actual={self.actual.value}"


class ScenarioRunner:
    """Drives scenarios; a scenario feeds readings then asserts the state."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.clock = FakeClock()
        self.monitor = QueueMonitor(settings, clock=self.clock)

    def feed(self, s1: tuple = (0, 0.0, 0.0), s2: tuple | None = None,
             s3: tuple | None = None, advance: float = 1.0,
             s1_health=SensorHealth.ONLINE, s2_health=SensorHealth.ONLINE,
             s3_health=SensorHealth.ONLINE) -> ZoneSnapshot:
        """Feed one evaluation step. tuples are (people, density, flow)."""
        if s1 is not None:
            p, d, f = s1
            self.monitor.ingest_sensor(make_reading(
                self.clock, "SENSOR_01", "vision", p, d, f, s1_health))
        if s2 is not None:
            p, d, f = s2
            self.monitor.ingest_sensor(make_reading(
                self.clock, "SENSOR_02", "vision_secondary", p, d, f, s2_health))
        if s3 is not None:
            p, d, f = s3
            self.monitor.ingest_sensor(make_reading(
                self.clock, "SENSOR_03", "vision_downstream", p, d, f, s3_health))
        snap = self.monitor.evaluate()
        self.clock.advance(advance)
        return snap

    def run(self, name: str, description: str, expected: QueueState,
            steps: list, notes: str = "") -> ScenarioResult:
        snapshots = [self.feed(**s) for s in steps]
        last = snapshots[-1]
        return ScenarioResult(name, description, expected, last.state,
                              last.state == expected, notes)


def run_all(settings: Settings | None = None, verbose: bool = True) -> list:
    settings = settings or Settings.load()
    results = []

    def fresh(clock_start: float = 1_700_000_000.0) -> ScenarioRunner:
        runner = ScenarioRunner(settings)
        runner.clock = FakeClock(clock_start)
        runner.monitor = QueueMonitor(settings, clock=runner.clock)
        return runner

    def step(s1, s2=None, s3=None, adv=1.0, s1h=SensorHealth.ONLINE,
             s2h=SensorHealth.ONLINE, s3h=SensorHealth.ONLINE):
        return {"s1": s1, "s2": s2, "s3": s3, "advance": adv,
                "s1_health": s1h, "s2_health": s2h, "s3_health": s3h}

    # 1. Empty zone
    runner = fresh()
    results.append(runner.run(
        "empty", "No passengers present", QueueState.EMPTY,
        [step((0, 0.0, 0.0), s2=(0, 0.0, 0.0)) for _ in range(6)]))

    # 2. Normal operations
    runner = fresh()
    results.append(runner.run(
        "normal_operations", "Moderate queue, healthy throughput", QueueState.NORMAL,
        [step((22, 0.44, 12.0), s2=(22, 0.44, 12.5), s3=(6, 0.12, 12.0))
         for _ in range(6)]))

    # 3. Increasing queue (build-up past crowded into congested)
    runner = fresh()
    ramp = []
    for _ in range(6):
        ramp.append(step((10, 0.20, 12.0), s2=(10, 0.20, 12.0)))
    for _ in range(6):
        ramp.append(step((22, 0.44, 12.0), s2=(22, 0.44, 12.0)))
    for _ in range(6):
        ramp.append(step((35, 0.70, 12.0), s2=(36, 0.72, 12.0)))
    for _ in range(6):
        ramp.append(step((45, 0.90, 12.0), s2=(46, 0.92, 12.0)))
    results.append(runner.run(
        "increasing_queue", "Queue builds through NORMAL, CROWDED into CONGESTED",
        QueueState.CONGESTED, ramp))

    # 4. Crowded
    runner = fresh()
    results.append(runner.run(
        "crowded", "Sustained high density", QueueState.CROWDED,
        [step((35, 0.70, 12.0), s2=(36, 0.72, 12.0)) for _ in range(6)]))

    # 5. Congested (healthy flow, so no backlog)
    runner = fresh()
    results.append(runner.run(
        "congested", "Very high density but flow still healthy", QueueState.CONGESTED,
        [step((45, 0.90, 12.0), s2=(46, 0.92, 12.0)) for _ in range(6)]))

    # 6. Backlog (low throughput persists)
    runner = fresh()
    results.append(runner.run(
        "backlog", "High queue + sustained low flow", QueueState.BACKLOG,
        [step((45, 0.90, 2.0), s2=(46, 0.92, 2.0), adv=2.0) for _ in range(7)],
        notes="needs persistence_seconds of low flow"))

    # 7. Sensor 1 failure -> still data via Sensor 2; total failure -> sensor error
    runner = fresh()
    results.append(runner.run(
        "sensor1_failure", "S1 offline, S2 still healthy", QueueState.SPARSE,
        [step(s1=(15, 0.30, 12.0), s2=(15, 0.30, 12.0),
              s1h=SensorHealth.OFFLINE) for _ in range(6)]))

    runner = fresh()
    results.append(runner.run(
        "all_sensors_failure", "All primary sensors offline", QueueState.SENSOR_ERROR,
        [step(s1=(0, 0.0, 0.0), s2=(0, 0.0, 0.0),
              s1h=SensorHealth.OFFLINE, s2h=SensorHealth.OFFLINE) for _ in range(6)]))

    # 8. Sensor 2 disagreement with Sensor 1 on a high state
    runner = fresh()
    results.append(runner.run(
        "sensor_disagreement", "S1 says crowded, S2 says empty", QueueState.SENSOR_DISAGREEMENT,
        [step(s1=(45, 0.90, 12.0), s2=(2, 0.04, 12.0)) for _ in range(6)]))

    # 9. Sensor 3 confirms downstream congestion
    runner = fresh()
    res = runner.run(
        "downstream_confirms", "All three sensors agree on congestion", QueueState.CONGESTED,
        [step(s1=(45, 0.90, 12.0), s2=(46, 0.92, 12.0), s3=(20, 0.40, 12.0))
         for _ in range(6)],
        notes="downstream sensor present and healthy")
    # extra assertion: downstream sensor acknowledged
    snap = runner.monitor.snapshot()
    if res.passed:
        res.passed = snap.agreement_confidence == 1.0
    results.append(res)

    # 10. Recovery (congested -> empty), hysteresis-aware
    runner = fresh()
    steps = []
    for _ in range(6):
        steps.append(step(s1=(45, 0.90, 12.0), s2=(46, 0.92, 12.0)))
    for _ in range(6):
        steps.append(step(s1=(1, 0.02, 12.0), s2=(1, 0.02, 12.0)))
    results.append(runner.run(
        "recovery", "Congested recovers to empty", QueueState.EMPTY,
        steps, notes="passes through SPARSE before EMPTY"))

    if verbose:
        for r in results:
            print(r.line())
        passed = sum(1 for r in results if r.passed)
        print(f"\n{passed}/{len(results)} scenarios passed")
    return results


if __name__ == "__main__":
    run_all()
