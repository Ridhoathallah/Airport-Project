"""Unit tests for the state machine (hysteresis + debounce)."""

import unittest

from core.config import Settings
from core.models import MetricsSnapshot, QueueState
from core.state_machine import QueueStateMachine


def snapshot(density: float, people: int = 1, occupancy: float = 0.0) -> MetricsSnapshot:
    return MetricsSnapshot(
        timestamp="t", people_count=people, density=density, occupancy_percent=occupancy)


class TestStateMachine(unittest.TestCase):
    def setUp(self):
        self.settings = Settings.load()
        self.machine = QueueStateMachine(self.settings)

    def _feed(self, dens, n=1, occ=0.0):
        return self.machine.update(snapshot(dens, n, occ))

    def _sustain(self, dens, frames=5, n=1, occ=0.0):
        for _ in range(frames):
            self._feed(dens, n, occ)
        return self.machine.state

    def test_initial_state_is_data_unavailable(self):
        self.assertEqual(self.machine.state, QueueState.DATA_UNAVAILABLE)

    def test_empty_at_zero_density(self):
        self._feed(0.0)
        self.assertEqual(self.machine.state, QueueState.EMPTY)

    def test_ladder_classification(self):
        self.assertEqual(self._sustain(0.12), QueueState.SPARSE)
        self.assertEqual(self._sustain(0.45), QueueState.NORMAL)
        self.assertEqual(self._sustain(0.65), QueueState.CROWDED)
        self.assertEqual(self._sustain(0.85), QueueState.CONGESTED)

    def test_critical_at_full_capacity(self):
        self._feed(0.90, occ=100.0)
        self.assertEqual(self.machine.state, QueueState.CRITICAL)

    def test_debounce_requires_consecutive_frames(self):
        self._sustain(0.0)  # commit EMPTY baseline first
        for _ in range(4):
            self.assertEqual(self._feed(0.65), QueueState.EMPTY)
        self.assertEqual(self._feed(0.65), QueueState.CROWDED)

    def test_noise_does_not_commit(self):
        self._sustain(0.0)
        for _ in range(4):
            self._feed(0.65)
        self._feed(0.45)  # outlier interrupts the candidate
        for _ in range(4):
            self._feed(0.65)
        self.assertEqual(self.machine.state, QueueState.EMPTY)

    def test_hysteresis_blocks_oscillation(self):
        self._sustain(0.65)
        self.assertEqual(self.machine.state, QueueState.CROWDED)
        # Drop to 0.62: still above crowded(0.60) -> stays CROWDED.
        self._sustain(0.62)
        self.assertEqual(self.machine.state, QueueState.CROWDED)
        # Now clearly below crowded(0.60) - hysteresis(0.05) = 0.55
        self._sustain(0.50)
        self.assertEqual(self.machine.state, QueueState.NORMAL)

    def test_recovery_from_no_data_is_immediate(self):
        self.assertEqual(self.machine.state, QueueState.DATA_UNAVAILABLE)
        self.assertEqual(self._feed(0.45), QueueState.NORMAL)  # no debounce wait


if __name__ == "__main__":
    unittest.main()
