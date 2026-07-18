import numpy as np
import pytest
from src.guidance.boost_guidance import BoostGuidance
from src.guidance.midcourse_guidance import MidcourseGuidance
from src.guidance.terminal_guidance import TerminalGuidance


class TestBoostGuidance:
    def test_initial_gimbal(self):
        g = BoostGuidance()
        state = {"r": np.zeros(3), "v": np.zeros(3), "q": np.array([1.0, 0.0, 0.0, 0.0]), "omega": np.zeros(3), "m": 1000.0}
        cmd = g.commanded_gimbal(0.0, state, 0.0, np.zeros(3))
        assert np.allclose(cmd, [0.0, 0.0])

    def test_pitch_over(self):
        g = BoostGuidance(pitch_over_q=1.0)
        state = {"r": np.zeros(3), "v": np.array([100.0, 0.0, 0.0]), "q": np.array([1.0, 0.0, 0.0, 0.0]), "omega": np.zeros(3), "m": 1000.0}
        cmd = g.commanded_gimbal(0.0, state, 1.225, np.array([100.0, 0.0, 0.0]))
        assert np.isclose(cmd[0], np.radians(5.0))


class TestMidcourseGuidance:
    def test_no_target(self):
        g = MidcourseGuidance()
        state = {"r": np.zeros(3), "v": np.zeros(3)}
        accel = g.commanded_accel(0.0, state, 0.0, 1000.0)
        assert np.allclose(accel, 0.0)

    def test_with_target(self):
        g = MidcourseGuidance(N=5.0, accel_limit=50.0)
        g.update_target(np.array([10000.0, 10000.0, 0.0, -200.0, 0.0, 0.0]))
        state = {"r": np.zeros(3), "v": np.zeros(3)}
        accel = g.commanded_accel(0.0, state, 0.0, 10000.0)
        assert np.linalg.norm(accel) > 0


class TestTerminalGuidance:
    def test_no_target(self):
        g = TerminalGuidance()
        state = {"r": np.zeros(3), "v": np.zeros(3)}
        accel = g.commanded_accel(0.0, state, None, 1000.0, 1000.0)
        assert np.allclose(accel, 0.0)

    def test_hit_to_kill_assessment(self):
        g = TerminalGuidance(kill_radius=0.5, mechanism="hit_to_kill")
        assert g.kill_assessment(0.3) is True
        assert g.kill_assessment(1.0) is False

    def test_blast_frag_assessment(self):
        g = TerminalGuidance(kill_radius=0.5, mechanism="blast_frag")
        assert g.kill_assessment(5.0) is True
        assert g.kill_assessment(15.0) is False
