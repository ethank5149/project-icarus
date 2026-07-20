import numpy as np
import pytest
from project_icarus.guidance.boost_guidance import BoostGuidance
from project_icarus.guidance.midcourse_guidance import MidcourseGuidance
from project_icarus.guidance.terminal_guidance import TerminalGuidance


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

    def test_pn_sign_convention(self):
        # Off-axis closing geometry produces a LOS rotation; PN accel is
        # N*Vc*lambda_dot, opposing the LOS rate (steers toward collision course).
        g = MidcourseGuidance(N=5.0)
        # Target ahead in +x but drifting in +y; interceptor closing in -x.
        g.update_target(np.array([1000.0, 0.0, 0.0, -200.0, 100.0, 0.0]))
        state = {"r": np.zeros(3), "v": np.zeros(3)}
        accel = g.commanded_accel(0.0, state)
        # Nonzero command due to LOS rotation from the +y drift.
        assert np.linalg.norm(accel) > 0
        # PN steers toward the target's lateral drift: positive y command.
        assert accel[1] > 0

    def test_data_link_delay(self):
        g = MidcourseGuidance(update_interval=2.0)
        g.update_target(np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0]), t=0.0)
        # Update at t=1.0 ignored (before interval elapses).
        g.update_target(np.array([999.0, 0.0, 0.0, 0.0, 0.0, 0.0]), t=1.0)
        assert np.allclose(g.target_state[:3], [100.0, 0.0, 0.0])
        # Update at t=2.5 applied.
        g.update_target(np.array([500.0, 0.0, 0.0, 0.0, 0.0, 0.0]), t=2.5)
        assert np.allclose(g.target_state[:3], [500.0, 0.0, 0.0])

    def test_seeker_fov(self):
        g = TerminalGuidance(fov=np.radians(30.0))
        # LOS within FOV -> nonzero command toward target.
        state = {"r": np.zeros(3), "v": np.zeros(3)}
        target_in = np.array([1000.0, 100.0, 0.0, -100.0, 0.0, 0.0])
        a_in = g.commanded_accel(0.0, state, target_in)
        assert np.linalg.norm(a_in) > 0
        # LOS outside FOV -> zero command.
        target_out = np.array([100.0, 2000.0, 0.0, 0.0, -100.0, 0.0])
        a_out = g.commanded_accel(0.0, state, target_out)
        assert np.allclose(a_out, 0.0)

    def test_discrimination(self):
        g = TerminalGuidance()
        features = np.array([0.9, 1.2, 0.3, 1.0])
        rv = lambda f: np.exp(-0.5 * np.sum((f - np.array([1.0, 1.0, 0.2, 1.0]))**2))
        decoy = lambda f: np.exp(-0.5 * np.sum((f - np.array([0.5, 0.5, 0.8, 0.0]))**2))
        assert g.discrimination(features, rv, decoy) is True
