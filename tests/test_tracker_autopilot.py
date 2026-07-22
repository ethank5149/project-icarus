import math
import numpy as np
import pytest

from project_icarus.guidance.tracker import TargetTracker, TrackerConfig
from project_icarus.guidance.autopilot import Autopilot, AutopilotConfig


class TestTargetTracker:
    def test_first_update_seeds_state(self):
        tracker = TargetTracker(TrackerConfig(dt=1.0))
        meas = np.array([1000.0, 2000.0, 3000.0])
        x = tracker.update(0.0, meas)
        assert np.allclose(tracker.position(), meas)
        assert np.allclose(tracker.velocity(), np.zeros(3))
        assert np.allclose(tracker.acceleration(), np.zeros(3))

    def test_constant_velocity_tracking(self):
        cfg = TrackerConfig(dt=1.0, sigma_meas=0.1, q_pos=1e-6, q_vel=1e-6, q_accel=1e-6)
        tracker = TargetTracker(cfg)
        true_pos = np.array([0.0, 0.0, 0.0])
        true_vel = np.array([100.0, 0.0, 0.0])
        true_accel = np.array([0.0, 0.0, 0.0])
        rng = np.random.default_rng(0)
        for k in range(20):
            true_pos = true_pos + true_vel * cfg.dt + 0.5 * true_accel * cfg.dt ** 2
            true_vel = true_vel + true_accel * cfg.dt
            meas = true_pos + rng.normal(0, cfg.sigma_meas, 3)
            tracker.update(k * cfg.dt, meas)
        est_pos = tracker.position()
        est_vel = tracker.velocity()
        assert np.linalg.norm(est_pos - true_pos) < 10.0
        assert np.linalg.norm(est_vel - true_vel) < 5.0

    def test_constant_acceleration_tracking(self):
        cfg = TrackerConfig(dt=0.5, sigma_meas=0.5, q_pos=0.1, q_vel=0.1, q_accel=1.0)
        tracker = TargetTracker(cfg)
        true_pos = np.array([0.0, 0.0, 0.0])
        true_vel = np.array([0.0, 0.0, 0.0])
        true_accel = np.array([10.0, 0.0, 0.0])
        rng = np.random.default_rng(1)
        for k in range(30):
            true_pos = true_pos + true_vel * cfg.dt + 0.5 * true_accel * cfg.dt ** 2
            true_vel = true_vel + true_accel * cfg.dt
            meas = true_pos + rng.normal(0, cfg.sigma_meas, 3)
            tracker.update(k * cfg.dt, meas)
        est_accel = tracker.acceleration()
        assert np.linalg.norm(est_accel - true_accel) < 3.0

    def test_covariance_remains_positive_definite(self):
        tracker = TargetTracker(TrackerConfig(dt=1.0))
        for k in range(50):
            meas = np.array([k * 10.0, k * 5.0, k * 2.0])
            tracker.update(float(k), meas)
            P = tracker.covariance()
            eigvals = np.linalg.eigvalsh(P)
            assert np.all(eigvals >= -1e-9)

    def test_reset_clears_state(self):
        tracker = TargetTracker(TrackerConfig(dt=1.0))
        tracker.update(0.0, np.array([100.0, 200.0, 300.0]))
        tracker.reset()
        assert np.allclose(tracker.position(), np.zeros(3))
        assert not tracker.ukf._initialized

    def test_los_rate_from_tracker(self):
        tracker = TargetTracker(TrackerConfig(dt=1.0))
        tracker.update(0.0, np.array([1000.0, 0.0, 0.0]))
        r_i = np.array([0.0, 0.0, 0.0])
        los_rate = tracker.estimated_los_rate(r_i)
        assert los_rate.shape == (2,)


class TestAutopilot:
    def test_second_order_step_response(self):
        cfg = AutopilotConfig(omega_n=10.0, damping=0.7, accel_limit=1e6)
        ap = Autopilot(cfg)
        dt = 0.01
        a_cmd = np.array([100.0, 0.0, 0.0])
        for _ in range(500):
            ap.update(a_cmd, dt, quat=np.array([1.0, 0.0, 0.0, 0.0]))
        assert np.linalg.norm(ap._a_real - a_cmd) < 1.0

    def test_body_rate_feedback_damps_response(self):
        cfg = AutopilotConfig(omega_n=50.0, damping=0.5, accel_limit=1e6, rate_gain=0.0)
        ap_no_feedback = Autopilot(cfg)
        cfg2 = AutopilotConfig(omega_n=50.0, damping=0.5, accel_limit=1e6, rate_gain=10.0)
        ap_feedback = Autopilot(cfg2)
        dt = 0.01
        a_cmd = np.array([100.0, 0.0, 0.0])
        omega = np.array([1.0, 0.0, 0.0])
        for _ in range(200):
            ap_no_feedback.update(a_cmd, dt, quat=np.array([1.0, 0.0, 0.0, 0.0]), omega_body=np.zeros(3))
            ap_feedback.update(a_cmd, dt, quat=np.array([1.0, 0.0, 0.0, 0.0]), omega_body=omega)
        overshoot_no_fb = float(np.max(np.linalg.norm(ap_no_feedback._a_real, axis=0)))
        overshoot_fb = float(np.max(np.linalg.norm(ap_feedback._a_real, axis=0)))
        assert overshoot_fb < overshoot_no_fb

    def test_gimbal_limit(self):
        cfg = AutopilotConfig(omega_n=10.0, damping=0.7, accel_limit=1e6,
                              gimbal_limit_deg=10.0, use_gimbal_limit=True)
        ap = Autopilot(cfg)
        a_cmd = np.array([1000.0, 1000.0, 0.0])
        ap.update(a_cmd, 0.01, quat=np.array([1.0, 0.0, 0.0, 0.0]))
        C = Autopilot._quat_to_dcm(np.array([1.0, 0.0, 0.0, 0.0]))
        v_body = C.T @ ap._a_real
        gimbal_mag = math.degrees(math.sqrt(v_body[1] ** 2 + v_body[2] ** 2) / np.linalg.norm(v_body))
        assert gimbal_mag <= cfg.gimbal_limit_deg + 1e-6

    def test_rate_limit(self):
        cfg = AutopilotConfig(omega_n=10.0, damping=0.0, accel_rate_limit=10.0, accel_limit=1e6)
        ap = Autopilot(cfg)
        dt = 0.01
        a_cmd_small = np.array([0.0, 0.0, 0.0])
        a_cmd_large = np.array([100.0, 0.0, 0.0])
        ap.update(a_cmd_small, dt, quat=np.array([1.0, 0.0, 0.0, 0.0]))
        for _ in range(5):
            ap.update(a_cmd_large, dt, quat=np.array([1.0, 0.0, 0.0, 0.0]))
        max_norm = np.linalg.norm(ap._a_real)
        assert max_norm < cfg.accel_rate_limit * 5 * 0.1 + 1e-6

    def test_saturation(self):
        cfg = AutopilotConfig(omega_n=1000.0, damping=0.0, accel_limit=50.0)
        ap = Autopilot(cfg)
        a_cmd = np.array([1000.0, 0.0, 0.0])
        ap.update(a_cmd, 0.01, quat=np.array([1.0, 0.0, 0.0, 0.0]))
        assert np.linalg.norm(ap._a_real) <= 50.0 + 1e-9

    def test_reset_clears_states(self):
        ap = Autopilot()
        ap.update(np.array([100.0, 0.0, 0.0]), 0.01, quat=np.array([1.0, 0.0, 0.0, 0.0]))
        ap.reset()
        assert np.allclose(ap._a_real, np.zeros(3))
        assert np.allclose(ap._a_real_dot, np.zeros(3))
        assert ap._gimbal_az == 0.0
        assert ap._gimbal_el == 0.0
        assert not ap._initialized
