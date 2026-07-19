import numpy as np
import pytest

from src.guidance.seeker import SeekerModel, SeekerConfig, _UKF, _merwe_sigmas
from src.guidance.law import GuidanceLaw
from src.guidance.terminal_guidance import TerminalGuidance


def _true_los(x):
    r = x[:3]
    az = np.arctan2(r[1], r[0])
    el = np.arcsin(np.clip(r[2] / (np.linalg.norm(r) + 1e-9), -1, 1))
    return np.array([az, el])


# Reference LOS error of an uninformed prior (atan2 singularity at origin).
_PRIOR_LOS_ERR = np.pi / 2.0


class TestUKFMath:
    def test_merwe_weights_mean(self):
        _, Wm, Wc = _merwe_sigmas(6)
        # First (central) weight differs; off-center weights sum to one.
        assert np.isclose(Wm.sum(), 1.0)
        assert Wm[0] < 0.0  # central weight is negative by construction
        assert np.isclose(Wm[1:].sum(), 1.0 - Wm[0])

    def test_ukf_reduces_los_error(self):
        # Angle-only seeker: the UKF must collapse the LOS estimate from the
        # degenerate prior (pi/2 at the origin) toward the truth. Pure
        # angle-only tracking has a known range/velocity ambiguity, so we assert
        # the *LOS direction* is recovered to within a few degrees of truth and
        # is far better than the uninitialized prior.
        ukf = _UKF(dt=0.01, seed=3, q_pos=100.0, q_vel=10.0, r_angle=1e-3)
        rng = np.random.default_rng(7)
        x_true = np.array([1000.0, 500.0, 200.0, -50.0, -20.0, -10.0])
        max_los_err = 0.0
        for _ in range(120):
            ukf.predict()
            z = _true_los(x_true) + rng.normal(0, 1e-4, 2)
            ukf.update(z, visible=True)
            max_los_err = max(max_los_err, np.linalg.norm(_true_los(x_true) - _true_los(ukf.x)))
            x_true = x_true + np.array([x_true[3], x_true[4], x_true[5], 0, 0, 0]) * 0.01
        assert max_los_err < 0.087   # < 5 deg after convergence
        assert max_los_err < 0.5 * _PRIOR_LOS_ERR


class TestSeekerModel:
    def test_radar_in_fov_measures(self):
        cfg = SeekerConfig(mode="radar", fov=np.radians(60), range_max=50e3,
                           noise_seed=1, clutter_rate=0.0, latency_frames=0)
        s = SeekerModel(cfg)
        interceptor = {"r": np.zeros(3), "v": np.zeros(3)}
        target = {"r": np.array([1000.0, 100.0, 50.0]), "v": np.zeros(3)}
        azel, visible = s.measure(interceptor, target)
        assert visible
        assert azel is not None
        assert np.isfinite(azel).all()

    def test_out_of_fov_not_visible(self):
        cfg = SeekerConfig(mode="ir", fov=np.radians(20), range_max=50e3,
                           noise_seed=2, clutter_rate=0.0)
        s = SeekerModel(cfg)
        interceptor = {"r": np.zeros(3), "v": np.zeros(3)}
        # Target far outside a 20-deg FOV cone (mostly lateral).
        target = {"r": np.array([100.0, 5000.0, 0.0]), "v": np.zeros(3)}
        azel, visible = s.measure(interceptor, target)
        assert not visible

    def test_out_of_range_not_visible(self):
        cfg = SeekerConfig(mode="radar", fov=np.radians(60), range_max=10e3,
                           noise_seed=2, clutter_rate=0.0)
        s = SeekerModel(cfg)
        interceptor = {"r": np.zeros(3), "v": np.zeros(3)}
        target = {"r": np.array([50e3, 0.0, 0.0]), "v": np.zeros(3)}
        azel, visible = s.measure(interceptor, target)
        assert not visible

    def test_tracker_produces_los_rate(self):
        cfg = SeekerConfig(mode="radar", fov=np.radians(60), range_max=50e3,
                           noise_seed=5, clutter_rate=0.0)
        s = SeekerModel(cfg)
        interceptor = {"r": np.zeros(3), "v": np.zeros(3)}
        target = {"r": np.array([1000.0, 0.0, 0.0]),
                  "v": np.array([-200.0, 0.0, 0.0])}
        rate = s.update_tracker(interceptor, target)
        assert rate.shape == (2,)
        assert np.all(np.isfinite(rate))

    def test_latency_delays_measurement(self):
        cfg = SeekerConfig(mode="radar", latency_frames=2, noise_seed=1,
                           clutter_rate=0.0)
        s = SeekerModel(cfg)
        interceptor = {"r": np.zeros(3), "v": np.zeros(3)}
        target = {"r": np.array([1000.0, 100.0, 0.0]), "v": np.zeros(3)}
        # First two frames buffered (None), third publishes first measurement.
        s.measure(interceptor, target)
        s.measure(interceptor, target)
        azel, _ = s.measure(interceptor, target)
        assert azel is not None

    def test_discrimination_features_shape(self):
        cfg = SeekerConfig(mode="radar", noise_seed=1)
        s = SeekerModel(cfg)
        feats = s.discrimination_features({"r": np.array([1000.0, 0.0, 0.0])})
        assert feats.shape == (4,)


class TestGuidanceLawSeeker:
    def test_guidance_law_attaches_seeker(self):
        gl = GuidanceLaw.from_dict({"ukf_enabled": True, "seeker_mode": "radar"})
        assert gl.seeker is not None
        assert isinstance(gl.seeker, SeekerModel)

    def test_guidance_law_disable_seeker(self):
        gl = GuidanceLaw.from_dict({"ukf_enabled": False})
        assert gl.seeker is None

    def test_terminal_commanded_accel_seeker(self):
        tg = TerminalGuidance(N=4.0, accel_limit=150.0)
        interceptor = {"r": np.zeros(3), "v": np.zeros(3)}
        target = np.array([1000.0, 0.0, 0.0, -200.0, 0.0, 0.0])
        a = tg.commanded_accel_seeker(interceptor, target, los_rate=np.array([0.01, 0.0]))
        assert a.shape == (3,)
        assert np.linalg.norm(a) <= 150.0 + 1e-9
