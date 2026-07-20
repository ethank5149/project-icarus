import numpy as np
import pytest

from project_icarus.scenarios.target_factory import (
    BallisticScenario,
    DecoyThreatScenario,
    ThreatSignatureLibrary,
)
from project_icarus.targets.decoy_model import DecoyModel
from project_icarus.guidance.law import GuidanceLaw
from project_icarus.guidance.seeker import DiscriminationModel
from project_icarus.scenarios.presets import build_interceptor_config
from project_icarus.scenarios.scenario import EngagementScenario
from project_icarus.sim.api import run_engagement


class TestThreatSignatureLibrary:
    def test_default_builds_labelled_data(self):
        lib = ThreatSignatureLibrary.default(n=30, seed=1)
        X, y = lib.labelled_matrix()
        assert X.shape == (60, 4)
        assert set(y.tolist()) == {0, 1}

    def test_rv_and_decoy_separate(self):
        lib = ThreatSignatureLibrary.default(n=60, seed=2)
        X, y = lib.labelled_matrix()
        dm = DiscriminationModel().calibrate(X, y)
        rv_idx = [i for i, v in enumerate(y) if v == 1]
        dec_idx = [i for i, v in enumerate(y) if v == 0]
        rv_correct = sum(dm.is_rv(X[i]) for i in rv_idx)
        dec_correct = sum(not dm.is_rv(X[i]) for i in dec_idx)
        # Calibrated model should classify the majority of each class correctly.
        assert rv_correct > 0.8 * len(rv_idx)
        assert dec_correct > 0.8 * len(dec_idx)


class TestDecoyThreatScenario:
    def _make(self):
        rv = BallisticScenario(r0=np.array([6371e3, 0.0, 0.0]),
                               v0=np.array([0.0, 1500.0, 600.0]))
        dts = DecoyThreatScenario(
            rv=rv,
            decoys=[{"radar_rcs_bias": -0.4, "ir_bias": 0.6},
                    {"radar_rcs_bias": -0.5, "ir_bias": 0.5}],
            release_t=50.0,
        )
        return dts

    def test_decoys_inactive_before_release(self):
        dts = self._make()
        assert all(s is None for s in dts.decoy_states(10.0))

    def test_decoys_active_after_release(self):
        dts = self._make()
        states = dts.decoy_states(120.0)
        assert all(s is not None for s in states)
        # DecoyModel.state_at returns (r, v), each a 3-vector.
        assert all(len(s) == 2 and all(len(vec) == 3 for vec in s) for s in states)

    def test_decoy_features_consistent_with_discriminator(self):
        dts = self._make()
        feats = dts.decoy_features(120.0, seed=0)
        assert len(feats) == 2
        for f in feats:
            assert f.shape == (4,)
            assert f[3] in (0.0, 1.0)  # micro-motion flag

    def test_rv_propagates_like_ballistic(self):
        dts = self._make()
        rv = BallisticScenario(r0=np.array([6371e3, 0.0, 0.0]),
                               v0=np.array([0.0, 1500.0, 600.0]))
        assert np.allclose(dts.propagate(80.0), rv.propagate(80.0))


class TestDecoyModelFeatures:
    def test_feature_vector_shape(self):
        d = DecoyModel()
        f = d.discrimination_features(np.random.default_rng(0))
        assert f.shape == (4,)
        assert f[3] in (0.0, 1.0)

    def test_release_sets_state(self):
        d = DecoyModel()
        r0 = np.array([6371e3, 50e3, 200e3])
        v0 = np.array([0.0, 1500.0, 600.0])
        d.release(100.0, r=r0, v=v0)
        assert d.active
        # state_at takes one Euler step forward from the released state; the
        # returned (position, velocity) should be finite and Earth-scale.
        r, v = d.state_at(100.0)
        assert np.all(np.isfinite(r)) and np.all(np.isfinite(v))
        assert np.linalg.norm(r) > 6.0e6
        assert np.linalg.norm(v) > 100.0


class TestGuidanceDiscriminator:
    def test_guidance_law_carries_calibrated_discriminator(self):
        gl = GuidanceLaw()
        assert gl.discriminate_target(np.array([0.4, 1.6, 75.0, 1.0])) is True
        assert gl.discriminate_target(np.array([-0.4, 0.6, 35.0, 0.0])) is False

    def test_decoy_engagement_runs_with_discrimination(self):
        rv = BallisticScenario(r0=np.array([6371e3, 0.0, 0.0]),
                               v0=np.array([0.0, 1500.0, 600.0]))
        dts = DecoyThreatScenario(rv=rv, decoys=[{"radar_rcs_bias": -0.4, "ir_bias": 0.6}],
                                  release_t=40.0)
        cfg, g = build_interceptor_config("arrow3")
        gl = GuidanceLaw(config=g)
        res = run_engagement(
            cfg, gl, dts,
            EngagementScenario(engagement_end=200.0), n_trials=2,
        )
        assert np.isfinite(res.miss_distance)
        assert gl._decoy_rejects >= 0
