import math
import numpy as np
import pytest

from src.guidance.terminal_guidance import TerminalGuidance
from src.guidance.law import GuidanceLaw
from src.guidance.seeker import SeekerModel, SeekerConfig, DiscriminationModel
from src.scenarios.presets import (
    build_interceptor_config,
    interceptor_config_preset,
    sample_interceptor_uq,
    get_interceptor_config_presets,
)


# ---------------------------------------------------------------------------
# 2B.2 — Selectable terminal guidance backends
# ---------------------------------------------------------------------------

class TestGuidanceBackends:
    def setup_method(self):
        self.r = {"r": np.array([0.0, 0.0, 0.0]), "v": np.zeros(3)}
        self.tgt = {
            "r": np.array([100.0, 10.0, 5.0]),
            "v": np.array([-200.0, -20.0, -10.0]),
        }

    def test_valid_laws_accepted(self):
        for law in ("pn", "apn", "zem", "sdre_mpc"):
            t = TerminalGuidance(law=law)
            assert t.law == law

    def test_invalid_law_rejected(self):
        with pytest.raises(ValueError):
            TerminalGuidance(law="lqr_plus")

    def test_out_of_fov_returns_zero(self):
        t = TerminalGuidance(law="zem")
        r = {"r": np.array([0.0, 0.0, 100.0]), "v": np.zeros(3)}
        cmd = t.commanded_accel(0, r, {"r": np.zeros(3), "v": np.zeros(3)})
        assert np.allclose(cmd, 0.0)

    def test_all_backends_finite_and_clipped(self):
        for law in ("pn", "apn", "zem", "sdre_mpc"):
            t = TerminalGuidance(law=law, accel_limit=50.0)
            cmd = t.commanded_accel(0, self.r, self.tgt)
            assert np.all(np.isfinite(cmd))
            assert np.linalg.norm(cmd) <= 50.0 + 1e-9

    def test_apn_adds_gravity_bias_vs_pn(self):
        t_pn = TerminalGuidance(law="pn", accel_limit=1e6)
        t_apn = TerminalGuidance(law="apn", accel_limit=1e6,
                                 gravity=np.array([0.0, 0.0, -9.81]))
        a_pn = t_pn.commanded_accel(0, self.r, self.tgt)
        a_apn = t_apn.commanded_accel(0, self.r, self.tgt)
        assert not np.allclose(a_pn, a_apn)

    def test_backend_via_guidance_law(self):
        cfg = GuidanceLaw.from_dict({"terminal_guidance_law": "sdre_mpc"}).config
        assert cfg.terminal_guidance_law == "sdre_mpc"

    def test_seeker_command_uses_los_rate(self):
        t = TerminalGuidance(law="pn", accel_limit=1e6)
        los_rate = np.array([0.01, -0.005])
        cmd = t.commanded_accel_seeker(self.r, self.tgt, los_rate=los_rate)
        assert np.all(np.isfinite(cmd))


# ---------------------------------------------------------------------------
# 2B.1/runner — Seeker result feeds guidance (unit-level loop)
# ---------------------------------------------------------------------------

class TestSeekerFeedsGuidance:
    def test_seeker_los_rate_drives_command(self):
        cfg = SeekerConfig(mode="ir", fov=math.radians(120.0), range_max=1e6,
                           snr_db=25.0, noise_seed=1)
        seeker = SeekerModel(cfg)
        r = {"r": np.array([0.0, 0.0, 0.0]), "v": np.zeros(3)}
        tgt = {"r": np.array([100.0, 5.0, 0.0]), "v": np.array([-300.0, 0.0, 0.0])}
        law = TerminalGuidance(law="apn", accel_limit=1e6)
        los_rates = []
        cmds = []
        for _ in range(20):
            lr = seeker.update_tracker(r, tgt)
            los_rates.append(lr)
            cmds.append(law.commanded_accel_seeker(r, tgt, los_rate=lr))
        los_rates = np.array(los_rates)
        cmds = np.array(cmds)
        assert np.all(np.isfinite(los_rates))
        assert np.all(np.isfinite(cmds))

    def test_fov_mask_drops_contact(self):
        cfg = SeekerConfig(mode="radar", fov=math.radians(20.0), range_max=1e6,
                           clutter_rate=0.0, noise_seed=0)
        seeker = SeekerModel(cfg)
        r = {"r": np.array([0.0, 0.0, 0.0]), "v": np.zeros(3)}
        tgt = {"r": np.array([100.0, 200.0, 0.0]), "v": np.zeros(3)}
        _, visible = seeker.measure(r, tgt)
        assert visible is False


# ---------------------------------------------------------------------------
# 2B.3 — Calibrated multi-feature discrimination
# ---------------------------------------------------------------------------

class TestDiscrimination:
    def test_rv_vs_decoy_llr(self):
        dm = DiscriminationModel()
        rv = np.array([0.4, 1.6, 75.0, 1.0])
        dec = np.array([-0.4, 0.6, 35.0, 0.0])
        assert dm.is_rv(rv) is True
        assert dm.is_rv(dec) is False
        assert dm.log_likelihood_ratio(rv) > 0.0 > dm.log_likelihood_ratio(dec)

    def test_calibrate_learns_from_labels(self):
        rng = np.random.default_rng(0)
        rv = np.array([0.4, 1.6, 75.0, 1.0])
        dec = np.array([-0.4, 0.6, 35.0, 0.0])
        X = np.vstack([rng.normal(rv, 0.1, (40, 4)),
                       rng.normal(dec, 0.1, (40, 4))])
        labels = np.array([1] * 40 + [0] * 40)
        dm = DiscriminationModel().calibrate(X, labels)
        assert dm.is_rv(rng.normal(rv, 0.05)) is True
        assert dm.is_rv(rng.normal(dec, 0.05)) is False


# ---------------------------------------------------------------------------
# 2B.4 — Interceptor presets with UQ
# ---------------------------------------------------------------------------

class TestInterceptorPresets:
    def test_all_presets_build(self):
        presets = get_interceptor_config_presets()
        for name in ("arrow3", "tamir", "gmd"):
            assert name in presets
            cfg, g = presets[name]
            assert cfg.stages is not None and len(cfg.stages) >= 1
            assert g.terminal_guidance_law in TerminalGuidance.VALID_LAWS

    def test_preset_names_and_backends(self):
        arrow, ga = build_interceptor_config("arrow3")
        assert ga.terminal_guidance_law == "apn"
        assert ga.ukf_enabled is True
        tamir, gt = build_interceptor_config("tamir")
        assert gt.terminal_guidance_law == "sdre_mpc"
        gmd, gg = build_interceptor_config("gmd")
        assert gg.terminal_guidance_law == "zem"

    def test_uq_perturbs_parameters_reproducibly(self):
        rng = np.random.default_rng(42)
        cfg_a, _ = sample_interceptor_uq("gmd", rng=rng, frac=0.20)
        rng = np.random.default_rng(42)
        cfg_b, _ = sample_interceptor_uq("gmd", rng=rng, frac=0.20)
        assert cfg_a.mass == pytest.approx(cfg_b.mass)
        assert cfg_a.stages[0].Isp == pytest.approx(cfg_b.stages[0].Isp)
        nom, _ = build_interceptor_config("gmd")
        assert cfg_a.mass != pytest.approx(nom.mass)

    def test_uq_frac_bounds(self):
        cfg, _ = sample_interceptor_uq("arrow3", rng=np.random.default_rng(0), frac=0.20)
        nom, _ = build_interceptor_config("arrow3")
        assert 0.5 * nom.mass < cfg.mass < 2.0 * nom.mass
