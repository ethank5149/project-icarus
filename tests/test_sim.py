import numpy as np
import pytest
from src.scenarios.target_factory import (
    BallisticScenario,
    FOBSScenario,
    HGVScenario,
    SuppressedScenario,
    SwarmScenario,
    MU_EARTH,
    R_EARTH,
)
from src.scenarios.scenario import EngagementScenario
from src.scenarios.presets import (
    get_interceptor_presets,
    get_target_presets,
    interceptor_preset,
    target_preset,
    set_interceptor_geodetic,
    set_target_geodetic,
    geodetic_to_ecef,
    ecef_to_geodetic,
    _WGS84_A,
    _WGS84_B,
)
from src.sim.api import run_engagement
from src.sim.runner import EngagementRunner
from src.interceptors.config import InterceptorConfig, GuidanceConfig
from src.guidance.law import GuidanceLaw


class TestBallisticScenario:
    def test_propagate_shape(self):
        tgt = BallisticScenario(r0=np.array([R_EARTH, 0.0, 0.0]), v0=np.array([0.0, 1000.0, 0.0]))
        state = tgt.propagate(10.0)
        assert state.shape == (6,)

    def test_energy_conservation(self):
        tgt = BallisticScenario(r0=np.array([R_EARTH, 0.0, 0.0]), v0=np.array([0.0, 1000.0, 0.0]))
        state0 = tgt.propagate(0.0)
        state10 = tgt.propagate(10.0)
        e0 = 0.5 * np.linalg.norm(state0[3:]) ** 2 - MU_EARTH / np.linalg.norm(state0[:3])
        e10 = 0.5 * np.linalg.norm(state10[3:]) ** 2 - MU_EARTH / np.linalg.norm(state10[:3])
        assert np.isclose(e0, e10, rtol=1e-2)


class TestFOBSScenario:
    def test_apoapsis(self):
        tgt = FOBSScenario.from_orbital_params(apoapsis_km=200.0, inclination_deg=0.0)
        assert tgt.apoapsis_km == 200.0

    def test_propagate(self):
        tgt = FOBSScenario(r0=np.array([R_EARTH, 0.0, 0.0]), v0=np.array([0.0, 7000.0, 0.0]))
        state = tgt.propagate(60.0)
        assert state.shape == (6,)


class TestHGVScenario:
    def test_propagate(self):
        tgt = HGVScenario.from_params(max_alt_km=80.0, lateral_range_km=2000.0)
        state = tgt.propagate(10.0)
        assert state.shape == (6,)


class TestSuppressedScenario:
    def test_propagate(self):
        tgt = SuppressedScenario(r0=np.array([R_EARTH, 0.0, 0.0]), v0=np.array([0.0, 800.0, 800.0]))
        state = tgt.propagate(30.0)
        assert state.shape == (6,)


class TestSwarmScenario:
    def test_propagate(self):
        tgt = SwarmScenario.from_params(n_payloads=3, spread_deg=2.0, range_km=500.0)
        state = tgt.propagate(10.0)
        assert state.shape == (6,)

    def test_payload_states(self):
        tgt = SwarmScenario.from_params(n_payloads=3, spread_deg=2.0, range_km=500.0)
        states = tgt.payload_states(10.0)
        assert len(states) == 3
        assert all(s.shape == (6,) for s in states)


class TestEngagementScenario:
    def test_defaults(self):
        s = EngagementScenario()
        assert s.name == "Default"
        assert s.engagement_end == 300.0


class TestSwarmScenario:
    def test_defaults(self):
        s = SwarmScenario()
        assert s.n_payloads == 3


class TestInterceptorPresets:
    def test_get_interceptor_presets(self):
        presets = get_interceptor_presets()
        assert isinstance(presets, dict)
        assert "vandenberg" in presets
        assert "cape_canaveral" in presets
        assert "kwajalein" in presets
        assert "schriever" in presets
        assert "fort_greely" in presets
        assert "clear_sfs" in presets
        assert "custom_geodetic" in presets

    def test_interceptor_preset_values(self):
        vandenberg = interceptor_preset("vandenberg")
        assert isinstance(vandenberg, np.ndarray)
        assert vandenberg.shape == (3,)
        assert np.linalg.norm(vandenberg) > R_EARTH

    def test_interceptor_preset_invalid(self):
        with pytest.raises(KeyError):
            interceptor_preset("nonexistent")

    def test_geodetic_interceptor_presets(self):
        cape = interceptor_preset("cape_canaveral")
        assert isinstance(cape, np.ndarray)
        assert cape.shape == (3,)
        assert np.linalg.norm(cape) > R_EARTH

    def test_set_interceptor_geodetic(self):
        site = set_interceptor_geodetic(34.7, -120.6, 0.0)
        assert isinstance(site, np.ndarray)
        assert site.shape == (3,)
        assert np.linalg.norm(site) > R_EARTH


class TestGeodeticConversion:
    def test_equator_zero_alt(self):
        r = geodetic_to_ecef(0.0, 0.0, 0.0)
        assert np.allclose(r, [_WGS84_A, 0.0, 0.0], atol=1.0)

    def test_north_pole(self):
        r = geodetic_to_ecef(90.0, 0.0, 0.0)
        assert np.allclose(r, [0.0, 0.0, _WGS84_B], atol=1.0)

    def test_roundtrip(self):
        lat, lon, alt = 34.7, -120.6, 100.0
        r = geodetic_to_ecef(lat, lon, alt)
        lat2, lon2, alt2 = ecef_to_geodetic(r)
        assert np.isclose(lat, lat2, atol=1e-4)
        assert np.isclose(lon, lon2, atol=1e-4)
        assert np.isclose(alt, alt2, atol=1.0)

    def test_custom_geodetic_target(self):
        preset = set_target_geodetic(34.7, -120.6, 0.0, v0=np.array([0.0, 1200.0, 0.0]))
        assert preset.name.startswith("custom_geodetic_")
        assert preset.target.r0.shape == (3,)
        assert preset.engagement.engagement_end == 300.0


class TestTargetPresets:
    def test_get_target_presets(self):
        presets = get_target_presets()
        assert isinstance(presets, dict)
        assert "ballistic_short_range" in presets
        assert "fobs_low_orbital" in presets
        assert "hgp_hypersonic_glide" in presets
        assert "suppressed_deep_dip" in presets
        assert "swarm_tight" in presets
        assert "ballistic_target_moscow" in presets
        assert "ballistic_target_beijing" in presets

    def test_target_preset_structure(self):
        preset = target_preset("ballistic_target_moscow")
        assert preset.name == "ballistic_target_moscow"
        assert isinstance(preset.target, BallisticScenario)
        assert isinstance(preset.engagement, EngagementScenario)
        assert preset.description != ""

    def test_target_preset_propagate(self):
        preset = target_preset("hgp_hypersonic_glide")
        state = preset.target.propagate(5.0)
        assert state.shape == (6,)

    def test_target_preset_invalid(self):
        with pytest.raises(KeyError):
            target_preset("nonexistent")


class TestEngagementRunner:
    def test_run_returns_result(self):
        interceptor = InterceptorConfig(name="Test", mass=1000.0)
        guidance = GuidanceLaw()
        target = BallisticScenario(r0=np.array([R_EARTH, 0.0, 0.0]), v0=np.array([0.0, 1000.0, 0.0]))
        scenario = EngagementScenario(engagement_end=60.0)
        runner = EngagementRunner(interceptor=interceptor, guidance=guidance, target=target, scenario=scenario)
        result = runner.run(n_trials=5)
        assert isinstance(result.miss_distance, float)
        assert isinstance(result.kill_assessment, bool)
        if result.monte_carlo:
            assert len(result.monte_carlo.miss_distances) == 5
