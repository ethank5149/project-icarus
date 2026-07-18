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
