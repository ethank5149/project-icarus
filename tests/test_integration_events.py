import numpy as np
import pytest
from src.interceptors.config import InterceptorConfig
from src.guidance.law import GuidanceLaw
from src.scenarios.target_factory import BallisticScenario
from src.scenarios.scenario import EngagementScenario
from src.sim.runner import (
    EngagementRunner,
    ThrustCutoffEvent,
    ReentryEvent,
    RangeEvent,
    _current_phase,
)
from src.sim.config import SimConfig
from src.dynamics.thrust import StageSpec, StageSeparation


R_EARTH = 6371e3


def _make_two_stage_interceptor():
    """A two-stage interceptor with sequential burns and separation."""
    s1 = StageSpec(
        thrust=lambda t: 20000.0 if t < 30.0 else 0.0,
        burn_time=30.0,
        wet_mass=1200.0,
        dry_mass=200.0,
        Isp=260.0,
        name="booster",
    )
    s2 = StageSpec(
        thrust=lambda t: 6000.0 if t < 60.0 else 0.0,
        burn_time=60.0,
        wet_mass=400.0,
        dry_mass=100.0,
        Isp=280.0,
        name="sustainer",
    )
    ic = InterceptorConfig(
        name="TwoStage",
        mass=2000.0,
        area=0.3,
        ref_length=7.0,
        stages=[s1, s2],
        sep_impulses=[np.zeros(3), np.zeros(3)],
    )
    return ic


class TestPhaseEvents:
    def test_thrust_cutoff_triggers_on_low_thrust(self):
        ev = ThrustCutoffEvent(frac=1e-3)
        ctx = {"peak_thrust": 20000.0, "thrust": 1.0, "phase": "boost", "dry_mass": -1.0}
        y = np.zeros(14)
        assert ev.should_trigger(31.0, y, ctx)
        ctx["thrust"] = 20000.0
        assert not ev.should_trigger(5.0, y, ctx)

    def test_thrust_cutoff_triggers_on_dry_mass(self):
        ev = ThrustCutoffEvent(frac=1e-3)
        y = np.zeros(14)
        y[13] = 150.0
        ctx = {"peak_thrust": 0.0, "thrust": 0.0, "phase": "boost", "dry_mass": 200.0}
        assert ev.should_trigger(40.0, y, ctx)

    def test_reentry_event(self):
        ev = ReentryEvent(alt=100e3)
        y = np.zeros(14)
        y[:3] = np.array([R_EARTH + 50e3, 0.0, 0.0])
        assert ev.should_trigger(100.0, y, {"phase": "midcourse"})
        y[:3] = np.array([R_EARTH + 150e3, 0.0, 0.0])
        assert not ev.should_trigger(100.0, y, {"phase": "midcourse"})

    def test_range_event(self):
        ev = RangeEvent(range_=50e3)
        y = np.zeros(14)
        y[:3] = np.array([R_EARTH + 100e3, 0.0, 0.0])
        target = np.array([R_EARTH + 100e3 + 40e3, 0.0, 0.0, 0, 0, 0])
        assert ev.should_trigger(100.0, y, {"phase": "midcourse", "target_state": target})
        target = np.array([R_EARTH + 200e3, 0.0, 0.0, 0, 0, 0])
        assert not ev.should_trigger(100.0, y, {"phase": "midcourse", "target_state": target})

    def test_phase_monotonic_boost_to_terminal(self):
        events = [
            ThrustCutoffEvent(frac=1e-3, next_phase="midcourse"),
            ReentryEvent(alt=100e3, next_phase="terminal"),
        ]
        y = np.zeros(14)
        ctx = {"phase": "boost", "peak_thrust": 0.0, "thrust": 0.0, "dry_mass": -1.0}
        assert _current_phase(0.0, y, events, ctx) == "boost"
        events2 = [ReentryEvent(alt=100e3, next_phase="terminal")]
        ctx2 = {"phase": "midcourse", "target_state": None}
        y[:3] = np.array([R_EARTH + 50e3, 0.0, 0.0])
        assert _current_phase(100.0, y, events2, ctx2) == "terminal"


class TestMultiStageThrust:
    def test_stage_index_and_thrust_profile(self):
        ic = _make_two_stage_interceptor()
        assert ic.peak_thrust == 20000.0
        assert ic.dry_mass == 300.0
        thrust_fn = ic._thrust_callable
        # Stage 1 burns 0-30 s; stage 2 ignites immediately after at t=30.
        assert thrust_fn(5.0, None) == 20000.0
        assert thrust_fn(35.0, None) == 6000.0  # stage 2 burning
        assert thrust_fn(95.0, None) == 0.0  # all burnt out

    def test_separations_built(self):
        ic = _make_two_stage_interceptor()
        seps = ic._separations
        assert len(seps) == 2
        assert abs(seps[0].mass_drop - 1000.0) < 1e-6
        assert abs(seps[1].mass_drop - 300.0) < 1e-6
        assert abs(seps[0].time - 30.0) < 1e-6
        assert abs(seps[1].time - 90.0) < 1e-6

    def test_separation_applies_mass_drop(self):
        sep = StageSeparation(time=0.0, mass_drop=500.0, impulse=np.array([10.0, 0.0, 0.0]))
        state = {"r": np.zeros(3), "v": np.zeros(3), "q": np.array([1.0, 0, 0, 0]),
                 "omega": np.zeros(3), "m": 1000.0}
        new = sep.apply(state)
        assert abs(new["m"] - 500.0) < 1e-6
        assert np.allclose(new["v"], [10.0 / 1000.0, 0.0, 0.0])


class TestRngDeterminism:
    def test_same_seed_reproducible(self):
        ic = _make_two_stage_interceptor()
        guidance = GuidanceLaw()
        target = BallisticScenario(r0=np.array([R_EARTH, 0.0, 0.0]), v0=np.array([0.0, 1500.0, 0.0]))
        scenario = EngagementScenario(
            interceptor_launch_site=np.array([R_EARTH + 100e3, 0.0, 0.0]),
            engagement_end=80.0,
        )
        cfg = SimConfig(seed=7)
        r1 = EngagementRunner(ic, guidance, target, scenario, cfg=cfg).run(n_trials=4)
        r2 = EngagementRunner(ic, guidance, target, scenario, cfg=cfg).run(n_trials=4)
        assert r1.miss_distance == r2.miss_distance
        assert r1.monte_carlo.miss_distances == r2.monte_carlo.miss_distances

    def test_different_seed_differs(self):
        ic = _make_two_stage_interceptor()
        guidance = GuidanceLaw()
        target = BallisticScenario(r0=np.array([R_EARTH, 0.0, 0.0]), v0=np.array([0.0, 1500.0, 0.0]))
        scenario = EngagementScenario(
            interceptor_launch_site=np.array([R_EARTH + 100e3, 0.0, 0.0]),
            engagement_end=80.0,
        )
        r1 = EngagementRunner(ic, guidance, target, scenario, cfg=SimConfig(seed=1)).run(n_trials=4)
        r2 = EngagementRunner(ic, guidance, target, scenario, cfg=SimConfig(seed=2)).run(n_trials=4)
        assert r1.monte_carlo.miss_distances != r2.monte_carlo.miss_distances


class TestMultiStageEngagementRuns:
    def test_two_stage_runs_with_separations(self):
        ic = _make_two_stage_interceptor()
        guidance = GuidanceLaw()
        target = BallisticScenario(r0=np.array([R_EARTH, 0.0, 0.0]), v0=np.array([0.0, 1500.0, 0.0]))
        scenario = EngagementScenario(
            interceptor_launch_site=np.array([R_EARTH + 100e3, 0.0, 0.0]),
            engagement_end=120.0,
        )
        runner = EngagementRunner(ic, guidance, target, scenario)
        result = runner.run(n_trials=2)
        assert np.isfinite(result.miss_distance)
        final_mass = result.nominal_trajectory["m"][-1]
        assert final_mass < 2000.0
        assert 50.0 < final_mass < 1000.0
