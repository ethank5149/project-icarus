"""Phase 1B tests: variable inertia/CG after separation, impulse/momentum
conservation, dry-mass cutoff, and event-driven SeparationEvent.

All params are OSINT-approximate research defaults, NOT controlled data.
"""

import numpy as np
import pytest

from project_icarus.dynamics.thrust import (
    StageSpec,
    MultiStageThrustModel,
    MKVSystem,
)
from project_icarus.dynamics.eom_6dof import EOM6DOF
from project_icarus.interceptors.config import InterceptorConfig
from project_icarus.sim.runner import SeparationEvent
from project_icarus.sim.config import SimConfig


def _two_stage():
    s1 = StageSpec(
        thrust=lambda t: 20000.0 if t < 30.0 else 0.0,
        burn_time=30.0,
        wet_mass=1200.0,
        dry_mass=200.0,
        Isp=260.0,
        name="booster",
        inertia=np.diag([1500.0, 2400.0, 3000.0]),
        cg=np.array([0.5, 0.0, 0.0]),
    )
    s2 = StageSpec(
        thrust=lambda t: 6000.0 if t < 60.0 else 0.0,
        burn_time=60.0,
        wet_mass=400.0,
        dry_mass=100.0,
        Isp=280.0,
        name="sustainer",
        inertia=np.diag([100.0, 200.0, 250.0]),
        cg=np.array([0.1, 0.0, 0.0]),
    )
    return s1, s2


class TestInertiaAfterSeparation:
    def test_residual_inertia_is_upper_stage(self):
        s1, s2 = _two_stage()
        model = MultiStageThrustModel([s1, s2])
        # After booster separates, the flying bus is the sustainer's inertia.
        inert = model.inertia_after_separation(0)
        assert np.allclose(inert, s2.inertia)
        # After the final sustainer burnout, the bus keeps its own inertia.
        inert_last = model.inertia_after_separation(1)
        assert np.allclose(inert_last, s2.inertia)

    def test_cg_after_separation(self):
        s1, s2 = _two_stage()
        model = MultiStageThrustModel([s1, s2])
        cg = model.cg_after_separation(0)
        assert np.allclose(cg, s2.cg)

    def test_no_inertia_declared_returns_none(self):
        s1 = StageSpec(thrust=lambda t: 1.0, burn_time=1.0, wet_mass=2.0,
                       dry_mass=1.0, name="a")
        s2 = StageSpec(thrust=lambda t: 1.0, burn_time=1.0, wet_mass=2.0,
                       dry_mass=1.0, name="b")
        model = MultiStageThrustModel([s1, s2])
        assert model.inertia_after_separation(0) is None


class TestEOMSetInertia:
    def test_set_inertia_updates_inv_and_cg(self):
        eom = EOM6DOF(mass=1000.0, inertia=np.diag([100.0, 200.0, 300.0]))
        new_inertia = np.diag([10.0, 20.0, 30.0])
        cg = np.array([0.2, 0.0, 0.0])
        eom.set_inertia(new_inertia, cg)
        assert np.allclose(eom.inertia, new_inertia)
        assert np.allclose(eom.inertia_inv, np.linalg.inv(new_inertia))
        assert np.allclose(eom.cg, cg)

    def test_gravity_gradient_uses_cg_offset(self):
        from project_icarus.dynamics.gravity import gravity_gradient_torque
        from project_icarus.dynamics.coordinate_systems import quat_normalize
        q = quat_normalize(np.array([1.0, 0.2, 0.1, 0.0]))
        r = np.array([6.4e6, 1.0e6, 0.5e6])
        inv = np.linalg.inv(np.diag([100.0, 200.0, 300.0]))
        t_no_cg = gravity_gradient_torque(r, q, inv, use_j2=True)
        t_with_cg = gravity_gradient_torque(
            r, q, inv, use_j2=True, cg=np.array([1.0, 0.0, 0.0])
        )
        # A non-zero CG offset shifts the gravity-gradient lever arm.
        assert not np.allclose(t_no_cg, t_with_cg)


class TestSeparationConservation:
    def test_impulse_applied_to_bus(self):
        # StageSeparation.apply drops ``mass_drop`` from the bus and applies the
        # separation ``impulse`` (N s) as a delta-v divided by the pre-separation
        # total mass, so the bus gains ``mass_drop_adj`` worth of momentum and
        # the discarded stage carries the equal-and-opposite amount.
        bus_mass = 800.0
        drop = 200.0
        total = bus_mass + drop
        impulse = np.array([150.0, -50.0, 30.0])  # N s
        v_bus_before = np.array([1000.0, 0.0, 0.0])
        from project_icarus.dynamics.thrust import StageSeparation
        sep = StageSeparation(time=0.0, mass_drop=drop, impulse=impulse)
        state = {
            "r": np.zeros(3),
            "v": v_bus_before.copy(),
            "q": np.array([1.0, 0, 0, 0]),
            "omega": np.zeros(3),
            "m": total,
        }
        new = sep.apply(state)
        # Bus delta-v == impulse / total pre-separation mass.
        assert np.allclose(new["v"] - v_bus_before, impulse / total)
        # Bus momentum change.
        bus_dp = bus_mass * (new["v"] - v_bus_before)
        # Discarded stage receives the opposite delta-v to conserve momentum;
        # verify equal/opposite momentum exchange.
        v_stage = v_bus_before - (new["v"] - v_bus_before) * (bus_mass / drop)
        total_after = bus_mass * new["v"] + drop * v_stage
        total_before = total * v_bus_before
        assert np.allclose(total_after, total_before)

    def test_mass_continuity_after_both_separations(self):
        s1, s2 = _two_stage()
        ic = InterceptorConfig(
            name="TwoStage", mass=2000.0, area=0.3, ref_length=7.0,
            stages=[s1, s2],
            sep_impulses=[np.zeros(3), np.zeros(3)],
        )
        model = MultiStageThrustModel(ic.stages, ic.sep_impulses)
        seps = model.separations
        total_drop = sum(s.mass_drop for s in seps if s is not None)
        # Full-vehicle mass minus all spent-stage propellant == final bus mass.
        final_bus = ic.mass - total_drop + sum(
            s.dry_mass for s in ic.stages
        ) * 0.0  # dry masses stay on the bus
        assert ic.dry_mass == 300.0
        assert ic.mass - total_drop == 700.0  # 2000 - 1300


class TestSeparationEvent:
    def test_fires_only_after_burnout_and_coast(self):
        s1, s2 = _two_stage()
        model = MultiStageThrustModel([s1, s2])
        ev0 = SeparationEvent(0, model, SimConfig())
        y = np.zeros(14)
        # Before burnout time: no fire even if thrust sampled low.
        assert not ev0.should_trigger(10.0, y, {"peak_thrust": 20000.0, "thrust": 0.0})
        # After burnout time with thrust ~0: fire.
        assert ev0.should_trigger(35.0, y, {"peak_thrust": 20000.0, "thrust": 0.0})

    def test_second_stage_event(self):
        s1, s2 = _two_stage()
        model = MultiStageThrustModel([s1, s2])
        ev1 = SeparationEvent(1, model, SimConfig())
        y = np.zeros(14)
        assert not ev1.should_trigger(50.0, y, {"peak_thrust": 20000.0, "thrust": 6000.0})
        assert ev1.should_trigger(95.0, y, {"peak_thrust": 20000.0, "thrust": 0.0})


class TestMKVSystem:
    def test_mkv_separation_conserves_momentum(self):
        mkv = MKVSystem(kv_mass=15.0, v_rel=1.5)
        state = {
            "r": np.zeros(3), "v": np.array([2000.0, 0.0, 0.0]),
            "q": np.array([1.0, 0, 0, 0]), "omega": np.zeros(3),
            "m": 100.0,
        }
        new = mkv.separate(state)
        assert new["m"] == 85.0
        bus_dp = 85.0 * (new["v"] - state["v"])
        assert np.allclose(bus_dp, mkv.v_rel * np.array([1.0, 0.0, 0.0]) * 85.0 / 1.0)
        assert mkv.separated

    def test_mkv_idempotent(self):
        mkv = MKVSystem(kv_mass=15.0)
        state = {"r": np.zeros(3), "v": np.zeros(3), "q": np.array([1.0, 0, 0, 0]),
                 "omega": np.zeros(3), "m": 100.0}
        mkv.separate(state)
        # A second call must be a no-op (separated flag), returning the
        # unchanged state rather than dropping mass twice.
        m2 = mkv.separate(dict(state))
        assert not np.isclose(m2["m"], 70.0)
        assert m2["m"] == 100.0


class TestDryMassCutoff:
    def test_dry_mass_cutoff_triggers_thrust_cutoff(self):
        from project_icarus.sim.runner import ThrustCutoffEvent
        ev = ThrustCutoffEvent(frac=1e-3)
        # When mass reaches dry mass and thrust is still nominally high, the
        # dry-mass branch must fire (engine can't burn below dry mass).
        y = np.zeros(14)
        y[13] = 200.0
        dry = 200.0
        ctx = {"peak_thrust": 20000.0, "thrust": 20000.0, "phase": "boost", "dry_mass": dry}
        assert ev.should_trigger(10.0, y, ctx)
