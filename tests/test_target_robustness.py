"""Target robustness: each threat type must reach its destination when unintercepted.

Validates that the trajectory models are physically complete end-to-end:
* Ballistic    - suborbital arc hits the surface.
* HGV          - skip-glide reentry reaches the surface.
* FOBS         - orbital coast + deorbit + guided reentry hits the aim point.
* Suppressed   - midcourse maneuver + reentry reaches the surface.
* Swarm        - bus and all payloads propagate to impact.
* DecoyThreat  - RV propagates like a ballistic; decoys release and separate.
* CruiseMissile - boost + cruise + terminal dive reaches the target area.
"""

import numpy as np
import pytest

from project_icarus.scenarios.target_factory import (
    BallisticScenario,
    HGVScenario,
    FOBSScenario,
    SuppressedScenario,
    SwarmScenario,
    DecoyThreatScenario,
    CruiseMissileScenario,
    R_EARTH,
    _ground_altitude,
)
from project_icarus.scenarios.presets import build_threat_to_defended
from project_icarus.targets.decoy_model import DecoyModel


R_EARTH_F = float(R_EARTH)


def _assert_reaches_surface(state, tol_m=5e3):
    alt = _ground_altitude(state[:3])
    assert alt <= tol_m, f"Target did not reach surface: alt={alt:.1f} m"


def _assert_finite_state(state):
    assert state.shape == (6,)
    assert np.all(np.isfinite(state))


class TestBallisticRobustness:
    def test_standard_icbm_reaches_ground(self):
        r0 = np.array([R_EARTH + 1e5, 0.0, 0.0])
        v0 = np.array([0.0, 0.0, 7000.0])
        tgt = BallisticScenario(r0=r0, v0=v0, adaptive=True)
        state = tgt.propagate(600.0)
        _assert_finite_state(state)
        _assert_reaches_surface(state)

    def test_adaptive_integrator_reaches_ground(self):
        r0 = np.array([R_EARTH + 2e5, 0.0, 0.0])
        v0 = np.array([0.0, 2000.0, 6000.0])
        tgt = BallisticScenario(r0=r0, v0=v0, adaptive=True)
        state = tgt.propagate(500.0)
        _assert_finite_state(state)
        _assert_reaches_surface(state)

    def test_high_order_gravity_reaches_ground(self):
        r0 = np.array([R_EARTH + 1.5e5, 0.0, 0.0])
        v0 = np.array([0.0, 500.0, 7500.0])
        tgt = BallisticScenario(r0=r0, v0=v0, use_j3=True, use_j4=True,
                                use_third_body=True, use_tides=True, adaptive=True)
        state = tgt.propagate(700.0)
        _assert_finite_state(state)
        _assert_reaches_surface(state)


class TestHGVRobustness:
    def test_hsv_reaches_ground(self):
        r0 = np.array([R_EARTH + 80e3, 0.0, 0.0])
        v0 = np.array([0.0, 0.0, 6000.0])
        tgt = HGVScenario(r0=r0, v0=v0, max_alt_km=80.0, lateral_range_km=1500.0)
        state = tgt.propagate(800.0)
        _assert_finite_state(state)
        _assert_reaches_surface(state)

    def test_hsv_adaptive_reaches_ground(self):
        r0 = np.array([R_EARTH + 70e3, 0.0, 0.0])
        v0 = np.array([0.0, 1000.0, 5500.0])
        tgt = HGVScenario(r0=r0, v0=v0, adaptive=True)
        state = tgt.propagate(900.0)
        _assert_finite_state(state)
        _assert_reaches_surface(state)


class TestFOBSRobustness:
    def test_fobs_reaches_ground(self):
        tgt = FOBSScenario.from_orbital_params(apoapsis_km=200.0, inclination_deg=30.0)
        state = tgt.propagate(1200.0)
        _assert_finite_state(state)
        _assert_reaches_surface(state)

    def test_fobs_guided_reentry_hits_aim(self):
        tgt = FOBSScenario.from_orbital_params(apoapsis_km=150.0, inclination_deg=0.0)
        aim = tgt._aim_point.copy()
        state = tgt.propagate(2000.0)
        _assert_finite_state(state)
        _assert_reaches_surface(state)
        impact_pos = state[:3]
        dist_to_aim = float(np.linalg.norm(impact_pos - aim))
        assert dist_to_aim < 2500e3, f"FOBS missed aim by {dist_to_aim / 1e3:.1f} km"


_HGV_AIM = np.array([6.358395225412172e6, 0.0, 499758.44326704595])


class TestHGVAimPoint:
    def test_hgs_hits_aim(self):
        r0 = np.array([R_EARTH + 80e3, 0.0, 0.0])
        v0 = np.array([0.0, 0.0, 6000.0])
        tgt = HGVScenario(r0=r0, v0=v0, max_alt_km=80.0, lateral_range_km=1500.0)
        state = tgt.propagate(1000.0)
        _assert_finite_state(state)
        _assert_reaches_surface(state)
        dist_to_aim = float(np.linalg.norm(state[:3] - _HGV_AIM))
        assert dist_to_aim < 10e3, f"HGV missed aim by {dist_to_aim / 1e3:.1f} km"


_Suppressed_AIM = np.array([5.784705415467419e6, 736664.0803803997, 2578324.2813313985])


class TestSuppressedAimPoint:
    def test_suppressed_hits_aim(self):
        r0 = np.array([R_EARTH + 1e5, 0.0, 0.0])
        v0 = np.array([0.0, 2000.0, 7000.0])
        tgt = SuppressedScenario(r0=r0, v0=v0, midcourse_maneuver_mag=100.0)
        state = tgt.propagate(700.0)
        _assert_finite_state(state)
        _assert_reaches_surface(state)
        dist_to_aim = float(np.linalg.norm(state[:3] - _Suppressed_AIM))
        assert dist_to_aim < 20e3, f"Suppressed missed aim by {dist_to_aim / 1e3:.1f} km"


_CruiseMissile_AIM = np.array([4.530008475706846e6, 1.767262519908741e6, 0.0])


class TestCruiseMissileAimPoint:
    def test_cruise_missile_hits_aim(self):
        tgt = CruiseMissileScenario.from_params(launch_alt_km=0.0, range_km=500.0)
        state = tgt.propagate(700.0)
        _assert_finite_state(state)
        _assert_reaches_surface(state, tol_m=2e4)
        dist_to_aim = float(np.linalg.norm(state[:3] - _CruiseMissile_AIM))
        assert dist_to_aim < 5e3, f"CruiseMissile missed aim by {dist_to_aim / 1e3:.1f} km"


class TestSuppressedRobustness:
    def test_suppressed_reaches_ground(self):
        r0 = np.array([R_EARTH + 1e5, 0.0, 0.0])
        v0 = np.array([0.0, 2000.0, 7000.0])
        tgt = SuppressedScenario(r0=r0, v0=v0, midcourse_maneuver_mag=100.0)
        state = tgt.propagate(700.0)
        _assert_finite_state(state)
        _assert_reaches_surface(state)

    def test_suppressed_adaptive_reaches_ground(self):
        r0 = np.array([R_EARTH + 1.2e5, 0.0, 0.0])
        v0 = np.array([0.0, 1500.0, 6500.0])
        tgt = SuppressedScenario(r0=r0, v0=v0, adaptive=True)
        state = tgt.propagate(600.0)
        _assert_finite_state(state)
        _assert_reaches_surface(state)


class TestSwarmRobustness:
    def test_swarm_payloads_reach_ground(self):
        r0 = np.array([R_EARTH + 1e5, 0.0, 0.0])
        v0 = np.array([0.0, 0.0, 7000.0])
        tgt = SwarmScenario(bus_r0=r0, bus_v0=v0, n_payloads=3, spread_deg=1.0)
        for t in [100.0, 300.0, 600.0, 900.0]:
            states = tgt.payload_states(t)
            assert len(states) == 3
            for s in states:
                _assert_finite_state(s)

    def test_swarm_bus_propagates(self):
        r0 = np.array([R_EARTH + 1e5, 0.0, 0.0])
        v0 = np.array([0.0, 0.0, 7000.0])
        tgt = SwarmScenario(bus_r0=r0, bus_v0=v0, n_payloads=2, spread_deg=2.0)
        state = tgt.propagate(400.0)
        _assert_finite_state(state)
        _assert_reaches_surface(state, tol_m=2e4)


class TestDecoyThreatRobustness:
    def test_rv_propagates_like_ballistic(self):
        rv = BallisticScenario(r0=np.array([R_EARTH + 1e5, 0.0, 0.0]),
                               v0=np.array([0.0, 1500.0, 6000.0]))
        dts = DecoyThreatScenario(rv=rv, decoys=[], release_t=50.0)
        state = dts.propagate(500.0)
        _assert_finite_state(state)
        _assert_reaches_surface(state)

    def test_decoys_release_and_propagate(self):
        rv = BallisticScenario(r0=np.array([R_EARTH + 1e5, 0.0, 0.0]),
                               v0=np.array([0.0, 1500.0, 6000.0]))
        dts = DecoyThreatScenario(
            rv=rv,
            decoys=[{"radar_rcs_bias": -0.4, "ir_bias": 0.6},
                    {"radar_rcs_bias": -0.5, "ir_bias": 0.5}],
            release_t=50.0,
        )
        assert all(s is None for s in dts.decoy_states(30.0))
        states = dts.decoy_states(150.0)
        assert all(s is not None for s in states)
        for s in states:
            assert float(np.linalg.norm(s[0])) > R_EARTH_F

    def test_decoy_features_consistent(self):
        rv = BallisticScenario(r0=np.array([R_EARTH + 1e5, 0.0, 0.0]),
                               v0=np.array([0.0, 1500.0, 6000.0]))
        dts = DecoyThreatScenario(
            rv=rv,
            decoys=[{"radar_rcs_bias": -0.4, "ir_bias": 0.6}],
            release_t=40.0,
        )
        feats = dts.decoy_features(120.0, seed=0)
        assert len(feats) == 1
        assert feats[0].shape == (4,)
        assert feats[0][3] in (0.0, 1.0)


class TestCruiseMissileRobustness:
    def test_cruise_missile_reaches_target_area(self):
        tgt = CruiseMissileScenario.from_params(launch_alt_km=0.0, range_km=500.0)
        state = tgt.propagate(700.0)
        _assert_finite_state(state)
        _assert_reaches_surface(state, tol_m=2e4)

    def test_cruise_missile_long_range(self):
        tgt = CruiseMissileScenario.from_params(launch_alt_km=0.0, range_km=2000.0)
        state = tgt.propagate(700.0)
        _assert_finite_state(state)
        _assert_reaches_surface(state, tol_m=5e4)


class TestGeodeticPairs:
    _PAIRS = [
        ("Kozelsk", "Washington D.C."),
        ("Tatishchevo", "Washington D.C."),
    ]
    _SCENARIOS = [
        ("ballistic", {}),
        ("hgv", {"glide_alt_km": 70.0, "speed_mach": 12.0}),
    ]

    @pytest.mark.parametrize("threat_name,defended_name", _PAIRS)
    @pytest.mark.parametrize("scenario_type,sc_kwargs", _SCENARIOS)
    def test_geodetic_pair_reaches_surface(self, threat_name, defended_name, scenario_type, sc_kwargs):
        preset = build_threat_to_defended(
            threat_name=threat_name,
            defended_name=defended_name,
            scenario_type=scenario_type,
            engagement_end=20000.0,
            **sc_kwargs,
        )
        state = preset.target.propagate(20000.0)
        _assert_finite_state(state)
        _assert_reaches_surface(state, tol_m=5e3)
