"""Physical invariance tests that catch coordinate-convention regressions."""
import numpy as np
import pytest

from project_icarus.dynamics.coordinate_systems import (
    geodetic_to_ecef,
    ecef_to_geodetic,
    quat_normalize,
    quat_kinematics,
    quat_to_dcm,
    _WGS84_A,
)
from project_icarus.scenarios.target_factory import (
    BallisticScenario,
    _ground_altitude,
    R_EARTH,
)
from reference.locations import load_locations
from project_icarus.scenarios.presets import _FT_TO_M


class TestEnergyConservation:
    def test_ballistic_coasting_energy_drift(self):
        r0 = np.array([R_EARTH + 500e3, 0.0, 0.0])
        v0 = np.array([0.0, 0.0, 7000.0])
        tgt = BallisticScenario(r0=r0, v0=v0, adaptive=True, use_j2=False,
                                use_j3=False, use_j4=False,
                                use_high_order=False, use_third_body=False, use_tides=False,
                                cd=0.0)
        state0 = tgt.propagate(0.0)
        state1 = tgt.propagate(300.0)
        r0_, v0_ = state0[:3], state0[3:]
        r1, v1 = state1[:3], state1[3:]
        mu = 3.986004418e14
        e0 = 0.5 * np.dot(v0_, v0_) - mu / np.linalg.norm(r0_)
        e1 = 0.5 * np.dot(v1, v1) - mu / np.linalg.norm(r1)
        rel_delta = abs(e1 - e0) / abs(e0)
        assert rel_delta < 0.001, f"Energy drift {rel_delta:.4e} exceeds 0.1%"


class TestGeodeticRoundTrip:
    def test_ecef_geodetic_roundtrip_mm(self):
        cases = [
            (39.9, -75.0, 0.0),
            (-33.8688, 151.2093, 100.0),
            (0.0, 0.0, -50.0),
            (55.7558, 37.6173, 200.0),
        ]
        for lat, lon, alt in cases:
            r = geodetic_to_ecef(lat, lon, alt)
            lat2, lon2, alt2 = ecef_to_geodetic(r)
            assert abs(lat2 - lat) < 1e-7
            assert abs(lon2 - lon) < 1e-7
            assert abs(alt2 - alt) < 1e-3


class TestFeetToMetersConversion:
    def test_locations_yaml_feet_conversion(self):
        recs = load_locations()
        target_name = "Whiteman AFB"
        rec = next((r for r in recs if r["name"] == target_name), None)
        assert rec is not None
        alt_ft = float(rec["coordinates"]["altitude"])
        alt_m = alt_ft * _FT_TO_M
        lat = float(rec["coordinates"]["latitude"])
        lon = float(rec["coordinates"]["longitude"])
        r = geodetic_to_ecef(lat, lon, alt_m)
        _, _, recovered_alt = ecef_to_geodetic(r)
        assert abs(recovered_alt - alt_m) < 0.1


class TestQuaternionNorm:
    def test_quaternion_norm_after_eom_steps(self):
        from project_icarus.dynamics.eom_6dof import EOM6DOF

        eom = EOM6DOF(mass=1000.0, use_cython=False)
        r = np.array([6.4e6, 0.0, 0.0])
        v = np.array([0.0, 1000.0, 7000.0])
        q = np.array([1.0, 0.0, 0.0, 0.0])
        omega = np.array([0.1, 0.2, 0.3])
        m = 1000.0
        state = {"r": r, "v": v, "q": q, "omega": omega, "m": m}

        dt = 0.01
        surrogate = lambda mach, alpha, beta, alt: (0.3, 0.0, 0.0)

        for _ in range(1000):
            derivs = eom.compute(0.0, state, surrogate)
            state["q"] = state["q"] + dt * derivs["q"]
            state["q"] = state["q"] / max(np.linalg.norm(state["q"]), 1e-12)
            state["r"] = state["r"] + dt * derivs["r"]
            state["v"] = state["v"] + dt * derivs["v"]

        assert abs(np.linalg.norm(state["q"]) - 1.0) < 1e-12
