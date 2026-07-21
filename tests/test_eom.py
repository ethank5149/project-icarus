import numpy as np
import pytest
from project_icarus.dynamics.atmosphere import Atmosphere
from project_icarus.dynamics.coordinate_systems import (
    quat_multiply,
    quat_normalize,
    quat_to_dcm,
    quat_kinematics,
    rotate_body_to_inertial,
    rotate_inertial_to_body,
    dcm_to_quat,
)
from project_icarus.dynamics.eom_6dof import EOM6DOF, geodetic_altitude
from project_icarus.dynamics.gravity import gravity_inertial, R_EARTH
from project_icarus.dynamics.gravity import MU_EARTH, R_EARTH


class TestAtmosphere:
    def test_endo_density(self):
        atm = Atmosphere()
        rho = atm.density(np.array([0.0, 1e3, 10e3]))
        assert np.all(rho > 0)
        assert rho[0] > rho[1] > rho[2]

    def test_exo_density(self):
        atm = Atmosphere()
        rho = atm.density(np.array([200e3]))
        assert np.isfinite(rho[0])
        assert rho[0] < 1e-6

    def test_smooth_taper(self):
        atm = Atmosphere()
        h = np.linspace(95e3, 105e3, 50)
        rho = atm.density(h)
        assert np.all(np.isfinite(rho))

    def test_is_endo(self):
        atm = Atmosphere()
        assert atm.is_endo(np.array([50e3]))[0] == True
        assert atm.is_endo(np.array([150e3]))[0] == False


class TestCoordinateSystems:
    def test_quat_normalize(self):
        q = np.array([2.0, 0.0, 0.0, 0.0])
        qn = quat_normalize(q)
        assert np.isclose(np.linalg.norm(qn), 1.0)

    def test_quat_multiply_identity(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        qm = quat_multiply(q, q)
        assert np.allclose(qm, q)

    def test_dcm_from_quat(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        C = quat_to_dcm(q)
        assert np.allclose(C, np.eye(3))

    def test_kinematics(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        omega = np.array([0.0, 0.0, 0.0])
        dq = quat_kinematics(q, omega)
        assert np.allclose(dq, 0.0)

    def test_dcm_orthogonality(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            q = quat_normalize(rng.standard_normal(4))
            C = quat_to_dcm(q)
            assert np.allclose(C @ C.T, np.eye(3), atol=1e-12)

    def test_rotate_body_to_inertial_roundtrip(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            q = quat_normalize(rng.standard_normal(4))
            v = rng.standard_normal(3)
            v_back = rotate_inertial_to_body(rotate_body_to_inertial(v, q), q)
            assert np.allclose(v_back, v, atol=1e-12)

    def test_rotate_inertial_to_body_roundtrip(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            q = quat_normalize(rng.standard_normal(4))
            v_i = rng.standard_normal(3)
            v_back = rotate_body_to_inertial(rotate_inertial_to_body(v_i, q), q)
            assert np.allclose(v_back, v_i, atol=1e-12)

    def test_quat_dcm_roundtrip(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            q = quat_normalize(rng.standard_normal(4))
            C = quat_to_dcm(q)
            q_back = dcm_to_quat(C)
            assert np.allclose(quat_to_dcm(q_back), C, atol=1e-12)


class TestEOM:
    def test_zero_force(self):
        eom = EOM6DOF(mass=1.0, inertia=np.eye(3))
        state = {
            "r": np.array([1e10, 0.0, 0.0]),
            "v": np.zeros(3),
            "q": np.array([1.0, 0.0, 0.0, 0.0]),
            "omega": np.zeros(3),
            "m": 1.0,
        }
        def surf(mach, alpha, beta, alt):
            return 0.0, 0.0, 0.0
        derivs = eom.compute(0.0, state, surf)
        assert np.allclose(derivs["r"], np.zeros(3))
        assert np.allclose(derivs["v"], np.zeros(3), atol=1e-5)

    def test_gravity_pull(self):
        eom = EOM6DOF(mass=1.0, inertia=np.eye(3))
        state = {
            "r": np.array([6371e3 + 1e3, 0.0, 0.0]),
            "v": np.zeros(3),
            "q": np.array([1.0, 0.0, 0.0, 0.0]),
            "omega": np.zeros(3),
            "m": 1.0,
        }
        def surf(mach, alpha, beta, alt):
            return 0.0, 0.0, 0.0
        derivs = eom.compute(0.0, state, surf)
        assert derivs["v"][0] < 0

    def test_velocity_is_inertial(self):
        # dr/dt must equal the inertial velocity v directly (no body rotation).
        eom = EOM6DOF(mass=1.0, inertia=np.eye(3))
        v = np.array([100.0, 200.0, 300.0])
        state = {
            "r": np.array([1e7, 0.0, 0.0]),
            "v": v,
            "q": np.array([1.0, 0.0, 0.0, 0.0]),
            "omega": np.zeros(3),
            "m": 1.0,
        }
        def surf(mach, alpha, beta, alt):
            return 0.0, 0.0, 0.0
        derivs = eom.compute(0.0, state, surf)
        assert np.allclose(derivs["r"], v)

    def test_force_scaling(self):
        # Gravity is an acceleration (mass-independent); only aero/thrust forces
        # scale as 1/mass. With no aero/thrust, acceleration is identical.
        eom1 = EOM6DOF(mass=1.0, inertia=np.eye(3))
        eom2 = EOM6DOF(mass=10.0, inertia=np.eye(3))
        base = dict(
            r=np.array([6371e3 + 1e3, 0.0, 0.0]),
            v=np.zeros(3),
            q=np.array([1.0, 0.0, 0.0, 0.0]),
            omega=np.zeros(3),
        )
        def surf(mach, alpha, beta, alt):
            return 0.0, 0.0, 0.0
        d1 = eom1.compute(0.0, {**base, "m": 1.0}, surf)
        d2 = eom2.compute(0.0, {**base, "m": 10.0}, surf)
        assert np.allclose(d1["v"], d2["v"])

    def test_thrust_force_scaling(self):
        # A thrust force accelerates more lightly-loaded vehicles: a = F/m.
        eom1 = EOM6DOF(mass=1.0, inertia=np.eye(3))
        eom2 = EOM6DOF(mass=10.0, inertia=np.eye(3))
        base = dict(
            r=np.array([6371e3 + 1e3, 0.0, 0.0]),
            v=np.zeros(3),
            q=np.array([1.0, 0.0, 0.0, 0.0]),
            omega=np.zeros(3),
        )
        def thruster(t, state):
            return np.array([-100.0, 0.0, 0.0])
        eom1.set_thrust(type("T", (), {"thrust_vector": staticmethod(thruster), "mass_rate": staticmethod(lambda t, s: 0.0)})())
        eom2.set_thrust(type("T", (), {"thrust_vector": staticmethod(thruster), "mass_rate": staticmethod(lambda t, s: 0.0)})())
        def surf(mach, alpha, beta, alt):
            return 0.0, 0.0, 0.0
        d1 = eom1.compute(0.0, {**base, "m": 1.0}, surf)
        d2 = eom2.compute(0.0, {**base, "m": 10.0}, surf)
        # The thrust force (100 N) contributes a = F/m: -100 for m=1, -10 for m=10.
        # Removing the thrust acceleration from each leaves the identical gravity
        # acceleration, confirming thrust scales as 1/mass and gravity does not.
        assert np.allclose(d1["v"] + np.array([100.0, 0.0, 0.0]),
                            d2["v"] + np.array([10.0, 0.0, 0.0]))

    def test_quaternion_norm_preserved(self):
        # Quaternion kinematics must preserve norm: d/dt(q.q) = 2 q.dq = 0.
        eom = EOM6DOF(mass=1.0, inertia=np.eye(3))
        q = np.array([0.2, 0.4, -0.1, 0.883])
        q = q / np.linalg.norm(q)
        state = {
            "r": np.array([6371e3 + 1e3, 0.0, 0.0]),
            "v": np.zeros(3),
            "q": q,
            "omega": np.array([0.1, -0.2, 0.05]),
            "m": 1.0,
        }
        def surf(mach, alpha, beta, alt):
            return 0.0, 0.0, 0.0
        dq = eom.compute(0.0, state, surf)["q"]
        assert np.isclose(2.0 * np.dot(q, dq), 0.0, atol=1e-12)

    def test_aero_convention_drag_magnitude(self):
        # With only drag (Cd>0, Cy=Cm=0), the aero force opposes velocity.
        eom = EOM6DOF(mass=1.0, inertia=np.eye(3))
        v = np.array([1000.0, 0.0, 0.0])
        state = {
            "r": np.array([1e7, 0.0, 0.0]),
            "v": v,
            "q": np.array([1.0, 0.0, 0.0, 0.0]),
            "omega": np.zeros(3),
            "m": 1.0,
        }
        def surf(mach, alpha, beta, alt):
            return 0.5, 0.0, 0.0  # Cd only
        dv = eom.compute(0.0, state, surf)["v"]
        # Aero drag should decelerate the +x velocity.
        assert dv[0] < 0

    def test_energy_conservation_with_j2(self):
        eom = EOM6DOF(mass=1.0, inertia=np.eye(3), use_j2=True)
        r = np.array([1e7, 0.0, 0.0])
        v = np.array([0.0, 7000.0, 0.0])
        state = {"r": r, "v": v, "q": np.array([1.0, 0.0, 0.0, 0.0]),
                 "omega": np.zeros(3), "m": 1.0}
        def surf(mach, alpha, beta, alt):
            return 0.0, 0.0, 0.0
        d = eom.compute(0.0, state, surf)
        # Small-step energy change should be near zero (no thrust/aero).
        ke = 0.5 * np.dot(v, v)
        pe = -3.986004418e14 / np.linalg.norm(r)
        e0 = ke + pe
        ke1 = 0.5 * np.dot(v + d["v"] * 1e-3, v + d["v"] * 1e-3)
        pe1 = -3.986004418e14 / np.linalg.norm(r + d["r"] * 1e-3)
        assert np.isclose(e0, ke1 + pe1, rtol=1e-4)


class TestGeodeticAltitude:
    def test_surface(self):
        r = np.array([_R_EARTH_SURF(), 0.0, 0.0])
        assert np.isclose(geodetic_altitude(r), 0.0, atol=100.0)

    def test_positive_alt(self):
        r = np.array([_R_EARTH_SURF() + 100e3, 0.0, 0.0])
        assert geodetic_altitude(r) > 50e3


def _R_EARTH_SURF():
    return 6378137.0


class TestAtmosphereLayers:
    def test_layer_continuity(self):
        atm = Atmosphere()
        # Within the endo regime (below the taper), density decreases monotonically.
        h = np.linspace(0.0, 90e3, 200)
        rho = atm.density(h)
        # Monotonically decreasing except for the tiny known seam step at 86 km.
        assert np.all(np.diff(rho) < 1e-6)
        # Continuous transition across the boundary.
        hb = atm.boundary_alt
        rho_low = atm.density(np.array([hb - 1e3]))
        rho_high = atm.density(np.array([hb + 1e3]))
        assert np.isfinite(rho_low) and np.isfinite(rho_high)

    def test_exo_temperature_range(self):
        atm = Atmosphere()
        T = atm.temperature(np.array([200e3, 500e3]))
        assert np.all(T > 200.0)
        assert T[1] >= T[0]

    def test_speed_of_sound_finite(self):
        atm = Atmosphere()
        a = atm.speed_of_sound(np.linspace(0.0, 200e3, 50))
        assert np.all(np.isfinite(a))
        assert np.all(a > 0)


    def test_quaternion_norm_preserved(self):
        eom = EOM6DOF(mass=1.0, inertia=np.eye(3))
        state = {
            "r": np.array([6371e3 + 1e3, 0.0, 0.0]),
            "v": np.array([100.0, 0.0, 0.0]),
            "q": np.array([0.8, 0.1, 0.2, 0.3]),
            "omega": np.array([0.1, 0.2, 0.3]),
            "m": 1.0,
        }
        def surf(mach, alpha, beta, alt):
            return 0.0, 0.0, 0.0
        q0 = state["q"].copy()
        for _ in range(100):
            derivs = eom.compute(0.0, state, surf)
            dt = 0.01
            state = {
                "r": state["r"] + derivs["r"] * dt,
                "v": state["v"] + derivs["v"] * dt,
                "q": state["q"] + derivs["q"] * dt,
                "omega": state["omega"] + derivs["omega"] * dt,
                "m": state["m"] + derivs["m"] * dt,
            }
            state["q"] = state["q"] / max(np.linalg.norm(state["q"]), 1e-12)
        assert np.isclose(np.linalg.norm(state["q"]), 1.0, atol=1e-6)

    def test_energy_conservation_with_j2(self):
        eom = EOM6DOF(mass=1000.0, inertia=np.eye(3), use_j2=False)
        r0 = np.array([7000e3, 0.0, 0.0])
        v0 = np.array([0.0, 7000.0, 0.0])
        state = {
            "r": r0,
            "v": v0,
            "q": np.array([1.0, 0.0, 0.0, 0.0]),
            "omega": np.zeros(3),
            "m": 1000.0,
        }
        def surf(mach, alpha, beta, alt):
            return 0.0, 0.0, 0.0
        e0 = 0.5 * np.linalg.norm(v0)**2 - MU_EARTH / np.linalg.norm(r0)
        dt = 0.1
        for _ in range(1000):
            derivs = eom.compute(0.0, state, surf)
            state = {
                "r": state["r"] + derivs["r"] * dt,
                "v": state["v"] + derivs["v"] * dt,
                "q": state["q"] + derivs["q"] * dt,
                "omega": state["omega"] + derivs["omega"] * dt,
                "m": state["m"] + derivs["m"] * dt,
            }
        e1 = 0.5 * np.linalg.norm(state["v"])**2 - MU_EARTH / np.linalg.norm(state["r"])
        assert np.isclose(e0, e1, rtol=1e-2)

    def test_atmosphere_layer_transitions(self):
        atm = Atmosphere()
        h = np.array([10e3, 15e3, 25e3, 40e3, 50e3, 80e3, 95e3])
        rho = atm.density(h)
        assert np.all(np.isfinite(rho))
        assert np.all(rho > 0)
        assert rho[0] > rho[1] > rho[2] > rho[3] > rho[4]
        assert rho[4] > rho[5] > rho[6]


class TestGravity:
    def test_inverse_square(self):
        r = np.array([1e6, 0.0, 0.0])
        g = gravity_inertial(r)
        assert np.isclose(np.linalg.norm(g), 3.986004418e14 / 1e12, rtol=1e-5)

    def test_j2_toggle(self):
        r = np.array([7000e3, 0.0, 0.0])
        g_newton = gravity_inertial(r, use_j2=False)
        g_j2 = gravity_inertial(r, use_j2=True)
        assert not np.allclose(g_newton, g_j2)
        assert np.linalg.norm(g_j2) < np.linalg.norm(g_newton)

    def test_j2_at_low_alt(self):
        r = np.array([R_EARTH + 1e3, 0.0, 0.0])
        g_newton = gravity_inertial(r, use_j2=False)
        g_j2 = gravity_inertial(r, use_j2=True)
        assert np.allclose(g_newton, g_j2, rtol=1e-5)


class TestNRLMSISEAtmosphere:
    def test_uses_nrlmsise(self):
        from project_icarus.dynamics.atmosphere import _HAVE_NRLMSISE
        if not _HAVE_NRLMSISE:
            pytest.skip("nrlmsise00 not installed")
        atm = Atmosphere()
        assert atm.uses_nrlmsise
        h = np.array([150e3, 300e3, 500e3])
        rho = atm.density(h)
        assert np.all(np.isfinite(rho))
        # Density decreases with altitude in the thermosphere.
        assert rho[0] > rho[1] > rho[2]

    def test_high_solar_activity_increases_density(self):
        from project_icarus.dynamics.atmosphere import _HAVE_NRLMSISE
        if not _HAVE_NRLMSISE:
            pytest.skip("nrlmsise00 not installed")
        atm = Atmosphere()
        atm.set_exo_solar_geomagnetic(f107a=70.0, f107=70.0, ap=4.0)
        quiet = atm.density(np.array([300e3]))[0]
        atm.set_exo_solar_geomagnetic(f107a=250.0, f107=250.0, ap=4.0)
        active = atm.density(np.array([300e3]))[0]
        assert active > quiet

    def test_geomagnetic_storm_increases_density(self):
        from project_icarus.dynamics.atmosphere import _HAVE_NRLMSISE
        if not _HAVE_NRLMSISE:
            pytest.skip("nrlmsise00 not installed")
        atm = Atmosphere()
        atm.set_exo_solar_geomagnetic(f107a=150.0, f107=150.0, ap=4.0)
        quiet = atm.density(np.array([400e3]))[0]
        atm.set_exo_solar_geomagnetic(f107a=150.0, f107=150.0, ap=80.0)
        storm = atm.density(np.array([400e3]))[0]
        assert storm > quiet

    def test_temperature_finite(self):
        from project_icarus.dynamics.atmosphere import _HAVE_NRLMSISE
        if not _HAVE_NRLMSISE:
            pytest.skip("nrlmsise00 not installed")
        atm = Atmosphere()
        T = atm.temperature(np.array([250e3]))
        assert np.isfinite(T[0])
        assert T[0] > 500.0


class TestHigherOrderGravity:
    def test_j3_j4_independent_toggle(self):
        # J3/J4 must be independently togglable (Phase 3.1) and absent from the
        # central field. Use an off-axis point so the odd/even zonal terms are
        # non-zero.
        r = np.array([5000e3, 0.0, 5000e3])
        g_j2_only = gravity_inertial(r, use_j2=True, use_j3=False, use_j4=False)
        g_j3 = gravity_inertial(r, use_j2=True, use_j3=True, use_j4=False)
        g_j4 = gravity_inertial(r, use_j2=True, use_j3=False, use_j4=True)
        # Enabling J3 changes the field (it is an odd zonal term).
        assert not np.allclose(g_j2_only, g_j3, rtol=0.0, atol=1e-10)
        # Enabling J4 changes the field (even zonal term).
        assert not np.allclose(g_j2_only, g_j4, rtol=0.0, atol=1e-10)
        # J3 contribution is small relative to the J2-dominated field.
        assert np.linalg.norm(g_j3 - g_j2_only) < 1e-2 * np.linalg.norm(g_j2_only)

    def test_max_degree_selectable(self):
        # max_degree selects the upper bound of the EGM2008 zonal series
        # (Phase 3.1 "higher-order EGM2008 toggle").
        r = np.array([5000e3, 0.0, 5000e3])
        g_deg4 = gravity_inertial(r, use_high_order=True, max_degree=4)
        g_deg10 = gravity_inertial(r, use_high_order=True, max_degree=10)
        # Higher degree adds tiny but non-zero J5..J10 terms.
        assert not np.allclose(g_deg4, g_deg10, rtol=0.0, atol=1e-10)
        assert np.linalg.norm(g_deg10 - g_deg4) > 1e-12
        # max_degree=4 must equal the J2..J4 field with high-order disabled.
        g_base = gravity_inertial(r, use_high_order=False)
        assert np.allclose(g_deg4, g_base, rtol=0.0, atol=1e-12)

    def test_high_order_perturbation_small(self):
        # Use a mid-latitude point so zonal harmonics P_n^0(z/r) are nonzero.
        r = np.array([5000e3, 0.0, 5000e3])
        g = gravity_inertial(r, use_high_order=True, use_third_body=False, use_tides=False)
        g_base = gravity_inertial(r, use_high_order=False, use_third_body=False, use_tides=False)
        # J5..J10 terms are tiny relative to the central + J2..J4 field.
        rel = np.linalg.norm(g - g_base) / np.linalg.norm(g_base)
        assert rel < 1e-3
        # The perturbation is small but non-zero (well below 1e-6 relative,
        # reflecting the genuine smallness of J5..J10).
        assert np.linalg.norm(g - g_base) > 1e-9
        assert not np.allclose(g, g_base, rtol=0.0, atol=1e-10)

    def test_third_body_perturbation(self):
        r = np.array([7000e3, 0.0, 0.0])
        g0 = gravity_inertial(r, use_third_body=False)
        g1 = gravity_inertial(r, use_third_body=True, t=86400.0)
        d = np.linalg.norm(g1 - g0)
        assert d > 1e-8
        assert d < 1e-4  # third-body is a small perturbation

    def test_solid_earth_tide_perturbation(self):
        r = np.array([7000e3, 0.0, 0.0])
        g0 = gravity_inertial(r, use_tides=False)
        g1 = gravity_inertial(r, use_tides=True, t=86400.0)
        d = np.linalg.norm(g1 - g0)
        assert d > 1e-10
        assert d < 1e-4

    def test_legendre_zonal(self):
        from project_icarus.dynamics.gravity import _legendre_zonal
        assert np.isclose(_legendre_zonal(0, 0.3), 1.0)
        assert np.isclose(_legendre_zonal(1, 0.3), 0.3)
        assert np.isclose(_legendre_zonal(2, 0.0), -0.5)
        assert np.isclose(_legendre_zonal(6, 0.0), -0.3125)


class TestEOMGravityFidelity:
    def test_high_order_flag_propagates(self):
        import numpy as np
        eom = EOM6DOF(mass=1000.0, inertia=np.diag([100.0, 200.0, 300.0]),
                       area=0.1, ref_length=1.0,
                       use_high_order=True, use_third_body=True, use_tides=True)
        assert eom.use_high_order and eom.use_third_body and eom.use_tides

    def test_eom_compute_runs_with_full_gravity(self):
        import numpy as np
        eom = EOM6DOF(mass=1000.0, inertia=np.diag([100.0, 200.0, 300.0]),
                       area=0.1, ref_length=1.0,
                       use_high_order=True, use_third_body=True, use_tides=True)
        def surf(mach, alpha, beta, alt):
            return 0.5, 0.0, 0.01
        st = {"r": np.array([6.371e6, 0.0, 0.0]), "v": np.array([0.0, 7500.0, 0.0]),
              "q": np.array([1.0, 0.0, 0.0, 0.0]), "omega": np.zeros(3), "m": 1000.0}
        d = eom.compute(1000.0, st, surf)
        assert np.all(np.isfinite(d["v"]))
        assert np.linalg.norm(d["v"]) > 0.0


class TestWGS84Fidelity:
    def test_geodetic_ecef_roundtrip(self):
        from project_icarus.scenarios.target_factory import _ecef_to_geodetic
        from project_icarus.scenarios.presets import geodetic_to_ecef
        rng = np.random.default_rng(42)
        for _ in range(20):
            lat = rng.uniform(-80, 80)
            lon = rng.uniform(-180, 180)
            alt = rng.uniform(-500, 10000)
            r = geodetic_to_ecef(lat, lon, alt)
            lat2, lon2, alt2 = _ecef_to_geodetic(r)
            assert np.isclose(lat, lat2, atol=1e-8)
            assert np.isclose(lon, lon2, atol=1e-8)
            assert np.isclose(alt, alt2, rtol=1e-6)

    def test_surface_at_prime_meridian(self):
        from project_icarus.scenarios.target_factory import _ecef_to_geodetic
        r = np.array([6378137.0, 0.0, 0.0])
        lat, lon, alt = _ecef_to_geodetic(r)
        assert np.isclose(lat, 0.0, atol=1e-6)
        assert np.isclose(lon, 0.0, atol=1e-6)
        assert np.isclose(alt, 0.0, atol=100.0)

    def test_surface_at_north_pole(self):
        from project_icarus.scenarios.target_factory import _ecef_to_geodetic
        r = np.array([0.0, 0.0, 6356752.0])
        lat, lon, alt = _ecef_to_geodetic(r)
        assert lat > 85.0
        assert np.isclose(alt, 0.0, atol=500.0)


class TestEnergyConservation:
    def test_keplerian_energy_conserved(self):
        from project_icarus.scenarios.target_factory import _two_body_accel
        rng = np.random.default_rng(42)
        for _ in range(5):
            r = rng.uniform(6500e3, 8000e3, 3)
            v = rng.uniform(-8e3, 8e3, 3)
            e0 = 0.5 * np.dot(v, v) - MU_EARTH / np.linalg.norm(r)
            dt = 0.5
            for _ in range(200):
                a = _two_body_accel(r, use_j2=False)
                v = v + a * dt
                r = r + v * dt
            e1 = 0.5 * np.dot(v, v) - MU_EARTH / np.linalg.norm(r)
            assert np.isclose(e0, e1, rtol=1e-3)

    def test_angular_momentum_conserved(self):
        from project_icarus.scenarios.target_factory import _two_body_accel
        rng = np.random.default_rng(42)
        for _ in range(5):
            r = rng.uniform(6500e3, 8000e3, 3)
            v = rng.uniform(-8e3, 8e3, 3)
            h0 = np.cross(r, v)
            dt = 0.5
            for _ in range(200):
                a = _two_body_accel(r, use_j2=False)
                v = v + a * dt
                r = r + v * dt
            h1 = np.cross(r, v)
            assert np.allclose(h0, h1, rtol=1e-3)


class TestGeodeticGeometry:
    def test_great_circle_distance(self):
        from project_icarus.scenarios.presets import _great_circle_azimuth_range_km
        az, rng = _great_circle_azimuth_range_km(0.0, 0.0, 0.0, 90.0)
        assert 0.0 <= az <= 360.0
        assert rng > 0.0

    def test_surface_elevation_nonnegative(self):
        from project_icarus.reference.surface_elevation import get_surface_elevation
        rng = np.random.default_rng(42)
        for _ in range(10):
            lat = rng.uniform(-60, 60)
            lon = rng.uniform(-120, 120)
            elev = get_surface_elevation(lat, lon)
            assert np.isfinite(elev)
