import numpy as np
import pytest
from src.dynamics.atmosphere import Atmosphere
from src.dynamics.coordinate_systems import quat_multiply, quat_normalize, quat_to_dcm, quat_kinematics
from src.dynamics.eom_6dof import EOM6DOF
from src.dynamics.gravity import gravity_inertial


class TestAtmosphere:
    def test_endo_density(self):
        atm = Atmosphere()
        rho = atm.density(np.array([0.0, 1e3, 10e3]))
        assert np.all(rho > 0)
        assert rho[0] > rho[1] > rho[2]

    def test_exo_density(self):
        atm = Atmosphere()
        rho = atm.density(np.array([200e3]))
        assert rho[0] == 0.0

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


class TestGravity:
    def test_inverse_square(self):
        r = np.array([1e6, 0.0, 0.0])
        g = gravity_inertial(r)
        assert np.isclose(np.linalg.norm(g), 3.986004418e14 / 1e12, rtol=1e-5)
