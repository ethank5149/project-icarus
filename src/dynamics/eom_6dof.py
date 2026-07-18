import numpy as np
from .coordinate_systems import quat_normalize, quat_kinematics, quat_to_dcm, rotate_body_to_inertial, rotate_inertial_to_body
from .gravity import gravity_inertial
from .atmosphere import Atmosphere


class EOM6DOF:
    """
    6-DOF Newton-Euler equations of motion in body frame.
    State: r(3), v(3), q(4), omega(3), m(1)
    """

    def __init__(
        self,
        mass=1000.0,
        inertia=np.diag([100.0, 200.0, 300.0]),
        area=0.1,
        ref_length=1.0,
        boundary_alt=100e3,
        taper_width=5e3,
        atmosphere=None,
    ):
        self.mass = mass
        self.inertia = np.array(inertia, dtype=float)
        self.inertia_inv = np.linalg.inv(self.inertia)
        self.area = area
        self.ref_length = ref_length
        self.boundary_alt = boundary_alt
        self.taper_width = taper_width
        self.atmosphere = atmosphere if atmosphere is not None else Atmosphere(boundary_alt, taper_width)
        self.mkv = None
        self.thrust_model = None
        self.separations = []

    def set_mkv(self, mkv):
        self.mkv = mkv

    def set_thrust(self, thrust_model):
        self.thrust_model = thrust_model

    def add_separation(self, sep):
        self.separations.append(sep)

    def aerodynamic_forces(self, q, v_body, cd, cl, cm):
        q_dyn = 0.5 * v_body @ v_body
        fx = -q_dyn * self.area * cd
        fz = -q_dyn * self.area * cl
        fy = q_dyn * self.area * cm
        return np.array([fx, fy, fz])

    def aerodynamic_moments(self, q, v_body, cd, cl, cm, omega):
        q_dyn = 0.5 * v_body @ v_body
        damping = -0.1 * np.array([omega[0], omega[1], omega[2]])
        moment = np.array([cm * q_dyn * self.area * self.ref_length, 0.0, 0.0])
        return moment + damping

    def compute(self, t, state, surrogate_func):
        r = np.asarray(state["r"], dtype=float)
        v = np.asarray(state["v"], dtype=float)
        q = np.asarray(state["q"], dtype=float)
        omega = np.asarray(state["omega"], dtype=float)
        m = float(state["m"])
        q = quat_normalize(q)

        alt = np.linalg.norm(r) - 6371e3
        v_inertial = rotate_body_to_inertial(v, q)
        speed = np.linalg.norm(v_inertial)
        mach = speed / max(self.atmosphere.speed_of_sound(np.array([alt]))[0], 1e-6)

        alpha = np.degrees(np.arctan2(v[2], v[0]))
        beta = np.degrees(np.arcsin(np.clip(v[1] / max(speed, 1e-6), -1.0, 1.0)))

        cd, cl, cm = surrogate_func(mach, alpha, beta, alt)
        endo = self.atmosphere.is_endo(np.array([alt]))[0]
        if not endo:
            cd *= 0.0

        f_aero_body = self.aerodynamic_forces(q, v, cd, cl, cm)
        m_aero_body = self.aerodynamic_moments(q, v, cd, cl, cm, omega)

        f_thrust_body = np.zeros(3)
        m_thrust_body = np.zeros(3)
        mass_dot = 0.0
        if self.thrust_model is not None:
            T = self.thrust_model.thrust(t, state)
            f_thrust_body = np.array([T, 0.0, 0.0])
            mass_dot = self.thrust_model.mass_rate(t, state)

        g_inertial = gravity_inertial(r)
        f_gravity_body = rotate_inertial_to_body(g_inertial, q)
        f_total_body = f_aero_body + f_thrust_body + f_gravity_body

        m_gravity_body = np.zeros(3)
        m_total_body = m_aero_body + m_thrust_body + m_gravity_body

        dr_dt = v
        dv_dt = f_total_body / max(m, 1e-6)
        dq_dt = quat_kinematics(q, omega)
        domega_dt = self.inertia_inv @ (m_total_body - np.cross(omega, self.inertia @ omega))
        dm_dt = mass_dot

        return {
            "r": dr_dt,
            "v": dv_dt,
            "q": dq_dt,
            "omega": domega_dt,
            "m": dm_dt,
        }
