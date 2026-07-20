import numpy as np
from datetime import datetime
from typing import Optional

from .coordinate_systems import (
    quat_normalize,
    quat_kinematics,
    quat_to_dcm,
    rotate_body_to_inertial,
    rotate_inertial_to_body,
    _geodetic_altitude,
    ecef_to_geodetic,
)
from .gravity import gravity_inertial, gravity_gradient_torque
from .atmosphere import Atmosphere


geodetic_altitude = _geodetic_altitude


class EOM6DOF:
    """
    6-DOF Newton-Euler equations of motion in body frame.
    State: r(3), v(3), q(4), omega(3), m(1)
    Convention: x-forward, y-right, z-down
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
        use_j2=True,
        use_high_order=True,
        use_j3=True,
        use_j4=True,
        max_degree=10,
        use_third_body=False,
        use_tides=False,
        solar_time=None,
        f107a=150.0,
        f107=150.0,
        ap=4.0,
    ):
        self.mass = mass
        self.inertia = np.array(inertia, dtype=float)
        self.inertia_inv = np.linalg.inv(self.inertia)
        self.cg = np.zeros(3)
        self.area = area
        self.ref_length = ref_length
        self.boundary_alt = boundary_alt
        self.taper_width = taper_width
        self.atmosphere = atmosphere if atmosphere is not None else Atmosphere(boundary_alt, taper_width)
        self.use_j2 = use_j2
        self.use_high_order = use_high_order
        self.use_j3 = use_j3
        self.use_j4 = use_j4
        self.max_degree = max_degree
        self.use_third_body = use_third_body
        self.use_tides = use_tides
        self.solar_time = solar_time if solar_time is not None else datetime(2000, 3, 21, 12, 0, 0)
        self.f107a = f107a
        self.f107 = f107
        self.ap = ap
        if atmosphere is not None and atmosphere.uses_nrlmsise:
            atmosphere.set_exo_solar_geomagnetic(
                f107a=self.f107a, f107=self.f107, ap=self.ap, time=self.solar_time
            )
        self.mkv = None
        self.thrust_model = None
        self.separations = []

    def set_mkv(self, mkv):
        self.mkv = mkv

    def set_thrust(self, thrust_model):
        self.thrust_model = thrust_model

    def set_inertia(self, inertia: np.ndarray, cg: Optional[np.ndarray] = None):
        """Update inertial properties after a stage separation (Phase 1B.2).

        Recomputes ``inertia_inv`` so the Newton-Euler angular term and the
        gravity-gradient torque use the post-separation bus inertia. ``cg`` is the
        post-separation centre-of-gravity offset (body frame) used by the
        gravity-gradient torque.
        """
        self.inertia = np.array(inertia, dtype=float)
        self.inertia_inv = np.linalg.inv(self.inertia)
        if cg is not None:
            self.cg = np.asarray(cg, dtype=float)

    def add_separation(self, sep):
        self.separations.append(sep)

    def aerodynamic_forces(self, q_dyn, cd, cy):
        return q_dyn * self.area * np.array([-cd, cy, 0.0])

    def aerodynamic_moments(self, q_dyn, cm, cn, cl_roll=0.0, omega=None, v_body_mag=None):
        cm_pitch = q_dyn * self.area * self.ref_length * cm
        cn_yaw = q_dyn * self.area * self.ref_length * cn
        cl_roll_moment = q_dyn * self.area * self.ref_length * cl_roll
        moment = np.array([cl_roll_moment, cm_pitch, cn_yaw])
        if omega is not None and v_body_mag is not None and v_body_mag > 1e-6:
            p = omega[0]
            q_omega = omega[1]
            r_omega = omega[2]
            c = self.ref_length
            damping = np.array([
                (p * c / (2.0 * v_body_mag)) * self.area * self.ref_length * 0.1,
                (q_omega * c / (2.0 * v_body_mag)) * self.area * self.ref_length * 0.1,
                (r_omega * c / (2.0 * v_body_mag)) * self.area * self.ref_length * 0.1,
            ])
            moment = moment + damping
        return moment

    def compute(self, t, state, surrogate_func):
        r = np.asarray(state["r"], dtype=float)
        v = np.asarray(state["v"], dtype=float)
        q = np.asarray(state["q"], dtype=float)
        omega = np.asarray(state["omega"], dtype=float)
        m = float(state["m"])
        q = quat_normalize(q)

        alt = _geodetic_altitude(r)
        v_mag = np.linalg.norm(v)
        rho = self.atmosphere.density_scalar(alt)
        q_dyn = 0.5 * rho * v_mag**2

        C = quat_to_dcm(q)
        v_body = C.T @ v
        v_body_mag = np.linalg.norm(v_body)
        mach = v_mag / max(self.atmosphere.speed_of_sound_scalar(alt), 1e-6)

        alpha = np.degrees(np.arctan2(v_body[2], v_body[0]))
        beta = np.degrees(np.arcsin(np.clip(v_body[1] / max(v_body_mag, 1e-6), -1.0, 1.0)))

        cd, cy, cm = surrogate_func(mach, alpha, beta, alt)
        cn, cl_roll = self._analytic_cn_cl_roll(mach, alpha, beta, alt)

        endo = alt < self.boundary_alt
        if not endo:
            cd *= 0.0
            cy *= 0.0

        f_aero_body = self.aerodynamic_forces(q_dyn, cd, cy)
        m_aero_body = self.aerodynamic_moments(q_dyn, cm, cn, cl_roll, omega, v_body_mag)

        f_thrust_body = np.zeros(3)
        m_thrust_body = np.zeros(3)
        mass_dot = 0.0
        if self.thrust_model is not None:
            thrust_vec = self.thrust_model.thrust_vector(t, state)
            f_thrust_body = np.asarray(thrust_vec, dtype=float)
            mass_dot = self.thrust_model.mass_rate(t, state)

        if self.atmosphere.uses_nrlmsise:
            try:
                lat, lon, _ = ecef_to_geodetic(r)
                self.atmosphere.set_exo_solar_geomagnetic(
                    lat=lat, lon=lon, time=self.solar_time
                )
            except Exception:
                pass
        g_inertial = gravity_inertial(
            r, use_j2=self.use_j2, use_high_order=self.use_high_order,
            use_j3=self.use_j3, use_j4=self.use_j4, max_degree=self.max_degree,
            use_third_body=self.use_third_body, use_tides=self.use_tides, t=t,
        )
        f_gravity_body = rotate_inertial_to_body(g_inertial, q)
        # ``gravity_inertial`` returns an acceleration; aero/thrust are forces and
        # are divided by mass below. Apply gravity as an acceleration directly so
        # it is not erroneously mass-scaled.
        f_total_body = f_aero_body + f_thrust_body

        f_total_inertial = rotate_body_to_inertial(f_total_body, q)
        m_gravity_body = gravity_gradient_torque(
            r, q, self.inertia_inv, cg=self.cg, use_j2=self.use_j2
        )
        m_total_body = m_aero_body + m_thrust_body + m_gravity_body

        dr_dt = v
        dv_dt = f_total_inertial / max(m, 1e-6) + g_inertial
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

    def _analytic_cn_cl_roll(self, mach, alpha, beta, alt):
        from ..aero.aero_analytical import newtonian_sideforce_moments_exo, linear_viscous_endo
        blend = np.clip((alt - (self.boundary_alt - self.taper_width)) /
                        (2.0 * self.taper_width), 0.0, 1.0)
        blend = 0.5 * (1.0 - np.cos(np.pi * blend))
        _, cn_exo, cl_roll_exo = newtonian_sideforce_moments_exo(mach, alpha, beta)
        _, _, _, cn_endo, cl_roll_endo = linear_viscous_endo(mach, alpha, beta)
        cn = cn_endo * (1.0 - blend) + cn_exo * blend
        cl_roll = cl_roll_endo * (1.0 - blend) + cl_roll_exo * blend
        return cn, cl_roll
