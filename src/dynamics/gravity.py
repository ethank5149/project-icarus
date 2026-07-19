from .coordinate_systems import rotate_inertial_to_body
import numpy as np


MU_EARTH = 3.986004418e14
R_EARTH = 6371e3
J2 = 1.08263e-3
J3 = -2.532e-6
J4 = -1.610e-6


def gravity_inertial(r_inertial, use_j2=True, j2=J2, j3=J3, j4=J4):
    r = np.linalg.norm(r_inertial)
    if r < 1e-6:
        return np.zeros(3)
    alt = r - R_EARTH
    if use_j2 and alt > 50e3:
        z = r_inertial[2]
        factor = -MU_EARTH / (r**5) * (
            1.0
            + 1.5 * j2 * (R_EARTH / r) ** 2 * (5.0 * (z / r) ** 2 - 1.0)
            + 0.5 * j3 * (R_EARTH / r) ** 3 * (7.0 * (z / r) ** 3 - 3.0 * (z / r))
            + 0.125 * j4 * (R_EARTH / r) ** 4 * (9.0 * (z / r) ** 4 + 3.0 * (z / r) ** 2 - 0.6)
        )
        return factor * r_inertial
    return -MU_EARTH / (r**3) * r_inertial


def gravity_body(r_inertial, q, use_j2=True):
    g_i = gravity_inertial(r_inertial, use_j2=use_j2)
    return rotate_inertial_to_body(g_i, q)


def gravity_gradient_torque(r_inertial, q, inertia_inv, use_j2=True):
    r = np.linalg.norm(r_inertial)
    if r < 1e-6:
        return np.zeros(3)
    g_i = gravity_inertial(r_inertial, use_j2=use_j2)
    g_body = rotate_inertial_to_body(g_i, q)
    r_body = rotate_inertial_to_body(r_inertial, q)
    return np.cross(r_body, inertia_inv @ (np.cross(r_body, g_body)))
