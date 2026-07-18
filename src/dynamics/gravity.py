from .coordinate_systems import rotate_inertial_to_body
import numpy as np


MU_EARTH = 3.986004418e14
R_EARTH = 6371e3


def gravity_inertial(r_inertial):
    r = np.linalg.norm(r_inertial)
    if r < 1e-6:
        return np.zeros(3)
    return -MU_EARTH / (r**3) * r_inertial


def gravity_body(r_inertial, q):
    g_i = gravity_inertial(r_inertial)
    return rotate_inertial_to_body(g_i, q)


def j2_gravity(r_inertial, j2=1.08263e-3):
    r = np.linalg.norm(r_inertial)
    if r < 1e-6:
        return np.zeros(3)
    mu = MU_EARTH
    z = r_inertial[2]
    factor = -mu / (r**5) * (1.0 + 1.5 * j2 * (R_EARTH / r) ** 2 * (5.0 * (z / r) ** 2 - 1.0))
    return factor * r_inertial


def gravity_gradient_torque(r_inertial, q, inertia_inv):
    r = np.linalg.norm(r_inertial)
    if r < 1e-6:
        return np.zeros(3)
    g_i = gravity_inertial(r_inertial)
    g_body = rotate_inertial_to_body(g_i, q)
    r_body = rotate_inertial_to_body(r_inertial, q)
    return np.cross(r_body, inertia_inv @ (np.cross(r_body, g_body)))
