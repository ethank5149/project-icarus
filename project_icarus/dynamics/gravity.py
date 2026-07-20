from .coordinate_systems import rotate_inertial_to_body
import numpy as np


MU_EARTH = 3.986004418e14
R_EARTH = 6371e3
J2 = 1.08263e-3
J3 = -2.532e-6
J4 = -1.610e-6
J5 = -3.698e-7  # EGM2008 normalized zonal coefficients (J_n) used by Phase 1D.
J6 = 1.607e-7
J7 = -1.983e-7
J8 = -3.965e-8
J9 = 4.751e-8
J10 = 1.661e-8

# Normalized zonal coefficients J2..J10 (EGM2008), index by n.
_ZONAL_JN = {2: J2, 3: J3, 4: J4, 5: J5, 6: J6, 7: J7, 8: J8, 9: J9, 10: J10}

# Third-body and tide constants.
MU_SUN = 1.32712440018e20
MU_MOON = 4.9028e12
AU = 1.495978707e11          # Earth-Sun mean distance [m]
EARTH_MOON_DIST = 3.844e8    # mean [m]

# Low-precision Sun/Moon unit-direction helpers (geocentric inertial, ECI J2000-ish).
def _sun_direction_ecef(t):
    """Approximate geocentric unit vector to the Sun (ecliptic, mean elements)."""
    n = 2.0 * np.pi / 365.25
    M = n * (t / 86400.0)
    # longitude
    lam = M + 2.0 * 0.0167 * np.sin(M) + np.radians(1.92) * np.sin(2.0 * M)
    eps = np.radians(23.44)
    x = np.cos(lam)
    y = np.cos(eps) * np.sin(lam)
    z = np.sin(eps) * np.sin(lam)
    v = np.array([x, y, z], dtype=float)
    return v / np.linalg.norm(v)

def _moon_direction_ecef(t):
    """Approximate geocentric unit vector to the Moon (mean elements)."""
    T = t / 86400.0
    L = np.radians(218.32 + 13.176396 * T)
    Mm = np.radians(134.96 + 13.064993 * T)
    F = np.radians(93.27 + 13.229350 * T)
    lam = L + 6.29 * np.sin(Mm) - 1.27 * np.sin(L - F) \
          + 0.66 * np.sin(L - F) + 0.21 * np.sin(L + Mm)
    beta = np.radians(5.13) * np.sin(F)
    x = np.cos(beta) * np.cos(lam)
    y = np.cos(beta) * np.sin(lam)
    z = np.sin(beta)
    v = np.array([x, y, z], dtype=float)
    return v / np.linalg.norm(v)


def gravity_inertial(r_inertial, use_j2=True, j2=J2, j3=J3, j4=J4,
                     use_high_order=True, use_third_body=False, use_tides=False, t=0.0):
    r = np.linalg.norm(r_inertial)
    if r < 1e-6:
        return np.zeros(3)
    alt = r - R_EARTH
    g = -MU_EARTH / (r**3) * r_inertial

    if (use_j2 or use_high_order) and alt > 50e3:
        z = r_inertial[2]
        zr = z / r
        if use_j2:
            factor = 1.0
            factor += 1.5 * j2 * (R_EARTH / r) ** 2 * (5.0 * zr**2 - 1.0)
            if j3 != 0.0:
                factor += 0.5 * j3 * (R_EARTH / r) ** 3 * (7.0 * zr**3 - 3.0 * zr)
            if j4 != 0.0:
                factor += 0.125 * j4 * (R_EARTH / r) ** 4 * (9.0 * zr**4 + 3.0 * zr**2 - 0.6)
            g = -MU_EARTH / (r**3) * factor * r_inertial
        if use_high_order:
            # Higher-order EGM2008 zonal harmonics J5..J10 (n>4), k=0 sectorials.
            series = 0.0
            for n in range(5, 11):
                jn = _ZONAL_JN.get(n, 0.0)
                if jn == 0.0:
                    continue
                p_n = _legendre_zonal(n, zr)
                series += jn * (R_EARTH / r) ** n * p_n
            if series != 0.0:
                g = g - MU_EARTH / (r**3) * series * r_inertial

    if use_third_body:
        g = g + third_body_accel(r_inertial, t)
    if use_tides:
        g = g + solid_earth_tide_accel(r_inertial, t)
    return g


def _legendre_zonal(n, x):
    """Associated Legendre polynomial P_n^0(x) for zonal terms (n = 0..10)."""
    if n == 0:
        return 1.0
    if n == 1:
        return x
    p = np.zeros(n + 1)
    p[0] = 1.0
    p[1] = x
    for k in range(2, n + 1):
        p[k] = ((2 * k - 1) * x * p[k - 1] - (k - 1) * p[k - 2]) / k
    return p[n]


def third_body_accel(r_inertial, t):
    """Direct point-mass acceleration from Sun and Moon (third-body perturbation)."""
    r = np.asarray(r_inertial, dtype=float)
    rs = _sun_direction_ecef(t) * AU
    rm = _moon_direction_ecef(t) * EARTH_MOON_DIST
    a_sun = MU_SUN * (rs - r) / np.linalg.norm(rs - r) ** 3 - MU_SUN * rs / np.linalg.norm(rs) ** 3
    a_moon = MU_MOON * (rm - r) / np.linalg.norm(rm - r) ** 3 - MU_MOON * rm / np.linalg.norm(rm) ** 3
    return a_sun + a_moon


def solid_earth_tide_accel(r_inertial, t):
    """Solid-Earth tidal perturbation from Moon and Sun (degree-2 nominal).

    Uses the gradient of the degree-2 tidal potential:
        a(r) = k2 * GM_body * (R_earth / R_body)^3 * (1 / |r|^3)
               * (3 (r . u) u - r)
    where u is the unit vector toward the third body.
    """
    r = np.asarray(r_inertial, dtype=float)
    norm = np.linalg.norm(r)
    if norm < 1e-6:
        return np.zeros(3)
    rs = _sun_direction_ecef(t) * AU
    rm = _moon_direction_ecef(t) * EARTH_MOON_DIST
    love = 0.609  # k2 Love number (nominal)
    def _tide(mu, body_vec):
        R = np.linalg.norm(body_vec)
        u = body_vec / R
        dot = float(np.dot(r, u))
        scale = love * mu * (R_EARTH / R) ** 3 / (norm ** 3)
        return scale * (3.0 * dot * u - r)
    a = _tide(MU_MOON, rm) + _tide(MU_SUN, rs)
    return a


def gravity_body(r_inertial, q, use_j2=True):
    g_i = gravity_inertial(r_inertial, use_j2=use_j2)
    return rotate_inertial_to_body(g_i, q)


def gravity_gradient_torque(r_inertial, q, inertia_inv, use_j2=True, cg=None):
    r = np.linalg.norm(r_inertial)
    if r < 1e-6:
        return np.zeros(3)
    g_i = gravity_inertial(r_inertial, use_j2=use_j2)
    g_body = rotate_inertial_to_body(g_i, q)
    r_body = rotate_inertial_to_body(r_inertial, q)
    # Post-separation centre-of-gravity offset (body frame, metres). The gravity
    # gradient torque is evaluated about the CG, so shift the lever arm when the
    # CG is not at the body origin (Phase 1B.2).
    if cg is not None:
        cg_body = np.asarray(cg, dtype=float)
        r_body = r_body - cg_body
    return np.cross(r_body, inertia_inv @ (np.cross(r_body, g_body)))
