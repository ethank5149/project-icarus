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
                      use_high_order=True, use_third_body=False, use_tides=False,
                      t=0.0, use_j3=True, use_j4=True, max_degree=10):
    """Central + zonal geopotential acceleration (EGM2008 zonal coefficients).

    The zonal series is now selectable by ``max_degree`` (the "higher-order
    EGM2008 toggle" from Phase 3.1) and J3/J4 can be enabled independently of
    J2. ``use_high_order`` retains its historical meaning (include J5..J10);
    when ``max_degree`` is set it overrides the upper bound explicitly. J2/J3/J4
    are each individually toggled via ``use_j2``/``use_j3``/``use_j4``.
    """
    r = np.linalg.norm(r_inertial)
    if r < 1e-6:
        return np.zeros(3)
    alt = r - R_EARTH
    g = -MU_EARTH / (r**3) * r_inertial

    if (use_j2 or use_high_order or use_j3 or use_j4) and alt > 50e3:
        z = r_inertial[2]
        zr = z / r
        factor = 1.0
        if use_j2:
            factor += 1.5 * j2 * (R_EARTH / r) ** 2 * (5.0 * zr**2 - 1.0)
        if use_j3:
            factor += 0.5 * j3 * (R_EARTH / r) ** 3 * (7.0 * zr**3 - 3.0 * zr)
        if use_j4:
            factor += 0.125 * j4 * (R_EARTH / r) ** 4 * (9.0 * zr**4 + 3.0 * zr**2 - 0.6)
        if use_j2 or use_j3 or use_j4:
            g = -MU_EARTH / (r**3) * factor * r_inertial
        if use_high_order:
            # Higher-order EGM2008 zonal harmonics, selectable up to max_degree.
            series = 0.0
            for n in range(5, max_degree + 1):
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


# ---------------------------------------------------------------------------
# Full-fidelity Numba JIT acceleration (preserves ALL physics: J2-J10,
# third-body, solid-Earth tides). The Python functions above remain the
# canonical reference; these JIT wrappers are drop-in replacements for the
# hot propagation loops in target_factory.py.
# ---------------------------------------------------------------------------
try:
    from numba import njit

    @njit
    def _sun_direction_ecef_jit(t):
        n = 2.0 * 3.141592653589793 / 365.25
        M = n * (t / 86400.0)
        lam = M + 2.0 * 0.0167 * np.sin(M) + 0.03306 * np.sin(2.0 * M)
        eps = 0.2617993877991494  # 23.44 deg
        x = np.cos(lam)
        y = np.cos(eps) * np.sin(lam)
        z = np.sin(eps) * np.sin(lam)
        norm = np.sqrt(x*x + y*y + z*z)
        return np.array([x/norm, y/norm, z/norm])

    @njit
    def _moon_direction_ecef_jit(t):
        T = t / 86400.0
        L = 3.8070338488  # 218.32 deg in rad
        Mm = 2.34593  # 134.96 deg in rad
        F = 1.62893  # 93.27 deg in rad
        lam = L + 6.29 * np.sin(Mm) - 1.27 * np.sin(L - F) + 0.66 * np.sin(L - F) + 0.21 * np.sin(L + Mm)
        beta = 0.08946  # 5.13 deg in rad
        x = np.cos(beta) * np.cos(lam)
        y = np.cos(beta) * np.sin(lam)
        z = np.sin(beta)
        norm = np.sqrt(x*x + y*y + z*z)
        return np.array([x/norm, y/norm, z/norm])

    @njit
    def _legendre_zonal_jit(n, x):
        if n == 0:
            return 1.0
        if n == 1:
            return x
        p0 = 1.0
        p1 = x
        for k in range(2, n + 1):
            pk = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k
            p0 = p1
            p1 = pk
        return p1

    @njit
    def third_body_accel_jit(r_inertial, t):
        r = r_inertial
        rs = _sun_direction_ecef_jit(t) * 1.495978707e11
        rm = _moon_direction_ecef_jit(t) * 3.844e8
        a_sun = 1.32712440018e20 * (rs - r) / np.linalg.norm(rs - r)**3 - 1.32712440018e20 * rs / np.linalg.norm(rs)**3
        a_moon = 4.9028e12 * (rm - r) / np.linalg.norm(rm - r)**3 - 4.9028e12 * rm / np.linalg.norm(rm)**3
        return a_sun + a_moon

    @njit
    def solid_earth_tide_accel_jit(r_inertial, t):
        r = r_inertial
        norm = np.linalg.norm(r)
        if norm < 1e-6:
            return np.zeros(3)
        rs = _sun_direction_ecef_jit(t) * 1.495978707e11
        rm = _moon_direction_ecef_jit(t) * 3.844e8
        love = 0.609
        def _tide(mu, body_vec):
            R = np.linalg.norm(body_vec)
            u = body_vec / R
            dot = r[0]*u[0] + r[1]*u[1] + r[2]*u[2]
            scale = love * mu * (6371000.0 / R) ** 3 / (norm ** 3)
            return scale * (3.0 * dot * u - r)
        a = _tide(4.9028e12, rm) + _tide(1.32712440018e20, rs)
        return a

    @njit
    def gravity_inertial_jit(r_inertial, use_j2, j2, j3, j4, use_high_order, use_third_body, use_tides, t, use_j3_flag, use_j4_flag, max_degree):
        r = np.linalg.norm(r_inertial)
        if r < 1e-6:
            return np.zeros(3)
        g = -3.986004418e14 / (r**3) * r_inertial
        
        alt = r - 6371000.0
        if (use_j2 or use_high_order or use_j3_flag or use_j4_flag) and alt > 50000.0:
            z = r_inertial[2]
            zr = z / r
            factor = 1.0
            if use_j2:
                factor += 1.5 * j2 * (6371000.0 / r) ** 2 * (5.0 * zr**2 - 1.0)
            if use_j3_flag:
                factor += 0.5 * j3 * (6371000.0 / r) ** 3 * (7.0 * zr**3 - 3.0 * zr)
            if use_j4_flag:
                factor += 0.125 * j4 * (6371000.0 / r) ** 4 * (9.0 * zr**4 + 3.0 * zr**2 - 0.6)
            if use_j2 or use_j3_flag or use_j4_flag:
                g = -3.986004418e14 / (r**3) * factor * r_inertial
            if use_high_order:
                series = 0.0
                for n in range(5, max_degree + 1):
                    jn = 0.0
                    if n == 5:
                        jn = -3.698e-7
                    elif n == 6:
                        jn = 1.607e-7
                    elif n == 7:
                        jn = -1.983e-7
                    elif n == 8:
                        jn = -3.965e-8
                    elif n == 9:
                        jn = 4.751e-8
                    elif n == 10:
                        jn = 1.661e-8
                    if jn == 0.0:
                        continue
                    p_n = _legendre_zonal_jit(n, zr)
                    series += jn * (6371000.0 / r) ** n * p_n
                if series != 0.0:
                    g = g - 3.986004418e14 / (r**3) * series * r_inertial
        
        if use_third_body:
            g = g + third_body_accel_jit(r_inertial, t)
        if use_tides:
            g = g + solid_earth_tide_accel_jit(r_inertial, t)
        return g

    GRAVITY_JIT_AVAILABLE = True
except Exception:
    GRAVITY_JIT_AVAILABLE = False
