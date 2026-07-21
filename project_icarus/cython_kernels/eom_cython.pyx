# distutils: language = c
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, exp, pow, log, log10, fabs, sin, cos, asin, atan2, fmax, M_PI

# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def quat_normalize(np.ndarray[np.float64_t, ndim=1] q):
    cdef double norm = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    if norm > 1e-12:
        return np.array([q[0]/norm, q[1]/norm, q[2]/norm, q[3]/norm], dtype=np.float64)
    return q.copy()


@cython.boundscheck(False)
@cython.wraparound(False)
def quat_kinematics(np.ndarray[np.float64_t, ndim=1] q, np.ndarray[np.float64_t, ndim=1] omega):
    cdef double w=q[0], x=q[1], y=q[2], z=q[3]
    cdef double p=omega[0], qo=omega[1], r=omega[2]
    cdef double n = sqrt(w*w + x*x + y*y + z*z)
    if n > 1e-12:
        w /= n; x /= n; y /= n; z /= n
    return np.array([
        0.5*(-x*p - y*qo - z*r),
        0.5*( w*p + y*r - z*qo),
        0.5*( w*qo - x*r + z*p),
        0.5*( w*r + x*qo - y*p),
    ], dtype=np.float64)


@cython.boundscheck(False)
@cython.wraparound(False)
def quat_to_dcm(np.ndarray[np.float64_t, ndim=1] q):
    cdef double w=q[0], x=q[1], y=q[2], z=q[3]
    cdef double n = sqrt(w*w + x*x + y*y + z*z)
    if n > 1e-12:
        w /= n; x /= n; y /= n; z /= n
    cdef double xx=2.0*x*x, yy=2.0*y*y, zz=2.0*z*z
    cdef double xy=2.0*x*y, xz=2.0*x*z, yz=2.0*y*z
    cdef double wx=2.0*w*x, wy=2.0*w*y, wz=2.0*w*z
    return np.array([
        [1.0-yy-zz, xy-wz,     xz+wy    ],
        [xy+wz,     1.0-xx-zz, yz-wx    ],
        [xz-wy,     yz+wx,     1.0-xx-yy],
    ], dtype=np.float64)


@cython.boundscheck(False)
@cython.wraparound(False)
def rotate_body_to_inertial(np.ndarray[np.float64_t, ndim=1] v_body, np.ndarray[np.float64_t, ndim=1] q):
    cdef np.ndarray C = quat_to_dcm(q)
    return np.array([
        C[0,0]*v_body[0] + C[0,1]*v_body[1] + C[0,2]*v_body[2],
        C[1,0]*v_body[0] + C[1,1]*v_body[1] + C[1,2]*v_body[2],
        C[2,0]*v_body[0] + C[2,1]*v_body[1] + C[2,2]*v_body[2],
    ], dtype=np.float64)


@cython.boundscheck(False)
@cython.wraparound(False)
def rotate_inertial_to_body(np.ndarray[np.float64_t, ndim=1] v_inertial, np.ndarray[np.float64_t, ndim=1] q):
    cdef np.ndarray C = quat_to_dcm(q)
    return np.array([
        C[0,0]*v_inertial[0] + C[1,0]*v_inertial[1] + C[2,0]*v_inertial[2],
        C[0,1]*v_inertial[0] + C[1,1]*v_inertial[1] + C[2,1]*v_inertial[2],
        C[0,2]*v_inertial[0] + C[1,2]*v_inertial[1] + C[2,2]*v_inertial[2],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Geodetic / ECEF
# ---------------------------------------------------------------------------

cdef double _WGS84_A = 6378137.0
cdef double _WGS84_F = 1.0 / 298.257223563
cdef double _WGS84_B = _WGS84_A * (1.0 - _WGS84_F)
cdef double _WGS84_E2 = (_WGS84_A*_WGS84_A - _WGS84_B*_WGS84_B) / (_WGS84_A*_WGS84_A)


@cython.boundscheck(False)
@cython.wraparound(False)
def ecef_to_geodetic(np.ndarray[np.float64_t, ndim=1] r):
    cdef double x=r[0], y=r[1], z=r[2]
    cdef double lon = atan2(y, x)
    cdef double p = sqrt(x*x + y*y)
    cdef double lat, alt
    cdef double e_prime2 = _WGS84_E2 / (1.0 - _WGS84_E2)
    cdef double theta, slat, N
    if p < 1e-6:
        lat = 1.5707963267948966 if z > 0 else -1.5707963267948966
        alt = fabs(z) - _WGS84_B
        return np.array([lat, lon, alt], dtype=np.float64)
    theta = atan2(z * _WGS84_A, p * _WGS84_B)
    lat = atan2(
        z + e_prime2 * _WGS84_B * sin(theta)*sin(theta)*sin(theta),
        p - _WGS84_E2 * _WGS84_A * cos(theta)*cos(theta)*cos(theta),
    )
    slat = sin(lat)
    N = _WGS84_A / sqrt(1.0 - _WGS84_E2 * slat*slat)
    alt = p / cos(lat) - N
    return np.array([lat, lon, alt], dtype=np.float64)


@cython.boundscheck(False)
@cython.wraparound(False)
def geodetic_altitude(np.ndarray[np.float64_t, ndim=1] r):
    cdef double x=r[0], y=r[1], z=r[2]
    cdef double p = sqrt(x*x + y*y)
    cdef double lat, alt
    cdef double e_prime2 = _WGS84_E2 / (1.0 - _WGS84_E2)
    cdef double theta, slat, N
    if p < 1e-6:
        return fabs(z) - _WGS84_B
    theta = atan2(z * _WGS84_A, p * _WGS84_B)
    lat = atan2(
        z + e_prime2 * _WGS84_B * sin(theta)*sin(theta)*sin(theta),
        p - _WGS84_E2 * _WGS84_A * cos(theta)*cos(theta)*cos(theta),
    )
    slat = sin(lat)
    N = _WGS84_A / sqrt(1.0 - _WGS84_E2 * slat*slat)
    alt = p / cos(lat) - N
    return alt


# ---------------------------------------------------------------------------
# Gravity
# ---------------------------------------------------------------------------

cdef double MU_EARTH = 3.986004418e14
cdef double R_EARTH = 6371e3
cdef double J2 = 1.08263e-3
cdef double J3 = -2.532e-6
cdef double J4 = -1.610e-6

cdef double _J5 = -3.698e-7
cdef double _J6 =  1.607e-7
cdef double _J7 = -1.983e-7
cdef double _J8 = -3.965e-8
cdef double _J9 =  4.751e-8
cdef double _J10 = 1.661e-8


cdef double _legendre_zonal(int n, double x):
    cdef int k
    cdef double p0=1.0, p1=x, pk
    if n == 0:
        return 1.0
    if n == 1:
        return x
    for k in range(2, n+1):
        pk = ((2.0*k - 1.0)*x*p1 - (k-1.0)*p0) / k
        p0 = p1
        p1 = pk
    return p1


cdef np.ndarray _sun_direction_ecef_c(double t):
    cdef double n = 2.0 * M_PI / 365.25
    cdef double M = n * (t / 86400.0)
    cdef double lam = M + 2.0*0.0167*sin(M) + 0.03306*sin(2.0*M)
    cdef double eps = 0.2617993877991494
    cdef double x = cos(lam)
    cdef double y = cos(eps)*sin(lam)
    cdef double z = sin(eps)*sin(lam)
    cdef double norm = sqrt(x*x + y*y + z*z)
    return np.array([x/norm, y/norm, z/norm], dtype=np.float64)


cdef np.ndarray _moon_direction_ecef_c(double t):
    cdef double T = t / 86400.0
    cdef double L = 3.8070338488
    cdef double Mm = 2.34593
    cdef double F = 1.62893
    cdef double lam = L + 6.29*sin(Mm) - 1.27*sin(L-F) + 0.66*sin(L-F) + 0.21*sin(L+Mm)
    cdef double beta = 0.08946
    cdef double x = cos(beta)*cos(lam)
    cdef double y = cos(beta)*sin(lam)
    cdef double z = sin(beta)
    cdef double norm = sqrt(x*x + y*y + z*z)
    return np.array([x/norm, y/norm, z/norm], dtype=np.float64)


cdef np.ndarray _third_body_accel_c(np.ndarray[np.float64_t, ndim=1] r_inertial, double t):
    cdef double rx=r_inertial[0], ry=r_inertial[1], rz=r_inertial[2]
    cdef np.ndarray rs = _sun_direction_ecef_c(t)
    cdef np.ndarray rm = _moon_direction_ecef_c(t)
    cdef double rsx=rs[0]*1.495978707e11, rsy=rs[1]*1.495978707e11, rsz=rs[2]*1.495978707e11
    cdef double rmx=rm[0]*3.844e8,      rmy=rm[1]*3.844e8,      rmz=rm[2]*3.844e8
    cdef double dx=rsx-rx, dy=rsy-ry, dz=rsz-rz
    cdef double nr = sqrt(dx*dx + dy*dy + dz*dz)
    cdef double dx2=rmx-rx, dy2=rmy-ry, dz2=rmz-rz
    cdef double nm = sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2)
    cdef double ns = sqrt(rsx*rsx + rsy*rsy + rsz*rsz)
    cdef double nm2 = sqrt(rmx*rmx + rmy*rmy + rmz*rmz)
    return np.array([
        1.32712440018e20*dx/(nr*nr*nr) - 1.32712440018e20*rsx/(ns*ns*ns) +
        4.9028e12*dx2/(nm*nm*nm) - 4.9028e12*rmx/(nm2*nm2*nm2),
        1.32712440018e20*dy/(nr*nr*nr) - 1.32712440018e20*rsy/(ns*ns*ns) +
        4.9028e12*dy2/(nm*nm*nm) - 4.9028e12*rmy/(nm2*nm2*nm2),
        1.32712440018e20*dz/(nr*nr*nr) - 1.32712440018e20*rsz/(ns*ns*ns) +
        4.9028e12*dz2/(nm*nm*nm) - 4.9028e12*rmz/(nm2*nm2*nm2),
    ], dtype=np.float64)


cdef np.ndarray _solid_earth_tide_accel_c(np.ndarray[np.float64_t, ndim=1] r_inertial, double t):
    cdef double rx=r_inertial[0], ry=r_inertial[1], rz=r_inertial[2]
    cdef double norm = sqrt(rx*rx + ry*ry + rz*rz)
    cdef np.ndarray rs = _sun_direction_ecef_c(t)
    cdef np.ndarray rm = _moon_direction_ecef_c(t)
    cdef double rsx=rs[0]*1.495978707e11, rsy=rs[1]*1.495978707e11, rsz=rs[2]*1.495978707e11
    cdef double rmx=rm[0]*3.844e8,      rmy=rm[1]*3.844e8,      rmz=rm[2]*3.844e8
    cdef double R = sqrt(rsx*rsx + rsy*rsy + rsz*rsz)
    cdef double Rm = sqrt(rmx*rmx + rmy*rmy + rmz*rmz)
    cdef double love = 0.609
    cdef double scale_s = love * 4.9028e12 * (6371000.0/Rm)**3 / (norm*norm*norm)
    cdef double dot_s = rx*rmx/Rm + ry*rmy/Rm + rz*rmz/Rm
    cdef double sx = scale_s*(3.0*dot_s*rmx/Rm - rx)
    cdef double sy = scale_s*(3.0*dot_s*rmy/Rm - ry)
    cdef double sz = scale_s*(3.0*dot_s*rmz/Rm - rz)
    cdef double scale_m = love * 1.32712440018e20 * (6371000.0/R)**3 / (norm*norm*norm)
    cdef double dot_m = rx*rsx/R + ry*rsy/R + rz*rsz/R
    return np.array([
        sx + scale_m*(3.0*dot_m*rsx/R - rx),
        sy + scale_m*(3.0*dot_m*rsy/R - ry),
        sz + scale_m*(3.0*dot_m*rsz/R - rz),
    ], dtype=np.float64)


@cython.boundscheck(False)
@cython.wraparound(False)
def gravity_inertial(np.ndarray[np.float64_t, ndim=1] r_inertial,
                     bint use_j2=True, double j2=J2, double j3=J3, double j4=J4,
                     bint use_high_order=True, bint use_third_body=False,
                     bint use_tides=False, double t=0.0,
                     bint use_j3=True, bint use_j4=True, int max_degree=10):
    cdef double rx=r_inertial[0], ry=r_inertial[1], rz=r_inertial[2]
    cdef double r = sqrt(rx*rx + ry*ry + rz*rz)
    cdef double gx, gy, gz
    cdef double alt, z, zr, factor
    cdef int n
    cdef double jn, series, p_n
    cdef np.ndarray tb, td
    if r < 1e-6:
        return np.zeros(3, dtype=np.float64)
    gx = -MU_EARTH / (r*r*r) * rx
    gy = -MU_EARTH / (r*r*r) * ry
    gz = -MU_EARTH / (r*r*r) * rz
    alt = r - R_EARTH
    if (use_j2 or use_high_order or use_j3 or use_j4) and alt > 50000.0:
        z = rz
        zr = z / r
        factor = 1.0
        if use_j2:
            factor += 1.5 * j2 * (R_EARTH/r)**2 * (5.0*zr*zr - 1.0)
        if use_j3:
            factor += 0.5 * j3 * (R_EARTH/r)**3 * (7.0*zr*zr*zr - 3.0*zr)
        if use_j4:
            factor += 0.125 * j4 * (R_EARTH/r)**4 * (9.0*zr*zr*zr*zr + 3.0*zr*zr - 0.6)
        if use_j2 or use_j3 or use_j4:
            gx = -MU_EARTH / (r*r*r) * factor * rx
            gy = -MU_EARTH / (r*r*r) * factor * ry
            gz = -MU_EARTH / (r*r*r) * factor * rz
        if use_high_order:
            series = 0.0
            for n in range(5, max_degree+1):
                if n == 5: jn = _J5
                elif n == 6: jn = _J6
                elif n == 7: jn = _J7
                elif n == 8: jn = _J8
                elif n == 9: jn = _J9
                elif n == 10: jn = _J10
                else: jn = 0.0
                if jn == 0.0:
                    continue
                p_n = _legendre_zonal(n, zr)
                series += jn * (R_EARTH/r)**n * p_n
            if series != 0.0:
                gx = gx - MU_EARTH / (r*r*r) * series * rx
                gy = gy - MU_EARTH / (r*r*r) * series * ry
                gz = gz - MU_EARTH / (r*r*r) * series * rz
    if use_third_body:
        tb = _third_body_accel_c(r_inertial, t)
        gx += tb[0]; gy += tb[1]; gz += tb[2]
    if use_tides:
        td = _solid_earth_tide_accel_c(r_inertial, t)
        gx += td[0]; gy += td[1]; gz += td[2]
    return np.array([gx, gy, gz], dtype=np.float64)


@cython.boundscheck(False)
@cython.wraparound(False)
def gravity_gradient_torque(np.ndarray[np.float64_t, ndim=1] r_inertial,
                            np.ndarray[np.float64_t, ndim=1] q,
                            np.ndarray[np.float64_t, ndim=2] inertia_inv,
                            bint use_j2=True, cg=None):
    cdef double rx=r_inertial[0], ry=r_inertial[1], rz=r_inertial[2]
    cdef double r = sqrt(rx*rx + ry*ry + rz*rz)
    cdef double cx=0.0, cy2=0.0, cz=0.0
    cdef np.ndarray g_i, r_body, g_body
    cdef double bx, by, bz, gbx, gby, gbz
    cdef double c1, c2, c3
    if r < 1e-6:
        return np.zeros(3, dtype=np.float64)
    g_i = gravity_inertial(r_inertial, use_j2=use_j2)
    r_body = rotate_inertial_to_body(r_inertial, q)
    if cg is not None:
        cx = cg[0]; cy2 = cg[1]; cz = cg[2]
        r_body[0] -= cx; r_body[1] -= cy2; r_body[2] -= cz
    g_body = rotate_inertial_to_body(g_i, q)
    bx = r_body[0]; by = r_body[1]; bz = r_body[2]
    gbx = g_body[0]; gby = g_body[1]; gbz = g_body[2]
    c1 = by*gbz - bz*gby
    c2 = bz*gbx - bx*gbz
    c3 = bx*gby - by*gbx
    return np.array([
        inertia_inv[0,0]*c1 + inertia_inv[0,1]*c2 + inertia_inv[0,2]*c3,
        inertia_inv[1,0]*c1 + inertia_inv[1,1]*c2 + inertia_inv[1,2]*c3,
        inertia_inv[2,0]*c1 + inertia_inv[2,1]*c2 + inertia_inv[2,2]*c3,
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Aero analytical (scalar versions for hot loop)
# ---------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def newtonian_sideforce_moments_exo(double mach, double alpha, double beta):
    cdef double ar = alpha * 0.017453292519943295
    cdef double br = beta * 0.017453292519943295
    cdef double aoa = sqrt(ar*ar + br*br)
    cdef double cy = 2.0 * sin(ar) * cos(ar)
    cdef double cn = 2.0 * sin(br) * cos(br)
    return np.array([cy, cn, 0.0], dtype=np.float64)


@cython.boundscheck(False)
@cython.wraparound(False)
def linear_viscous_endo(double mach, double alpha, double beta,
                        double ref_area=0.1, double ref_length=1.0,
                        double rho=1.225, double viscosity=1.78e-5):
    cdef double ar = alpha * 0.017453292519943295
    cdef double br = beta * 0.017453292519943295
    cdef double Re = rho * mach * ref_length / max(viscosity, 1e-10)
    cdef double logRe, Cf, S_wet, Cd_friction, beta_pg, b_kt, Cd_induced
    if Re < 10.0:
        Re = 10.0
    logRe = log10(Re)
    if fabs(logRe) > 1e-12:
        Cf = 0.455 / pow(logRe, 2.58)
    else:
        Cf = 0.0
    S_wet = 3.141592653589793 * ref_length * sqrt(ref_area / 3.141592653589793) * 2.0
    Cd_friction = Cf * (S_wet / ref_area)
    beta_pg = 1.0
    if mach < 0.6:
        beta_pg = 1.0
    elif mach < 0.7:
        beta_pg = sqrt(max(1.0 - mach*mach, 1e-6))
    else:
        b_kt = (mach*mach + 2.0) / (mach*mach*sqrt(max(1.0 - 0.2*mach*mach, 1e-6)) + 2.0)
        beta_pg = b_kt
    Cd_friction = Cd_friction / max(beta_pg, 1e-6)
    Cd_induced = ar*ar + br*br
    cdef double cd = Cd_friction + Cd_induced
    if cd > 2.0:
        cd = 2.0
    elif cd < 0.0:
        cd = 0.0
    cdef double cl_side = 2.0 * 3.141592653589793 * ar + (-0.1 * Cf * (1.0 if ar >= 0 else -1.0))
    cdef double cm_pitch = -1.2 * ar + (0.05 * Cf * (1.0 if ar >= 0 else -1.0))
    cdef double cn_yaw = 2.0 * 3.141592653589793 * br
    return np.array([cd, cl_side, cm_pitch, cn_yaw, 0.0], dtype=np.float64)
