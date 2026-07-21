# distutils: language = c
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, fmax, fmin, fabs, acos, M_PI

# ---------------------------------------------------------------------------
# Terminal guidance kernels
# ---------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def los_vec(np.ndarray[np.float64_t, ndim=1] interceptor_r,
            np.ndarray[np.float64_t, ndim=1] interceptor_v,
            np.ndarray[np.float64_t, ndim=1] target_r,
            np.ndarray[np.float64_t, ndim=1] target_v):
    cdef double losx = target_r[0] - interceptor_r[0]
    cdef double losy = target_r[1] - interceptor_r[1]
    cdef double losz = target_r[2] - interceptor_r[2]
    cdef double range_ = sqrt(losx*losx + losy*losy + losz*losz)
    cdef double rinv = 1.0 / max(range_, 1e-6)
    cdef double rel_vx = target_v[0] - interceptor_v[0]
    cdef double rel_vy = target_v[1] - interceptor_v[1]
    cdef double rel_vz = target_v[2] - interceptor_v[2]
    return (np.array([losx, losy, losz], dtype=np.float64),
            range_,
            np.array([losx*rinv, losy*rinv, losz*rinv], dtype=np.float64),
            np.array([rel_vx, rel_vy, rel_vz], dtype=np.float64))


@cython.boundscheck(False)
@cython.wraparound(False)
def in_fov(np.ndarray[np.float64_t, ndim=1] los_unit, double fov):
    cdef double dot = los_unit[0]
    cdef double angle = acos(fmax(fmin(dot, 1.0), -1.0))
    return angle <= fov


@cython.boundscheck(False)
@cython.wraparound(False)
def pn_cmd(double N, double Vc, np.ndarray[np.float64_t, ndim=1] los_dot,
           double accel_limit, np.ndarray[np.float64_t, ndim=1] target_accel):
    cdef double ax = N * Vc * los_dot[0]
    cdef double ay = N * Vc * los_dot[1]
    cdef double az = N * Vc * los_dot[2]
    if target_accel is not None and target_accel.shape[0] == 3:
        ax += (N / 2.0) * target_accel[0]
        ay += (N / 2.0) * target_accel[1]
        az += (N / 2.0) * target_accel[2]
    cdef double mag = sqrt(ax*ax + ay*ay + az*az)
    if mag > accel_limit:
        ax = ax * accel_limit / mag
        ay = ay * accel_limit / mag
        az = az * accel_limit / mag
    return np.array([ax, ay, az], dtype=np.float64)


@cython.boundscheck(False)
@cython.wraparound(False)
def zem_cmd(np.ndarray[np.float64_t, ndim=1] los,
            np.ndarray[np.float64_t, ndim=1] rel_vel,
            double range_, double N, double zem_horizon, double accel_limit):
    cdef double losx=los[0], losy=los[1], losz=los[2]
    cdef double rvx=rel_vel[0], rvy=rel_vel[1], rvz=rel_vel[2]
    cdef double Vc = -(rvx*losx + rvy*losy + rvz*losz) / max(range_, 1e-6)
    cdef double t_go = range_ / max(fabs(Vc), 1e-3)
    if t_go < zem_horizon * 0.1:
        t_go = zem_horizon * 0.1
    if t_go > zem_horizon:
        t_go = zem_horizon
    cdef double zemx = losx + t_go * rvx
    cdef double zemy = losy + t_go * rvy
    cdef double zemz = losz + t_go * rvz
    cdef double los_inv = 1.0 / max(range_, 1e-6)
    cdef double dot = zemx*losx*los_inv + zemy*losy*los_inv + zemz*losz*los_inv
    cdef double perpx = zemx - dot*losx
    cdef double perpy = zemy - dot*losy
    cdef double perpz = zemz - dot*losz
    cdef double ax = (N / max(t_go, 1e-3)) * perpx
    cdef double ay = (N / max(t_go, 1e-3)) * perpy
    cdef double az = (N / max(t_go, 1e-3)) * perpz
    cdef double mag = sqrt(ax*ax + ay*ay + az*az)
    if mag > accel_limit:
        ax *= accel_limit / mag
        ay *= accel_limit / mag
        az *= accel_limit / mag
    return np.array([ax, ay, az], dtype=np.float64)


@cython.boundscheck(False)
@cython.wraparound(False)
def sdre_mpc_cmd(np.ndarray[np.float64_t, ndim=1] los,
                 np.ndarray[np.float64_t, ndim=1] rel_vel,
                 double range_, double N, double zem_horizon,
                 double sdre_q_pos, double sdre_q_vel, double sdre_r_accel,
                 double accel_limit):
    cdef double losx=los[0], losy=los[1], losz=los[2]
    cdef double rvx=rel_vel[0], rvy=rel_vel[1], rvz=rel_vel[2]
    cdef double Vc = -(rvx*losx + rvy*losy + rvz*losz) / max(range_, 1e-6)
    cdef double t_go = range_ / max(fabs(Vc), 1e-3)
    if t_go < zem_horizon * 0.1:
        t_go = zem_horizon * 0.1
    if t_go > zem_horizon:
        t_go = zem_horizon
    cdef double kp = 1.4142135623730951 * sqrt(sdre_q_pos / max(sdre_r_accel, 1e-9)) * t_go
    cdef double kv = sqrt(sdre_q_vel / max(sdre_r_accel, 1e-9)) * t_go
    cdef double ax = -(kp*losx + kv*rvx)
    cdef double ay = -(kp*losy + kv*rvy)
    cdef double az = -(kp*losz + kv*rvz)
    cdef double mag = sqrt(ax*ax + ay*ay + az*az)
    if mag > accel_limit:
        ax *= accel_limit / mag
        ay *= accel_limit / mag
        az *= accel_limit / mag
    return np.array([ax, ay, az], dtype=np.float64)


# ---------------------------------------------------------------------------
# Midcourse guidance kernels
# ---------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def midcourse_pn(np.ndarray[np.float64_t, ndim=1] interceptor_r,
                 np.ndarray[np.float64_t, ndim=1] interceptor_v,
                 np.ndarray[np.float64_t, ndim=1] target_state,
                 double N, double accel_limit):
    cdef double losx = target_state[0] - interceptor_r[0]
    cdef double losy = target_state[1] - interceptor_r[1]
    cdef double losz = target_state[2] - interceptor_r[2]
    cdef double range_ = sqrt(losx*losx + losy*losy + losz*losz)
    cdef double rinv = 1.0 / max(range_, 1e-6)
    cdef double los_unitx = losx * rinv
    cdef double los_unity = losy * rinv
    cdef double los_unitz = losz * rinv
    cdef double rel_vx = target_state[3] - interceptor_v[0]
    cdef double rel_vy = target_state[4] - interceptor_v[1]
    cdef double rel_vz = target_state[5] - interceptor_v[2]
    cdef double dot = rel_vx*los_unitx + rel_vy*los_unity + rel_vz*los_unitz
    cdef double Vc = -dot
    cdef double los_dotx = (rel_vx - dot*los_unitx) * rinv
    cdef double los_doty = (rel_vy - dot*los_unity) * rinv
    cdef double los_dotz = (rel_vz - dot*los_unitz) * rinv
    cdef double ax = N * Vc * los_dotx
    cdef double ay = N * Vc * los_doty
    cdef double az = N * Vc * los_dotz
    cdef double mag = sqrt(ax*ax + ay*ay + az*az)
    if mag > accel_limit:
        ax *= accel_limit / mag
        ay *= accel_limit / mag
        az *= accel_limit / mag
    return np.array([ax, ay, az], dtype=np.float64)
