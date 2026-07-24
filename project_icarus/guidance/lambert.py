"""Lambert solver for ICBM burnout targeting.

Solves the boundary-value problem: given two position vectors r1, r2
and the time of flight dt, find the velocity vector v1 at r1 that
reaches r2 in time dt under gravitational influence.

Uses the universal-variable (Stumpff-function) formulation with
Newton-Raphson with bracketing fallback. Handles elliptic, parabolic,
and hyperbolic transfers seamlessly.

Reference: improved-mathematical-analysis.md Section V (Universal Variables)
and Section VI (Gooding initial-guess heuristics).
"""

import numpy as np

MU_EARTH = 3.986004418e14


def _stumpff_c(z: float) -> float:
    """Stumpff C(z) with series/closed-form fallback."""
    if z > 1e-8:
        return (1.0 - np.cos(np.sqrt(z))) / z
    if z < -1e-8:
        return (np.cosh(np.sqrt(-z)) - 1.0) / (-z)
    return 0.5 - z / 24.0 + z**2 / 720.0


def _stumpff_s(z: float) -> float:
    """Stumpff S(z) with series/closed-form fallback."""
    if z > 1e-8:
        s = np.sqrt(z)
        return (s - np.sin(s)) / z**1.5
    if z < -1e-8:
        s = np.sqrt(-z)
        return (np.sinh(s) - s) / (-z)**1.5
    return 1.0 / 6.0 - z / 120.0 + z**2 / 5040.0


def _y_of_z(z: float, r1: float, r2: float, A: float) -> float:
    """Lambert y(z) = r1 + r2 + A*(z*S(z) - 1) / sqrt(C(z))."""
    cz = _stumpff_c(z)
    if abs(cz) < 1e-15:
        return r1 + r2
    return r1 + r2 + A * (z * _stumpff_s(z) - 1.0) / np.sqrt(cz)


def _tof_residual(z: float, r1: float, r2: float, A: float,
                   mu: float, dt: float, n_rev: int = 0) -> float:
    """Universal-variable TOF residual: TOF_calc - dt. Returns inf for invalid z."""
    cz = _stumpff_c(z)
    if abs(cz) < 1e-15:
        return 0.0
    y = _y_of_z(z, r1, r2, A)
    if y <= 1e-12:
        return np.inf
    sz = _stumpff_s(z)
    tof = (y / cz) ** 1.5 * sz + A * np.sqrt(y)
    if n_rev > 0:
        tof = tof + n_rev * np.pi * (y / cz) ** 1.5
    return tof / np.sqrt(mu) - dt


def _gooding_initial_z(r1: float, r2: float, dnu: float, A: float,
                        dt: float, mu: float) -> float:
    """Gooding-inspired initial guess for universal variable z."""
    c = np.sqrt(r1**2 + r2**2 - 2.0 * r1 * r2 * np.cos(dnu))
    s = (r1 + r2 + c) / 2.0
    q = np.sqrt(r1 * r2) / s * np.cos(dnu / 2.0)
    T = np.sqrt(8.0 * mu / s**3) * dt
    T0 = 2.0 / 3.0 * (1.0 - q**3)

    if abs(T - T0) < 1e-12:
        return 0.0

    if T > T0:
        dT = T - T0
        x = -dT / (dT + T0 * (1.0 + (1.0 + q) / 2.0 * np.sqrt(dT / (dT + T0))))
    else:
        dT = T0 - T
        x = dT / (T0 * (1.0 + (1.0 - q) / 2.0 * dT / T0 * (1.0 + np.sqrt(dT / T0))))

    a_min = s / 2.0
    z = (1.0 - x**2) / a_min if abs(a_min) > 1e-15 else 0.0
    return z


def lambert(r1: np.ndarray, r2: np.ndarray, dt: float, mu: float = MU_EARTH,
            short_way: bool = True, num_iter: int = 50,
            tol: float = 1e-10) -> np.ndarray:
    """Solve the Lambert problem for velocity at r1.

    Uses the universal-variable formulation with Newton-Raphson
    and bracketing fallback for robustness.

    Parameters
    ----------
    r1, r2 : ndarray
        Initial and final position vectors (m).
    dt : float
        Time of flight (s). Must be positive.
    mu : float
        Gravitational parameter (m^3/s^2).
    short_way : bool
        If True, use the short-way transfer (dnu < pi).
    num_iter : int
        Maximum number of Newton iterations.
    tol : float
        Convergence tolerance on time residual (s).

    Returns
    -------
    v1 : ndarray
        Velocity vector at r1 (m/s) that reaches r2 in time dt.
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)

    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    if r1_mag < 1e-9 or r2_mag < 1e-9:
        raise ValueError("Position vectors must have non-zero magnitude.")
    if dt <= 0:
        raise ValueError("Time of flight must be positive.")

    cos_dnu = np.dot(r1, r2) / (r1_mag * r2_mag)
    cos_dnu = np.clip(cos_dnu, -1.0, 1.0)
    dnu = np.arccos(cos_dnu)
    if not short_way:
        dnu = 2.0 * np.pi - dnu

    if abs(np.sin(dnu)) < 1e-15:
        raise ValueError("Position vectors are collinear; Lambert problem is degenerate.")

    A = np.sqrt(r1_mag * r2_mag * (1.0 + np.cos(dnu)))
    if not short_way:
        A = -A

    # Bracket the valid z range
    c = np.sqrt(r1_mag**2 + r2_mag**2 - 2.0 * r1_mag * r2_mag * np.cos(dnu))
    s = (r1_mag + r2_mag + c) / 2.0
    a_min = s / 2.0
    z_lo = 1e-12
    z_hi = max(100.0, 1.0 / a_min * 100.0) if a_min > 1e-15 else 100.0

    res_lo = _tof_residual(z_lo, r1_mag, r2_mag, A, mu, dt, 0)
    res_hi = _tof_residual(z_hi, r1_mag, r2_mag, A, mu, dt, 0)

    if not np.isfinite(res_lo) or not np.isfinite(res_hi) or res_lo * res_hi > 0:
        if res_lo > 0:
            z_lo = z_lo * 0.1
            res_lo = _tof_residual(z_lo, r1_mag, r2_mag, A, mu, dt, 0)
        if res_hi < 0:
            z_hi = z_hi * 10.0
            res_hi = _tof_residual(z_hi, r1_mag, r2_mag, A, mu, dt, 0)

    z = _gooding_initial_z(r1_mag, r2_mag, dnu, A, dt, mu)
    z = np.clip(z, z_lo * 0.5, z_hi * 2.0)
    sqrt_mu = np.sqrt(mu)

    # Newton-Raphson with bracketing fallback
    converged = False
    for _ in range(num_iter):
        res = _tof_residual(z, r1_mag, r2_mag, A, mu, dt, 0)
        if abs(res) < tol:
            converged = True
            break
        if not np.isfinite(res):
            break
        dz = 1e-8 * max(1.0, abs(z))
        res_p = _tof_residual(z + dz, r1_mag, r2_mag, A, mu, dt, 0)
        res_m = _tof_residual(z - dz, r1_mag, r2_mag, A, mu, dt, 0)
        if not np.isfinite(res_p) or not np.isfinite(res_m):
            break
        dT_dz = (res_p - res_m) / (2.0 * dz)
        if abs(dT_dz) < 1e-15:
            break
        step = res / (dT_dz / sqrt_mu)
        z_new = z - step
        if z_new <= z_lo or z_new >= z_hi:
            break
        z = z_new

    # Bisection fallback
    if not converged:
        for _ in range(num_iter):
            z_mid = (z_lo + z_hi) / 2.0
            res_mid = _tof_residual(z_mid, r1_mag, r2_mag, A, mu, dt, 0)
            if abs(res_mid) < tol or (z_hi - z_lo) < 1e-15:
                z = z_mid
                converged = True
                break
            if res_lo * res_mid < 0:
                z_hi = z_mid
                res_hi = res_mid
            else:
                z_lo = z_mid
                res_lo = res_mid
            z = (z_lo + z_hi) / 2.0

    y = _y_of_z(z, r1_mag, r2_mag, A)
    if y <= 0.0:
        y = 1e-15
    f = 1.0 - y / r1_mag
    g = A * np.sqrt(y / mu) if abs(mu) > 1e-15 else 0.0
    if abs(g) < 1e-15:
        g = dt / (1.0 - f + 1e-15)
    v1 = (r2 - f * r1) / g
    return v1


def required_burnout_velocity_lambert(
    r_burnout: np.ndarray,
    r_target: np.ndarray,
    tof_coast: float,
    mu: float = MU_EARTH,
) -> np.ndarray:
    """Compute the required burnout velocity using Lambert's problem.

    Parameters
    ----------
    r_burnout : ndarray
        Position vector at engine cutoff (m).
    r_target : ndarray
        Target position vector (m).
    tof_coast : float
        Coast time of flight from burnout to impact (s).
    mu : float
        Gravitational parameter (m^3/s^2).

    Returns
    -------
    v_burnout : ndarray
        Required velocity vector at burnout (m/s) in the same frame as r_burnout.
    """
    return lambert(r_burnout, r_target, tof_coast, mu)


def lambert_with_correction(
    r_burnout: np.ndarray,
    r_target: np.ndarray,
    tof_coast: float,
    mu: float,
    propagate_fn,
    max_iter: int = 10,
    tol_miss: float = 100.0,
    tol_dv: float = 1.0,
    dv_perturb: float = 50.0,
) -> np.ndarray:
    """Lambert solution refined by a differential corrector.

    Uses Lambert's problem for an analytical initial guess, then iteratively
    corrects the burnout velocity by propagating under the full perturbed
    gravity model specified by ``propagate_fn``.

    Parameters
    ----------
    r_burnout, r_target : ndarray
        Burnout position and target position (m).
    tof_coast : float
        Coast time of flight (s).
    mu : float
        Gravitational parameter (m^3/s^2).
    propagate_fn : callable
        ``propagate_fn(r, v) -> r_impact``; integrates a coast trajectory
        from ``(r, v)`` to impact and returns the final position.
    max_iter : int
        Maximum correction iterations.
    tol_miss : float
        Converge when miss distance (m) is below this threshold.
    tol_dv : float
        Converge when velocity correction magnitude (m/s) is below this threshold.
    dv_perturb : float
        Perturbation magnitude (m/s) for finite-difference Jacobian.

    Returns
    -------
    v_corrected : ndarray
        Corrected burnout velocity (m/s).
    """
    r_burnout = np.asarray(r_burnout, dtype=float)
    r_target = np.asarray(r_target, dtype=float)

    v_guess = lambert(r_burnout, r_target, tof_coast, mu)

    for _ in range(max_iter):
        impact_r = propagate_fn(r_burnout, v_guess)
        miss = impact_r - r_target
        miss_3d = float(np.linalg.norm(miss))
        if miss_3d < tol_miss:
            return v_guess

        dv = dv_perturb
        jac = np.zeros((3, 3))
        for axis in range(3):
            delta = np.zeros(3)
            delta[axis] = dv
            r_p = propagate_fn(r_burnout, v_guess + delta)
            jac[:, axis] = (r_p - impact_r) / dv

        try:
            delta_v = np.linalg.solve(jac, -miss)
        except np.linalg.LinAlgError:
            delta_v = -miss / (miss_3d + 1e-12) * min(miss_3d, dv)

        v_new = v_guess + delta_v
        if float(np.linalg.norm(v_new - v_guess)) < tol_dv:
            return v_new
        v_guess = v_new

    return v_guess

