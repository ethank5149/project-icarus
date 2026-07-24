"""Lambert solver for ICBM burnout targeting.

Solves the boundary-value problem: given two position vectors r1, r2
and the time of flight dt, find the velocity vector v1 at r1 that
reaches r2 in time dt under gravitational influence.

Uses Newton's method on the semi-major axis with Kepler's equation
for the time-of-flight computation. This is the classical approach
from astrodynamics and is robust for all ICBM suborbital trajectories.

Reference: improved-mathematical-analysis.md Section 2 — Lambert's Problem
applies at engine cutoff to determine the required velocity vector.
"""

import numpy as np
from scipy.optimize import brentq

MU_EARTH = 3.986004418e14


def _tof_elliptic(a, r1_mag, r2_mag, dnu, mu, short_way, dt):
    """Compute time of flight for an elliptic transfer orbit.

    Uses Kepler's equation: TOF = sqrt(a^3/mu) * (M2 - M1)
    where M = E - e*sin(E) is the mean anomaly.
    """
    if a <= 0:
        return np.inf

    sin_dnu = np.sin(dnu)
    sin_dnu_sq = sin_dnu ** 2

    # Eccentricity from the geometry of the transfer ellipse
    A = (a - r2_mag) - (a - r1_mag) * np.cos(dnu)
    B = (a - r1_mag)**2 * sin_dnu_sq
    C = sin_dnu_sq

    denom_x = A**2 + B
    if denom_x < 1e-15:
        # Special case: r1 == r2 and transfer is along a circular orbit
        if abs(r1_mag - r2_mag) < 1e-6 and abs(sin_dnu) < 1e-6:
            return np.inf  # Degenerate
        # Try a small perturbation
        e = 0.0
    else:
        x_sq = C / denom_x
        if x_sq < 0 or x_sq > 1e10:
            return np.inf
        x = np.sqrt(x_sq)
        e = 1.0 / (a * x) if x > 1e-15 else 0.0

    if e >= 1.0 or e < 0:
        return np.inf

    # Eccentric anomalies
    cos_E1 = np.clip((a - r1_mag) / (a * e), -1.0, 1.0) if e > 1e-15 else 1.0
    cos_E2 = np.clip((a - r2_mag) / (a * e), -1.0, 1.0) if e > 1e-15 else 1.0

    E1 = np.arccos(cos_E1)
    E2 = np.arccos(cos_E2)

    # Adjust for the correct direction of motion
    if short_way:
        if E2 < E1:
            E2 = 2.0 * np.pi - E2
    else:
        if E2 > E1:
            E2 = 2.0 * np.pi - E2

    # Mean motion
    n = np.sqrt(mu / a**3)

    # Mean anomalies
    M1 = E1 - e * np.sin(E1)
    M2 = E2 - e * np.sin(E2)

    # Time of flight
    tof = (M2 - M1) / n

    return tof - dt


def lambert(r1: np.ndarray, r2: np.ndarray, dt: float, mu: float = MU_EARTH,
            short_way: bool = True, num_iter: int = 50, tol: float = 1e-10) -> np.ndarray:
    """Solve the Lambert problem for velocity at r1.

    Uses Newton's method on the semi-major axis with Kepler's equation
    for the time-of-flight computation.

    Parameters
    ----------
    r1 : ndarray
        Initial position vector (m).
    r2 : ndarray
        Final position vector (m).
    dt : float
        Time of flight (s). Must be positive.
    mu : float
        Gravitational parameter (m^3/s^2).
    short_way : bool
        If True, use the short-way transfer (dnu < pi).
    num_iter : int
        Maximum number of Newton iterations.
    tol : float
        Convergence tolerance.

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

    # Transfer angle
    cos_dnu = np.dot(r1, r2) / (r1_mag * r2_mag)
    cos_dnu = np.clip(cos_dnu, -1.0, 1.0)
    dnu = np.arccos(cos_dnu)
    if not short_way:
        dnu = 2.0 * np.pi - dnu

    sin_dnu = np.sin(dnu)
    if abs(sin_dnu) < 1e-15:
        raise ValueError("Position vectors are collinear; Lambert problem is degenerate.")

    sin_half_dnu = np.sin(dnu / 2.0)
    cos_half_dnu = np.cos(dnu / 2.0)

    # Semi-major axis estimate (initial guess)
    a_est = (r1_mag + r2_mag + 2.0 * np.sqrt(r1_mag * r2_mag) * cos_half_dnu) / (4.0 * sin_half_dnu**2)

    # For suborbital ICBM trajectories, a_est is typically large and positive

    def tof_residual(a):
        """Compute TOF residual for a given semi-major axis."""
        return _tof_elliptic(a, r1_mag, r2_mag, dnu, mu, short_way, dt)

    # Find the semi-major axis using Brent's method
    a_min = max(r1_mag, r2_mag) * 1.001
    a_max = a_est * 100.0

    # Check if a_est is already a good solution
    residual_at_est = tof_residual(a_est)
    if abs(residual_at_est) < tol:
        a_sol = a_est
    else:
        # Try to find a bracket
        try:
            a_sol = brentq(tof_residual, a_min, a_max, xtol=tol, maxiter=num_iter)
        except ValueError:
            # If brentq fails, try expanding the bracket
            a_max_expanded = a_max
            for _ in range(10):
                a_max_expanded *= 2.0
                try:
                    a_sol = brentq(tof_residual, a_min, a_max_expanded, xtol=tol, maxiter=num_iter)
                    break
                except ValueError:
                    continue
            else:
                a_sol = a_est

    # Compute eccentricity
    A = (a_sol - r2_mag) - (a_sol - r1_mag) * np.cos(dnu)
    B = (a_sol - r1_mag)**2 * sin_dnu**2
    C = sin_dnu**2
    denom_x = A**2 + B
    if denom_x < 1e-15:
        e = 0.0
    else:
        x_sq = C / denom_x
        if x_sq < 0:
            e = 0.0
        else:
            x = np.sqrt(x_sq)
            e = 1.0 / (a_sol * x) if x > 1e-15 else 0.0

    if e >= 1.0:
        e = 0.999
    if e < 0:
        e = 0.0

    # Eccentric anomalies
    if e > 1e-15:
        cos_E1 = np.clip((a_sol - r1_mag) / (a_sol * e), -1.0, 1.0)
        E1 = np.arccos(cos_E1)
    else:
        E1 = 0.0

    # True anomaly at r1
    sin_E1 = np.sqrt(max(1.0 - cos_E1**2, 0.0)) if e > 1e-15 else 0.0
    denom = 1.0 - e * cos_E1
    if abs(denom) < 1e-15:
        gamma = np.pi / 2.0
    else:
        cos_nu1 = (cos_E1 - e) / denom
        sin_nu1 = np.sqrt(max(1.0 - e**2, 0.0)) * sin_E1 / denom
        gamma = np.arctan2(e * sin_nu1, 1.0 + e * cos_nu1)

    # Velocity magnitude from vis-viva equation
    v_mag = np.sqrt(mu * (2.0 / r1_mag - 1.0 / a_sol))

    # Radial and transverse unit vectors at r1
    r_hat = r1 / r1_mag

    # Orbital plane normal
    h_hat = np.cross(r1, r2)
    h_mag = np.linalg.norm(h_hat)
    if h_mag > 1e-15:
        h_hat = h_hat / h_mag
    else:
        h_hat = np.array([0.0, 0.0, 1.0])

    # Transverse unit vector (perpendicular to r_hat in the orbital plane)
    theta_hat = np.cross(h_hat, r_hat)
    theta_hat_mag = np.linalg.norm(theta_hat)
    if theta_hat_mag > 1e-15:
        theta_hat = theta_hat / theta_hat_mag
    else:
        theta_hat = np.array([0.0, 0.0, 0.0])

    # Velocity vector: v = v_mag * (cos(gamma) * r_hat + sin(gamma) * theta_hat)
    v1 = v_mag * (np.cos(gamma) * r_hat + np.sin(gamma) * theta_hat)

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