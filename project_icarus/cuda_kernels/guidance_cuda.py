"""
CUDA-accelerated guidance kernels for batch Monte Carlo.

These kernels preserve the exact guidance logic of the CPU terminal/midcourse
guidance while vectorizing across trials on the GPU.
"""

import numpy as np
import cupy as cp


# ---------------------------------------------------------------------------
# Batch guidance helpers
# ---------------------------------------------------------------------------
def batch_los_vec(interceptor_pos, interceptor_vel, target_pos, target_vel):
    """Compute LOS vector, range, and unit LOS for a batch of trials.

    Parameters
    ----------
    interceptor_pos : (n, 3) CuPy array
    interceptor_vel : (n, 3) CuPy array
    target_pos : (n, 3) CuPy array
    target_vel : (n, 3) CuPy array
    Returns
    -------
    los : (n, 3) CuPy array
    range_ : (n,) CuPy array
    los_unit : (n, 3) CuPy array
    rel_vel : (n, 3) CuPy array
    """
    los = target_pos - interceptor_pos
    range_ = cp.linalg.norm(los, axis=1)
    los_unit = los / cp.clip(range_[:, None], 1e-6, None)
    rel_vel = target_vel - interceptor_vel
    return los, range_, los_unit, rel_vel


def batch_pn_cmd(N, Vc, los_dot, accel_limit, target_accel=None):
    """Batch proportional navigation command.

    Parameters
    ----------
    N : float
        Navigation gain.
    Vc : (n,) CuPy array
        Closing speed.
    los_dot : (n, 3) CuPy array
        LOS rate vector.
    accel_limit : float
        Acceleration limit.
    target_accel : (n, 3) CuPy array, optional
        Target acceleration for APN.
    Returns
    -------
    a_cmd : (n, 3) CuPy array
    """
    a_cmd = N[:, None] * Vc[:, None] * los_dot
    if target_accel is not None:
        a_cmd = a_cmd + (N[:, None] / 2.0) * target_accel
    mag = cp.linalg.norm(a_cmd, axis=1)
    scale = cp.clip(accel_limit / cp.clip(mag, 1e-9, None), None, 1.0)
    return a_cmd * scale[:, None]


def batch_zem_cmd(los, rel_vel, range_, N, zem_horizon, accel_limit):
    """Batch zero-effort-miss guidance command.

    Parameters
    ----------
    los : (n, 3) CuPy array
    rel_vel : (n, 3) CuPy array
    range_ : (n,) CuPy array
    N : float
    zem_horizon : float
    accel_limit : float
    Returns
    -------
    a_cmd : (n, 3) CuPy array
    """
    los_unit = los / cp.clip(range_[:, None], 1e-6, None)
    Vc = -cp.sum(rel_vel * los_unit, axis=1)
    t_go = range_ / cp.clip(cp.abs(Vc), 1e-3, None)
    t_go = cp.clip(t_go, zem_horizon * 0.1, zem_horizon)
    zem = los + t_go[:, None] * rel_vel
    zem_perp = zem - cp.sum(zem * los_unit, axis=1)[:, None] * los_unit
    a_cmd = (N / cp.clip(t_go, 1e-3, None))[:, None] * zem_perp
    mag = cp.linalg.norm(a_cmd, axis=1)
    scale = cp.clip(accel_limit / cp.clip(mag, 1e-9, None), None, 1.0)
    return a_cmd * scale[:, None]


def batch_midcourse_pn(r, v, target_state, N, accel_limit):
    """Batch midcourse proportional navigation.

    Parameters
    ----------
    r : (n, 3) CuPy array
    v : (n, 3) CuPy array
    target_state : (n, 6) CuPy array
    N : float
    accel_limit : float
    Returns
    -------
    a_cmd : (n, 3) CuPy array
    """
    target_pos = target_state[:, 0:3]
    target_vel = target_state[:, 3:6]
    los, range_, los_unit, rel_vel = batch_los_vec(r, v, target_pos, target_vel)
    Vc = -cp.sum(rel_vel * los_unit, axis=1)
    los_dot = (rel_vel - cp.sum(rel_vel * los_unit, axis=1)[:, None] * los_unit) / cp.clip(range_[:, None], 1e-6, None)
    return batch_pn_cmd(N * cp.ones_like(Vc), Vc, los_dot, accel_limit)


# ---------------------------------------------------------------------------
# Batch phase-aware RHS for GPU Monte Carlo
# ---------------------------------------------------------------------------
def batch_rhs_gpu(t, y_batch, cfg_batch):
    """GPU RHS for Monte Carlo trials (simplified closed-loop).

    Parameters
    ----------
    t : float
    y_batch : (n, 14) CuPy array
    cfg_batch : dict
        Must contain: mass, area, ref_length, boundary_alt, thrust_val,
        target_pos (n,3), target_vel (n,3), N, accel_limit, guidance_law.
    Returns
    -------
    dy_batch : (n, 14) CuPy array
    """
    from .eom_cuda import batch_eom_rhs

    r = y_batch[:, 0:3]
    v = y_batch[:, 3:6]
    alt = cp.linalg.norm(r, axis=1) - R_EARTH

    # Guidance command
    target_pos = cfg_batch['target_pos']
    target_vel = cfg_batch['target_vel']
    N = float(cfg_batch.get('N', 4.0))
    accel_limit = float(cfg_batch.get('accel_limit', 150.0))
    guidance_law = cfg_batch.get('guidance_law', 'pn')

    los, range_, los_unit, rel_vel = batch_los_vec(r, v, target_pos, target_vel)
    Vc = -cp.sum(rel_vel * los_unit, axis=1)

    if guidance_law == 'pn':
        los_dot = (rel_vel - cp.sum(rel_vel * los_unit, axis=1)[:, None] * los_unit) / cp.clip(range_[:, None], 1e-6, None)
        accel_cmd = batch_pn_cmd(cp.full_like(Vc, N), Vc, los_dot, accel_limit)
    elif guidance_law == 'midcourse_pn':
        accel_cmd = batch_midcourse_pn(r, v, cp.concatenate([target_pos, target_vel], axis=1), N, accel_limit)
    elif guidance_law == 'zem':
        accel_cmd = batch_zem_cmd(target_pos - r, rel_vel, range_, N, 5.0, accel_limit)
    else:
        accel_cmd = cp.zeros((y_batch.shape[0], 3), dtype=cp.float64)

    # Phase gating: zero guidance above boundary_alt
    boundary_alt = float(cfg_batch.get('boundary_alt', 100e3))
    in_terminal = alt < boundary_alt
    if guidance_law == 'midcourse_pn':
        accel_cmd = accel_cmd  # active in midcourse
    else:
        accel_cmd = accel_cmd * in_terminal[:, None]

    # EOM RHS
    dy = batch_eom_rhs(t, y_batch, cfg_batch)
    # Add guidance acceleration to inertial velocity derivative
    m = cp.clip(y_batch[:, 13:14], 1e-6, None)
    dy[:, 3:6] = dy[:, 3:6] + accel_cmd / m
    return dy
