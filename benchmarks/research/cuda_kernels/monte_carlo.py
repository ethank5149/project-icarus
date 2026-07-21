"""
CUDA-accelerated Monte Carlo campaign runner.

Parallelizes independent engagement trials on GPU for massive speedup.
Each trial is a completely independent simulation, making this ideal for GPU.
"""

import numpy as np
from typing import Any, Optional, Dict, List, Tuple

try:
    import cupy as cp
    _HAVE_CUDA = True
except ImportError:
    _HAVE_CUDA = False


class CUDAMonteCarloRunner:
    """Run Monte Carlo engagement trials in parallel on CUDA GPU.

    This class provides GPU-accelerated Monte Carlo simulation by:
    1. Vectorizing trajectory propagation on GPU
    2. Running independent trials in parallel
    3. Using CuPy for GPU array operations

    Falls back to CPU if CUDA is not available.
    """

    def __init__(self, n_parallel: int = 256):
        self.n_parallel = n_parallel
        self._have_cuda = _HAVE_CUDA and cp.cuda.runtime.getDeviceCount() > 0

    def run_batch(self, initial_states: np.ndarray, cfg_batch: Dict[str, Any],
                  t_span: Tuple[float, float] = (0.0, 300.0), n_steps: int = 600,
                  use_j2: bool = True, use_high_order: bool = True,
                  use_j3: bool = True, use_j4: bool = True, max_degree: int = 10,
                  use_third_body: bool = False, use_tides: bool = False) -> Dict[str, np.ndarray]:
        """Run a batch of Monte Carlo trials.

        Parameters
        ----------
        initial_states : np.ndarray
            Initial interceptor states (n_trials, 14).
        cfg_batch : dict
            Per-trial config arrays of length n_trials. Must contain keys:
            mass, area, ref_length, boundary_alt, thrust_val, target_pos,
            target_vel, N, accel_limit, guidance_law.
        t_span : tuple
            (t0, tf).
        n_steps : int
            Number of RK4 steps.
        Returns
        -------
        dict with 'miss_distances', 'kill_assessments', 'final_states'
        """
        if not self._have_cuda:
            return self._run_cpu_fallback(initial_states, cfg_batch, t_span, n_steps,
                                          use_j2, use_high_order, use_j3, use_j4,
                                          max_degree, use_third_body, use_tides)

        return self._run_cuda(initial_states, cfg_batch, t_span, n_steps,
                              use_j2, use_high_order, use_j3, use_j4,
                              max_degree, use_third_body, use_tides)

    def _run_cuda(self, initial_states: np.ndarray, cfg_batch: Dict[str, Any],
                  t_span: Tuple[float, float], n_steps: int,
                  use_j2: bool = True, use_high_order: bool = True,
                  use_j3: bool = True, use_j4: bool = True, max_degree: int = 10,
                  use_third_body: bool = False, use_tides: bool = False) -> Dict[str, np.ndarray]:
        """CUDA implementation - vectorized trajectory propagation."""
        from .eom_cuda import rk4_batch

        y_final = rk4_batch(
            initial_states, t_span, n_steps, cfg_batch,
            use_j2, use_high_order, use_j3, use_j4, max_degree,
            use_third_body, use_tides,
        )
        r_final = cp.asnumpy(y_final[:, 0:3])
        target_pos = np.asarray(cfg_batch['target_pos'])
        miss = np.linalg.norm(r_final - target_pos, axis=1)
        kill = miss < 0.5
        return {
            'miss_distances': miss,
            'kill_assessments': kill,
            'final_states': cp.asnumpy(y_final),
        }

    def _run_cpu_fallback(self, initial_states: np.ndarray, cfg_batch: Dict[str, Any],
                          t_span: Tuple[float, float], n_steps: int,
                          use_j2: bool = True, use_high_order: bool = True,
                          use_j3: bool = True, use_j4: bool = True, max_degree: int = 10,
                          use_third_body: bool = False, use_tides: bool = False) -> Dict[str, np.ndarray]:
        """CPU fallback using NumPy vectorization."""
        R_EARTH = 6371e3
        MU = 3.986004418e14
        J2 = 1.08263e-3
        y = np.asarray(initial_states, dtype=float)
        n = y.shape[0]
        t0, tf = float(t_span[0]), float(t_span[1])
        dt = (tf - t0) / max(n_steps, 1)
        target_pos = np.asarray(cfg_batch['target_pos'])

        for _ in range(n_steps):
            r = y[:, 0:3]
            v = y[:, 3:6]
            q = y[:, 6:10]
            omega = y[:, 10:13]
            m = y[:, 13:14]
            r_mag = np.linalg.norm(r, axis=1, keepdims=True)
            alt = r_mag - R_EARTH
            v_mag = np.linalg.norm(v, axis=1, keepdims=True)
            rho = 1.225 * np.exp(-np.clip(alt, 0, None) / 8500.0)
            rho = np.where(alt < 100e3, rho, 0.0)
            q_dyn = 0.5 * rho * v_mag**2
            g0_vec = -MU / r_mag**2 * (r / np.clip(r_mag, 1e-6, None))
            z_comp = r[:, 2:3] / np.clip(r_mag, 1e-6, None)
            j2_corr = 1.5 * J2 * (R_EARTH / r_mag)**2 * np.hstack([
                (5*z_comp**2 - 1) * (r / np.clip(r_mag, 1e-6, None))[:, 0:1],
                (5*z_comp**2 - 1) * (r / np.clip(r_mag, 1e-6, None))[:, 1:2],
                (5*z_comp**2 - 3) * (r / np.clip(r_mag, 1e-6, None))[:, 2:3],
            ])
            g_inertial = g0_vec + j2_corr
            f_total_inertial = g_inertial
            dr_dt = v
            dv_dt = f_total_inertial / np.clip(m, 1e-6, None) + g_inertial
            dq_dt = 0.5 * np.hstack([
                q[:, 3:4]*omega[:, 0:1] + q[:, 0:1]*omega[:, 2:3] - q[:, 1:2]*omega[:, 1:2],
                q[:, 3:4]*omega[:, 1:2] + q[:, 1:2]*omega[:, 0:1] - q[:, 0:1]*omega[:, 2:3],
                q[:, 3:4]*omega[:, 2:3] + q[:, 2:3]*omega[:, 0:1] - q[:, 1:2]*omega[:, 0:1],
                -q[:, 0:1]*omega[:, 0:1] - q[:, 1:2]*omega[:, 1:2] - q[:, 2:3]*omega[:, 2:3],
            ])
            domega_dt = np.zeros_like(omega)
            dm_dt = np.zeros((n, 1))
            dy = np.hstack([dr_dt, dv_dt, dq_dt, domega_dt, dm_dt])
            y = y + dt * dy

        r_final = y[:, 0:3]
        miss = np.linalg.norm(r_final - target_pos, axis=1)
        kill = miss < 0.5
        return {
            'miss_distances': miss,
            'kill_assessments': kill,
            'final_states': y,
        }

    def is_available(self) -> bool:
        """Check if CUDA acceleration is available."""
        return self._have_cuda
