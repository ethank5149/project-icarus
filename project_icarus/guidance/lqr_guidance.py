"""Linear Quadratic Regulator (LQR) guidance for ICBM boost phase.

Implements the closed-loop guidance law described in
reference/improved-mathematical-analysis.md Section 3:

    Delta_u(t) = -K(t) Delta_x(t)

where K is the feedback gain from the Continuous-Time Algebraic
Riccati Equation and Delta_x is the tracking error relative to
a precalculated reference trajectory.
"""

import numpy as np
from scipy.linalg import solve_continuous_are


class LQRGuidance:
    """LQR feedback guidance for ICBM boost-phase trajectory tracking.

    Linearizes the missile dynamics about a reference trajectory and
    computes the optimal feedback gain K that minimizes a quadratic
    cost function balancing tracking error against control effort.
    """

    def __init__(
        self,
        reference_trajectory: np.ndarray,
        reference_times: np.ndarray,
        Q: np.ndarray = None,
        R: np.ndarray = None,
        mu: float = 3.986004418e14,
    ):
        """
        Parameters
        ----------
        reference_trajectory : ndarray, shape (N, 6)
            Reference state sequence [r, v] at each time step.
        reference_times : ndarray, shape (N,)
            Time values corresponding to each reference state.
        Q : ndarray, shape (6, 6), optional
            State cost matrix.  Higher values penalize tracking error
            more aggressively.  Defaults to diag([1e6, 1e6, 1e6, 1e4, 1e4, 1e4]).
        R : ndarray, shape (3, 3), optional
            Control cost matrix.  Higher values penalize actuator
            deflection more.  Defaults to diag([1e3, 1e3, 1e3]).
        mu : float
            Earth gravitational parameter (m^3/s^2).
        """
        self.ref_traj = np.asarray(reference_trajectory, dtype=float)
        self.ref_times = np.asarray(reference_times, dtype=float)
        self.mu = mu

        if Q is None:
            Q = np.diag([1e6, 1e6, 1e6, 1e4, 1e4, 1e4])
        if R is None:
            R = np.diag([1e3, 1e3, 1e3])

        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)

        # Pre-compute Riccati gain at each reference time step
        self._gains = None
        self._compute_gains()

    def _linearize(self, idx: int):
        """Linearize dynamics about reference state at index idx.

        Returns A (6x6) and B (6x3) Jacobians.
        """
        x_ref = self.ref_traj[idx]
        r_ref = x_ref[:3]
        v_ref = x_ref[3:6]
        m_ref = 208100.0  # nominal mass

        r_mag = np.linalg.norm(r_ref)
        r_mag = max(r_mag, 1e-6)

        # A matrix: d/dx of f(x, u) evaluated at reference state
        # State: [r, v],  dx/dt = [v, a]
        # a = -mu*r/|r|^3 + u/m  (simplified, no drag/aero for boost)

        # d/dr of gravitational acceleration
        dgrad_dr = np.zeros((3, 3))
        factor = self.mu / r_mag**3
        I3 = np.eye(3)
        dgrad_dr = -factor * I3 + 3.0 * factor * np.outer(r_ref, r_ref) / r_mag**2

        A = np.zeros((6, 6))
        A[0:3, 3:6] = I3
        A[3:6, 0:3] = dgrad_dr

        # B matrix: d/du of f(x, u) = (1/m) * I_3
        B = np.zeros((6, 3))
        B[3:6, :] = np.eye(3) / max(m_ref, 1e-6)

        return A, B

    def _compute_gains(self):
        """Pre-compute LQR feedback gains at each reference time step."""
        n = len(self.ref_traj)
        gains = np.zeros((n, 3, 6))

        for i in range(n):
            A, B = self._linearize(i)
            try:
                P = solve_continuous_are(A, B, self.Q, self.R)
                K = np.linalg.inv(self.R) @ B.T @ P
                gains[i] = K
            except np.linalg.LinAlgError:
                gains[i] = np.zeros((3, 6))

        self._gains = gains

    def compute_correction(self, state: np.ndarray, t: float) -> np.ndarray:
        """Compute LQR steering correction for the given state.

        Parameters
        ----------
        state : ndarray, shape (6,)
            Current [r, v] state vector.
        t : float
            Current time.

        Returns
        -------
        delta_u : ndarray, shape (3,)
            LQR steering correction (thrust gimbal adjustment).
        """
        state = np.asarray(state, dtype=float)

        # Find nearest reference time index
        idx = np.searchsorted(self.ref_times, t)
        idx = min(max(idx, 0), len(self.ref_traj) - 1)

        # Get reference state at this time
        x_ref = self.ref_traj[idx]

        # Tracking error
        delta_x = state - x_ref

        # Get LQR gain
        K = self._gains[idx]

        # Control law: delta_u = -K * delta_x
        delta_u = -K @ delta_x

        return delta_u

    def get_gain_at_time(self, t: float) -> np.ndarray:
        """Return the LQR gain matrix K at time t."""
        idx = np.searchsorted(self.ref_times, t)
        idx = min(max(idx, 0), len(self._gains) - 1)
        return self._gains[idx]