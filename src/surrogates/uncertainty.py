import numpy as np
from scipy.linalg import cholesky


class UncertaintyPropagator:
    """
    Propagate aero coefficient uncertainties through 6-DOF EOM to miss-distance statistics.
    Uses first-order error propagation for efficiency.
    """

    def __init__(self, n_samples=200, seed=42):
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def propagate(self, nominal_state, sigma_cd, sigma_cy, sigma_cm, eom_func):
        """
        Monte Carlo propagation of coefficient uncertainties.

        Parameters
        ----------
        nominal_state : array
            Nominal final interceptor state.
        sigma_cd, sigma_cy, sigma_cm : float
            1-sigma coefficient uncertainties.
        eom_func : callable
            Function(cd, cy, cm) -> miss_distance.

        Returns
        -------
        mean_miss : float
        std_miss : float
        """
        samples = self.rng.normal(
            loc=[0.0, 0.0, 0.0],
            scale=[sigma_cd, sigma_cy, sigma_cm],
            size=(self.n_samples, 3),
        )
        misses = np.array([eom_func(s[0], s[1], s[2]) for s in samples])
        return float(np.mean(misses)), float(np.std(misses))

    def elementary_effects(self, nominal_state, sigma_cd, sigma_cy, sigma_cm, eom_func, n_samples=1000):
        """
        Cheap sensitivity indices via elementary effects.
        Returns relative sensitivity of miss distance to each coefficient.
        """
        base_miss, _ = self.propagate(nominal_state, sigma_cd, sigma_cy, sigma_cm, eom_func)
        factors = [1.0, 1.05, 0.95]
        sensitivities = []
        for sigma, factor in zip([sigma_cd, sigma_cy, sigma_cm], factors):
            scaled = sigma * factor
            if factor > 1:
                m, _ = self.propagate(nominal_state, scaled, sigma_cy, sigma_cm, eom_func)
            else:
                m, _ = self.propagate(nominal_state, sigma_cd, sigma_cy, scaled, eom_func)
            sensitivities.append(abs(m - base_miss) / max(sigma, 1e-12))
        total = sum(sensitivities) + 1e-12
        return [s / total for s in sensitivities]
