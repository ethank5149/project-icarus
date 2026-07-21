"""
Parallel Monte Carlo runner using joblib multiprocessing.

Preserves full CPU physics fidelity (same EOM, guidance, integrator) while
parallelizing independent engagement trials across host cores.
"""

from typing import Any, Dict, List, Optional
import numpy as np


def _run_single_trial(worker_idx, interceptor_cfg, guidance, target, scenario,
                      perturb, cfg, base_seed):
    """Run a single Monte Carlo trial in a worker process.

    Each worker gets a deterministic seed derived from the base seed and its
    index so results are reproducible.
    """
    from ..sim.runner import _integrate_trajectory
    rng = np.random.default_rng(base_seed + worker_idx)
    try:
        _, _, miss, kill = _integrate_trajectory(
            interceptor_cfg, guidance, target, scenario,
            perturb=perturb, cfg=cfg, rng=rng,
        )
        if not np.isfinite(miss):
            miss = float("nan")
            kill = False
        return miss, kill, perturb
    except Exception:
        return float("nan"), False, perturb


class ParallelMonteCarloRunner:
    """Run Monte Carlo engagement trials in parallel on CPU.

    Uses joblib multiprocessing to run independent trials across all available
    CPU cores.  Falls back to sequential execution if joblib is unavailable.
    """

    def __init__(self, n_jobs: int = -1, prefer: str = "processes"):
        self.n_jobs = n_jobs
        self.prefer = prefer
        try:
            import joblib
            self._joblib = joblib
            self._have_joblib = True
        except ImportError:
            self._have_joblib = False

    def run(self, interceptor, guidance, target, scenario,
            n_trials: int = 50, perturbations: Optional[Dict[str, float]] = None,
            cfg: Optional[Any] = None, base_seed: int = 42) -> Dict[str, Any]:
        """Run a batch of Monte Carlo trials.

        Parameters
        ----------
        interceptor : InterceptorConfig
        guidance : GuidanceLaw
        target : TargetScenario
        scenario : EngagementScenario
        n_trials : int
            Number of Monte Carlo trials.
        perturbations : dict, optional
            Perturbation parameters.
        cfg : SimConfig, optional
        base_seed : int
            Base RNG seed for reproducibility.

        Returns
        -------
        dict with keys 'miss_distances', 'kill_assessments', 'perturbations'.
        """
        from ..sim.runner import get_config, SimConfig
        cfg = cfg or get_config()
        if perturbations is None:
            perturbations = dict(cfg.perturbations)

        tasks = [
            (i, interceptor, guidance, target, scenario, perturbations, cfg, base_seed)
            for i in range(n_trials)
        ]

        if self._have_joblib and n_trials > 1:
            results = self._joblib.Parallel(
                n_jobs=self.n_jobs,
                prefer=self.prefer,
                return_as="list",
            )(
                self._joblib.delayed(_run_single_trial)(*task)
                for task in tasks
            )
        else:
            results = [_run_single_trial(*task) for task in tasks]

        misses = [float(r[0]) for r in results]
        kills = [bool(r[1]) for r in results]
        perts = [r[2] for r in results]

        finite = [m for m in misses if np.isfinite(m)]
        n_rejected = len(misses) - len(finite)

        return {
            "miss_distances": misses,
            "kill_assessments": kills,
            "perturbations": perts,
            "mean_miss": float(np.mean(finite)) if finite else 0.0,
            "std_miss": float(np.std(finite)) if finite else 0.0,
            "kill_probability": float(np.mean(kills)) if kills else 0.0,
            "n_rejected": n_rejected,
        }
