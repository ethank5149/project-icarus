from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np

from .runner import EngagementRunner, EngagementResult


@dataclass
class SweepResult:
    results: List[EngagementResult]
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError:
            print("pandas not available")
            return None
        rows = []
        for r in self.results:
            rows.append({
                "miss_distance": r.miss_distance,
                "kill_assessment": r.kill_assessment,
                "kill_probability": r.monte_carlo.kill_probability if r.monte_carlo else None,
                "mean_miss": r.monte_carlo.mean_miss if r.monte_carlo else None,
                "std_miss": r.monte_carlo.std_miss if r.monte_carlo else None,
                "n_trials": r.metadata.get("n_trials", 0),
            })
        return pd.DataFrame(rows)


class SweepRunner:
    def __init__(self, runners: List[EngagementRunner]):
        self.runners = runners

    def run(self, n_trials: int = 50, perturbations: Optional[Dict[str, float]] = None, n_jobs: int = 1) -> SweepResult:
        results = []
        for runner in self.runners:
            result = runner.run(n_trials=n_trials, perturbations=perturbations)
            results.append(result)
        return SweepResult(results=results)

    def run_parallel(self, n_trials: int = 50, perturbations: Optional[Dict[str, float]] = None, n_jobs: int = -1) -> SweepResult:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            print("joblib not available, running serially")
            return self.run(n_trials=n_trials, perturbations=perturbations)
        results = Parallel(n_jobs=n_jobs)(
            delayed(lambda r: r.run(n_trials=n_trials, perturbations=perturbations))(runner)
            for runner in self.runners
        )
        return SweepResult(results=results)
