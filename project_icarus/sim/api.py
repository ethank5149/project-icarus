from __future__ import annotations

from typing import List, Optional, Dict, Any
import numpy as np

from .runner import EngagementRunner, EngagementResult
from .sweep import SweepRunner, SweepResult
from ..interceptors.config import InterceptorConfig, GuidanceConfig
from ..guidance.law import GuidanceLaw
from ..scenarios.target_factory import (
    TargetScenario,
    BallisticScenario,
    FOBSScenario,
    HGVScenario,
    SuppressedScenario,
    SwarmScenario,
)
from ..scenarios.scenario import EngagementScenario, SwarmScenario as SwarmScenarioClass


def run_engagement(
    interceptor: InterceptorConfig,
    guidance: GuidanceLaw,
    target: TargetScenario,
    scenario: EngagementScenario,
    n_trials: int = 50,
    perturbations: Optional[Dict[str, float]] = None,
    audit: bool = False,
) -> Any:
    """Run a single end-to-end engagement simulation.

    Parameters
    ----------
    interceptor : InterceptorConfig
        Interceptor system configuration.
    guidance : GuidanceLaw
        Guidance law configuration.
    target : TargetScenario
        Target trajectory scenario.
    scenario : EngagementScenario
        Engagement scenario definition.
    n_trials : int
        Number of Monte Carlo trials.
    perturbations : dict, optional
        Perturbation sigmas for Monte Carlo.
    audit : bool, default False
        When True, returns ``(EngagementResult, AuditReport)`` so every
        component (interceptor, guidance, seeker, C2, integrator, kill
        assessment) is auditable. When False (default), returns the
        ``EngagementResult`` exactly as before.

    Returns
    -------
    EngagementResult | tuple[EngagementResult, AuditReport]
    """
    runner = EngagementRunner(
        interceptor=interceptor,
        guidance=guidance,
        target=target,
        scenario=scenario,
    )
    result = runner.run(n_trials=n_trials, perturbations=perturbations)
    if audit:
        from ..reporting import EngagementAuditor
        report = EngagementAuditor().audit(
            interceptor=interceptor,
            guidance=guidance.config if hasattr(guidance, "config") else None,
            target=target,
            scenario=scenario,
            result=result,
        )
        return result, report
    return result


def run_sweep(
    interceptors: List[InterceptorConfig],
    targets: List[TargetScenario],
    scenarios: List[EngagementScenario],
    n_trials: int = 50,
    perturbations: Optional[Dict[str, float]] = None,
    parallel: bool = False,
    n_jobs: int = -1,
) -> SweepResult:
    """Run a batch sweep over interceptor/target/scenario combinations.

    Parameters
    ----------
    interceptors : list of InterceptorConfig
    targets : list of TargetScenario
    scenarios : list of EngagementScenario
    n_trials : int
        Monte Carlo trials per engagement.
    perturbations : dict, optional
    parallel : bool
        Whether to parallelize with joblib.
    n_jobs : int
        Number of parallel jobs.

    Returns
    -------
    SweepResult
    """
    runners = []
    for interceptor in interceptors:
        for target in targets:
            for scenario in scenarios:
                guidance = GuidanceLaw()
                runners.append(
                    EngagementRunner(
                        interceptor=interceptor,
                        guidance=guidance,
                        target=target,
                        scenario=scenario,
                    )
                )
    sweep = SweepRunner(runners=runners)
    if parallel:
        return sweep.run_parallel(n_trials=n_trials, perturbations=perturbations, n_jobs=n_jobs)
    return sweep.run(n_trials=n_trials, perturbations=perturbations)
