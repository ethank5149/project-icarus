"""Phase 7 VV&A — engagement audit & report generation.

Additive verification/validation-and-audit tooling. An :class:`EngagementAuditor`
runs (or consumes) an engagement and records a structured, component-by-component
event log covering the interceptor, guidance law, seeker/discriminator, target,
C2/engagement scenario, the integrator (nominal + Monte-Carlo), and the final
kill assessment. :func:`render_report` emits the log as Markdown or JSON.

The module is fully additive: it never mutates the simulation engine. The
``audit=True`` flag on :func:`project_icarus.sim.api.run_engagement` returns
``(EngagementResult, AuditReport)`` instead of just the result, so every
component at every stage is auditable without changing default behavior.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..interceptors.config import InterceptorConfig, GuidanceConfig
from ..scenarios.target_factory import TargetScenario
from ..scenarios.scenario import EngagementScenario
from ..guidance.law import GuidanceLaw
from ..sim.runner import EngagementResult

logger = logging.getLogger("project_icarus.reporting")


# ---------------------------------------------------------------------------
# Audit record
# ---------------------------------------------------------------------------

@dataclass
class AuditEvent:
    """A single audited step / component observation."""
    stage: str
    detail: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"stage": self.stage, "detail": self.detail, "data": self._jsonable(self.data)}

    @staticmethod
    def _jsonable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: AuditEvent._jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [AuditEvent._jsonable(v) for v in obj]
        return obj


@dataclass
class AuditReport:
    """Structured, component-by-component audit of a single engagement."""
    interceptor_name: str = ""
    target_name: str = ""
    events: List[AuditEvent] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def record(self, stage: str, detail: str, **data: Any) -> None:
        ev = AuditEvent(stage=stage, detail=detail, data=data)
        self.events.append(ev)
        logger.info("[audit:%s] %s %s", stage, detail,
                    {k: AuditEvent._jsonable(v) for k, v in data.items()})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interceptor_name": self.interceptor_name,
            "target_name": self.target_name,
            "summary": AuditEvent._jsonable(self.summary),
            "events": [e.to_dict() for e in self.events],
        }


# ---------------------------------------------------------------------------
# Auditor
# ---------------------------------------------------------------------------

class EngagementAuditor:
    """Audit an engagement, component by component, with a structured event log."""

    def __init__(self, log: bool = True):
        self.log = log

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _summarize_interceptor(cfg: InterceptorConfig) -> Dict[str, Any]:
        return {
            "name": cfg.name,
            "mass_kg": cfg.mass,
            "area_m2": cfg.area,
            "kill_mechanism": cfg.kill_mechanism,
            "kill_radius_m": cfg.kill_radius,
            "accel_limit_m_s2": cfg.accel_limit,
            "n_stages": len(cfg.stages) if cfg.stages else (1 if cfg.thrust_profile else 0),
            "peak_thrust_N": cfg.peak_thrust,
        }

    @staticmethod
    def _summarize_guidance(cfg: GuidanceConfig) -> Dict[str, Any]:
        return {
            "midcourse_n": cfg.midcourse_n,
            "midcourse_accel_limit": cfg.midcourse_accel_limit,
            "terminal_n": cfg.terminal_n,
            "terminal_accel_limit": cfg.terminal_accel_limit,
            "terminal_guidance_law": cfg.terminal_guidance_law,
            "seeker_mode": cfg.seeker_mode,
            "seeker_fov_deg": cfg.seeker_fov_deg,
            "ukf_enabled": cfg.ukf_enabled,
        }

    @staticmethod
    def _summarize_target(target: TargetScenario) -> Dict[str, Any]:
        scenario_type = type(target).__name__
        out: Dict[str, Any] = {"type": scenario_type}
        if hasattr(target, "decoys"):
            out["n_decoys"] = len(getattr(target, "decoys", []) or [])
        return out

    # -- main entry ---------------------------------------------------------
    def audit(
        self,
        interceptor: InterceptorConfig,
        guidance: GuidanceConfig,
        target: TargetScenario,
        scenario: EngagementScenario,
        result: EngagementResult,
        guidance_law: Optional[GuidanceConfig] = None,
    ) -> AuditReport:
        """Build an :class:`AuditReport` from a completed engagement result.

        Parameters
        ----------
        interceptor, guidance, target, scenario
            The configuration objects used for the engagement (for traceability).
        result
            The :class:`EngagementResult` returned by ``run_engagement``.
        guidance_law
            Optional :class:`GuidanceConfig` (e.g. the law actually flown). When
            omitted, ``guidance`` is used.
        """
        report = AuditReport(
            interceptor_name=interceptor.name,
            target_name=getattr(target, "name", type(target).__name__),
        )
        if self.log:
            logger.info("Starting engagement audit: %s vs %s",
                        interceptor.name, report.target_name)

        # 1) Interceptor component
        icfg = self._summarize_interceptor(interceptor)
        report.record("interceptor", "configuration", **icfg)

        # 2) Guidance component
        gcfg = self._summarize_guidance(guidance_law or guidance)
        report.record("guidance", "configuration", **gcfg)

        # 3) Target component
        tcfg = self._summarize_target(target)
        report.record("target", "configuration", **tcfg)

        # 4) C2 / engagement scenario (aim point, end time, sensor noise, etc.)
        report.record(
            "c2",
            "engagement_scenario",
            engagement_end_s=scenario.engagement_end,
            sensor_noise=scenario.sensor_noise,
            threat_axis=scenario.threat_axis,
            interceptor_launch_site=scenario.interceptor_launch_site,
            target_launch_site=scenario.target_launch_site,
        )

        # 5) Integrator / nominal trajectory
        nominal = getattr(result, "nominal_trajectory", None)
        nominal_target = getattr(result, "nominal_target_trajectory", None)
        report.record(
            "integrator",
            "nominal_trajectory",
            n_points=len(nominal) if nominal is not None else 0,
            target_n_points=len(nominal_target) if nominal_target is not None else 0,
        )

        # 6) Monte-Carlo statistics (if present)
        mc = getattr(result, "monte_carlo", None)
        if mc is not None and getattr(mc, "miss_distances", None):
            report.record(
                "integrator",
                "monte_carlo",
                n_trials=len(mc.miss_distances),
                mean_miss_m=float(np.mean(mc.miss_distances)),
                std_miss_m=float(np.std(mc.miss_distances)),
                min_miss_m=float(np.min(mc.miss_distances)),
                max_miss_m=float(np.max(mc.miss_distances)),
            )

        # 7) Kill assessment / end-game
        mc_kill_prob = getattr(mc, "kill_probability", None) if mc is not None else None
        report.record(
            "kill_assessment",
            "endgame",
            miss_distance_m=result.miss_distance,
            kill_assessment=bool(result.kill_assessment),
            kill_probability=mc_kill_prob,
        )

        report.summary = {
            "interceptor": interceptor.name,
            "target": report.target_name,
            "miss_distance_m": result.miss_distance,
            "kill_assessment": bool(result.kill_assessment),
            "kill_probability": mc_kill_prob,
        }
        if self.log:
            logger.info(
                "Engagement audit complete: miss=%.2f m, kill=%s",
                result.miss_distance, result.kill_assessment,
            )
        return report


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_report(report: AuditReport, fmt: str = "markdown") -> str:
    """Render an :class:`AuditReport` to a string.

    Parameters
    ----------
    report : AuditReport
    fmt : str
        ``"markdown"`` or ``"json"``.
    """
    if fmt == "json":
        return json.dumps(report.to_dict(), indent=2, default=str)
    if fmt != "markdown":
        raise ValueError(f"Unknown report format: {fmt!r} (use 'markdown' or 'json')")

    lines: List[str] = []
    lines.append("# Engagement Audit Report")
    lines.append("")
    lines.append(f"- **Interceptor:** {report.interceptor_name}")
    lines.append(f"- **Target:** {report.target_name}")
    s = report.summary
    if s:
        lines.append(f"- **Miss distance:** {s.get('miss_distance_m')} m")
        lines.append(f"- **Kill assessment:** {s.get('kill_assessment')}")
        if s.get("kill_probability") is not None:
            lines.append(f"- **Kill probability (MC):** {s.get('kill_probability')}")
    lines.append("")

    # Group events by stage for readability.
    by_stage: Dict[str, List[AuditEvent]] = {}
    for ev in report.events:
        by_stage.setdefault(ev.stage, []).append(ev)

    for stage, evs in by_stage.items():
        lines.append(f"## {stage.replace('_', ' ').title()}")
        lines.append("")
        for ev in evs:
            if ev.data:
                lines.append(f"- **{ev.detail}**")
                for k, v in ev.data.items():
                    v = AuditEvent._jsonable(v)
                    if isinstance(v, (list, dict)):
                        v = json.dumps(v)
                    lines.append(f"  - {k}: {v}")
            else:
                lines.append(f"- {ev.detail}")
        lines.append("")
    return "\n".join(lines)
