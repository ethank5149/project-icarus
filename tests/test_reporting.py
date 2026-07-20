import json

import numpy as np

from project_icarus.scenarios.presets import build_interceptor_config, target_preset
from project_icarus.guidance.law import GuidanceLaw
from project_icarus.sim.api import run_engagement
from project_icarus.interceptors.config import InterceptorConfig, GuidanceConfig
from project_icarus.scenarios.scenario import EngagementScenario
from project_icarus.scenarios.target_factory import BallisticScenario, R_EARTH
from project_icarus.reporting import (
    EngagementAuditor,
    AuditReport,
    AuditEvent,
    render_report,
)


def _fast_interceptor():
    # Lightweight interceptor launched from altitude so the engagement avoids the
    # surface-skimming regime (which is numerically stiff/slow). Mirrors the
    # existing TestEngagementRunner smoke test to keep audit tests fast.
    icfg, gcfg = build_interceptor_config("arrow3")
    icfg.mass = 1000.0
    return icfg, gcfg


def _fast_engagement(audit=False):
    icfg, gcfg = _fast_interceptor()
    target = BallisticScenario(r0=np.array([R_EARTH, 0.0, 0.0]),
                               v0=np.array([0.0, 1000.0, 0.0]))
    scenario = EngagementScenario(
        interceptor_launch_site=np.array([R_EARTH + 100e3, 0.0, 0.0]),
        engagement_end=60.0,
    )
    law = GuidanceLaw()
    return run_engagement(icfg, law, target, scenario, n_trials=2, audit=audit)


class TestReporting:
    def test_audit_returns_result_and_report(self):
        out = _fast_engagement(audit=True)
        assert isinstance(out, tuple) and len(out) == 2
        result, report = out
        assert isinstance(report, AuditReport)
        # Component-by-component traces are present.
        stages = {e.stage for e in report.events}
        assert {"interceptor", "guidance", "target", "c2",
                "integrator", "kill_assessment"} <= stages

    def test_audit_false_is_backward_compatible(self):
        result = _fast_engagement(audit=False)
        # Without audit, the plain EngagementResult is returned.
        assert not isinstance(result, tuple)
        assert hasattr(result, "miss_distance")

    def test_audit_records_events_for_synthetic_engagement(self):
        icfg, gcfg = _fast_interceptor()
        target = BallisticScenario(r0=np.array([R_EARTH, 0.0, 0.0]),
                                   v0=np.array([0.0, 1000.0, 0.0]))
        scenario = EngagementScenario(
            interceptor_launch_site=np.array([R_EARTH + 100e3, 0.0, 0.0]),
            engagement_end=60.0,
        )
        law = GuidanceLaw()
        result = run_engagement(icfg, law, target, scenario, n_trials=2)
        report = EngagementAuditor().audit(
            interceptor=icfg, guidance=gcfg, target=target,
            scenario=scenario, result=result,
        )
        kinds = [e.stage for e in report.events]
        assert "interceptor" in kinds and "kill_assessment" in kinds
        # Miss distance is captured in the endgame event.
        endgame = [e for e in report.events if e.stage == "kill_assessment"][0]
        assert "miss_distance_m" in endgame.data

    def test_render_markdown_and_json(self):
        _, report = _fast_engagement(audit=True)
        md = render_report(report, "markdown")
        assert md.startswith("# Engagement Audit Report")
        assert "## Interceptor" in md
        js = render_report(report, "json")
        parsed = json.loads(js)
        assert parsed["interceptor_name"] == report.interceptor_name
        assert len(parsed["events"]) == len(report.events)

    def test_render_unknown_format_raises(self):
        report = AuditReport()
        try:
            render_report(report, "pdf")
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_audit_event_jsonable(self):
        ev = AuditEvent(stage="x", detail="d", data={"arr": np.array([1.0, 2.0])})
        d = ev.to_dict()
        assert d["data"]["arr"] == [1.0, 2.0]
