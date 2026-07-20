"""Phase 6.4 — National "Golden Dome" live dashboard (Panel + PyVista).

A small interactive dashboard that drives a national layered campaign end to end:

* builds the tiered architecture from ``reference/locations.yml``;
* generates a synthetic inbound raid of N threats aimed at defended targets;
* passes the raid through the distributed-C2 handoff (space warning -> ground
  with saturation drop);
* runs the ``BattleManager`` doctrine across all layers (or, with the
  *real engagements* toggle, precomputes true per-(threat, battery) miss
  distances in parallel via joblib);
* shows the PyVista ECEF coverage scene, national metrics, and the per-tier
  coverage table.

Run it with::

    panel serve project_icarus/c2/dashboard.py --show

The dashboard is intentionally light: the default (synthetic assess) evaluates in
well under a second so the controls are responsive; ticking *real engagements*
runs the genuine 6-DOF integrator (parallel across cores) for a higher-fidelity
score. Both paths feed the same HDF5/JSON transport in ``campaign.py`` /
``persistence.py``.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np

import panel as pn
import param


from .layers import (
    build_architecture_from_locations,
    run_layered_campaign,
    distributed_handoff,
    DistributedC2Config,
    SpaceSensor,
    architecture_summary,
    national_metrics,
)
from .battle_manager import ThreatTrack, BattleManagerConfig
from .visualization import build_national_scene, coverage_summary_table


# --- site database helpers ---------------------------------------------------
def _sites(designation: str):
    from reference.locations import locations_by_designation, coordinates_to_ecef
    groups = locations_by_designation()
    return [(rec["name"], np.asarray(coordinates_to_ecef(rec), dtype=float))
            for rec in groups.get(designation, [])]


def _raid(n: int, defended_points: List[np.ndarray], seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        aim = defended_points[i % len(defended_points)]
        # jitter the aim point so the raid is not all on one spot
        jit = rng.normal(0, 2.0e3, size=3)
        out.append(ThreatTrack(threat_id=i, target=None,
                               aim_point=np.asarray(aim, float) + jit))
    return out


def _scene_png(architecture, defended, threats, earth_radius=6.371e6) -> Optional[bytes]:
    """Render the PyVista scene offscreen to PNG bytes for Panel."""
    try:
        import pyvista as pv
    except ImportError:
        return None
    pv.off_screen = True
    scene = build_national_scene(architecture, defended, threats, earth_radius)
    plotter = pv.Plotter(off_screen=True, window_size=(900, 600))
    plotter.add_mesh(scene["earth"], color="navy", opacity=0.35)
    for key in scene.keys():
        if key.endswith("_bases"):
            kind = key.split("_")[0]
            color = {
                "boost": "red", "upper": "orange", "mid": "blue", "lower": "green",
            }.get(kind, "gray")
            plotter.add_points(scene[key], color=color, point_size=12,
                               render_points_as_spheres=True)
    for key in ("defended", "threats"):
        if key in scene:
            plotter.add_points(scene[key],
                               color="cyan" if key == "defended" else "magenta",
                               point_size=10, render_points_as_spheres=True)
    plotter.add_axes()
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        plotter.show(screenshot=path)
        with open(path, "rb") as fh:
            data = fh.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
        plotter.close()
    return data


class NationalDashboard(param.Parameterized):
    """Param-backed panel for the Phase-6 national campaign."""

    n_threats = param.Integer(default=8, bounds=(1, 200), doc="Inbound raid size")
    magazine_per_base = param.Integer(default=4, bounds=(1, 50))
    bandwidth = param.Number(default=5.0, bounds=(0.0, 100.0),
                             doc="Space-sensor track bandwidth (tracks/s)")
    p_detect = param.Number(default=0.95, bounds=(0.0, 1.0))
    raid_window_s = param.Number(default=5.0, bounds=(0.0, 120.0),
                                 doc="Raid arrival spread (s)")
    real_engagements = param.Boolean(default=False,
                                     doc="Precompute true 6-DOF miss distances")
    parallel = param.Boolean(default=True)
    run = param.Action(doc="Run the campaign")
    reset = param.Action(doc="Rebuild architecture / clear")

    def __init__(self, **params):
        super().__init__(**params)
        self.architecture = build_architecture_from_locations(
            magazine_per_base=self.magazine_per_base)
        self.defended = [loc for _, loc in _sites("defended-target")]
        self._metrics = {}
        self._c2 = {}
        self._scene_png = None
        self._coverage = coverage_summary_table(self.architecture)
        self.run = self._do_run
        self.reset = self._do_reset

    # --- actions -------------------------------------------------------------
    def _do_run(self, *events):
        arch = self.architecture
        raid = _raid(self.n_threats, self.defended, seed=0)
        c2 = DistributedC2Config()
        c2.space.bandwidth_tracks_per_s = self.bandwidth
        c2.space.p_detect = self.p_detect
        c2.raid_arrival_window_s = self.raid_window_s

        if self.real_engagements:
            def scenario_builder(track, battery):
                from ..scenarios.scenario import EngagementScenario
                return EngagementScenario(
                    interceptor_launch_site=np.asarray(battery.location, float),
                    target_launch_site=np.asarray([0.0, 0.0, 0.0]),
                    engagement_end=12.0,
                )
            res = run_layered_campaign(
                raid, arch, scenario_builder=scenario_builder, n_trials=1,
                cfg=BattleManagerConfig(), parallel=self.parallel, n_jobs=-1,
                backend="joblib",
            )
        else:
            # Lightweight: perfect interceptors (miss = 0) so the run is instant;
            # C2 saturation still drops tracks realistically.
            def assess(t, bi):
                return 0.1
            res = run_layered_campaign(raid, arch, assess, c2=c2)

        self._metrics = national_metrics(res)
        self._c2 = res.get("c2_diag", {})
        threats = [np.asarray(t.aim_point, float) for t in raid]
        self._scene_png = _scene_png(arch, self.defended, threats)
        self._coverage = coverage_summary_table(arch)
        self.param.trigger("run")

    def _do_reset(self, *events):
        self.architecture = build_architecture_from_locations(
            magazine_per_base=self.magazine_per_base)
        self._metrics = {}
        self._c2 = {}
        self._scene_png = _scene_png(self.architecture, self.defended,
                                     [np.zeros(3) for _ in range(self.n_threats)])
        self._coverage = coverage_summary_table(self.architecture)
        self.param.trigger("reset")

    # --- view ----------------------------------------------------------------
    def metrics_pane(self):
        if not self._metrics:
            return pn.pane.Markdown("Run a campaign to see national metrics.")
        m = self._metrics
        c2 = self._c2
        rows = [
            ("Inbound threats", m["n_inbound"]),
            ("Tasked (cued)", m["n_tasked"]),
            ("Dropped — C2 saturation", m["n_dropped_c2_saturation"]),
            ("Defeated", m["n_defeated"]),
            ("Leaked", m["n_leakage"]),
            ("  ↳ cued but missed", m["n_cued_but_missed"]),
            ("Leakage fraction", f"{m['leakage_fraction']:.2%}"),
            ("Kill probability", f"{m['kill_probability']:.2%}"),
            ("Shots fired", m["shots_fired"]),
            ("Mean battery utilization", f"{m['mean_battery_utilization']:.2%}"),
            ("Mean cue latency (s)", f"{m['mean_cue_latency_s']:.1f}"),
            ("Space bandwidth (tracks/s)", c2.get("space_bandwidth_tracks_per_s", "-")),
        ]
        md = "### National metrics\n\n" + "\n".join(
            f"- **{k}**: {v}" for k, v in rows)
        return pn.pane.Markdown(md)

    def coverage_pane(self):
        rows = [("<th>#</th><th>Kind</th><th>Interceptor</th>"
                 "<th>Bases</th><th>Mag/base</th><th>Total</th>"
                 "<th>Spread km</th><th>C2 lat s</th>")]
        for i, r in enumerate(self._coverage):
            rows.append(
                f"<tr><td>{i}</td><td>{r['kind']}</td><td>{r['interceptor']}</td>"
                f"<td>{r['n_bases']}</td><td>{r['magazine_per_base']}</td>"
                f"<td>{r['total_magazine']}</td><td>{r['mean_base_spread_km']:.0f}</td>"
                f"<td>{r['c2_latency_s']}</td></tr>")
        html = ("### Per-tier coverage\n<table border='1' cellspacing='0'>" +
                "".join(rows) + "</table>")
        return pn.pane.HTML(html, width=640)

    def scene_pane(self):
        if self._scene_png is None:
            return pn.pane.Markdown("Select **Run** to render the coverage map.")
        return pn.pane.PNG(self._scene_png, width=900)

    def view(self):
        controls = pn.Column(
            "## National 'Golden Dome' dashboard",
            self.param.n_threats,
            self.param.magazine_per_base,
            self.param.bandwidth,
            self.param.p_detect,
            self.param.raid_window_s,
            self.param.real_engagements,
            self.param.parallel,
            pn.Row(pn.widgets.Button(label="Run campaign",
                                     button_type="primary",
                                     on_click=self._do_run),
                   pn.widgets.Button(label="Reset map",
                                     on_click=self._do_reset)),
        )
        right = pn.Column(self.metrics_pane, self.coverage_pane)
        return pn.Row(controls, pn.Column(self.scene_pane, right),
                      sizing_mode="stretch_width")


def make_dashboard():
    """Return a fresh dashboard view (used by ``panel serve``)."""
    dash = NationalDashboard()
    dash._do_reset()
    return dash.view()


# When served directly, expose the app at module level.
try:
    app = make_dashboard()
except Exception:  # pragma: no cover - keep import safe
    app = pn.pane.Markdown("Dashboard failed to initialize; check site DB / pyvista.")


if __name__ == "__main__":
    pn.serve(make_dashboard, show=True, port=5006)
