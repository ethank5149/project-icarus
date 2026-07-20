import numpy as np
import pytest

from project_icarus.c2 import (
    build_architecture_from_locations,
    run_layered_campaign,
    run_discrete_event,
    distributed_handoff,
    DistributedC2Config,
    SpaceSensor,
    Tier,
    Layer,
    DefenseArchitecture,
    ThreatTrack,
    C2Scenario,
    BattleManager,
    BattleManagerConfig,
    Battery,
    architecture_summary,
    national_metrics,
)
from project_icarus.c2.visualization import build_national_scene, coverage_summary_table
from project_icarus.scenarios.presets import build_interceptor_config

pytest.importorskip("pyvista")


def _raid(n, batteries, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        loc = np.asarray(batteries[i % len(batteries)].location, float)
        aim = loc + rng.normal(0, 1e3, size=3)
        out.append(ThreatTrack(threat_id=i, target=None, aim_point=aim, priority=1.0))
    return out


class TestLayers:
    def test_architecture_from_locations(self):
        arch = build_architecture_from_locations(magazine_per_base=4, salvo_size=1)
        # 11 interceptor-launch-site records in the DB (7 legacy + Patriot/THAAD).
        assert len(arch.batteries) == 11
        assert arch.total_magazine == 44
        # All batteries carry a real interceptor config.
        for bat in arch.batteries:
            assert bat.interceptor_config is not None
            assert bat.magazine == 4

    def test_architecture_selects_patriot_thaad_by_site(self):
        # The new Patriot/THAAD deploy sites carry an ``interceptor`` key in
        # reference/locations.yml; the architecture must build dedicated tiers
        # for them (lower->patriot, mid->thaad) rather than the kind defaults.
        arch = build_architecture_from_locations()
        by_kind_interceptor = {
            (t.kind, t.interceptor_name): t for ly in arch.layers for t in ly.tiers
        }
        assert by_kind_interceptor[("lower", "patriot")].bases
        assert by_kind_interceptor[("mid", "thaad")].bases
        # Patriot/THAAD batteries resolve to their own presets.
        patriot_bats = [
            b for b in arch.batteries
            if b.interceptor_config.name.startswith("Patriot")
        ]
        thaad_bats = [
            b for b in arch.batteries
            if b.interceptor_config.name.startswith("THAAD")
        ]
        assert len(patriot_bats) == 2
        assert len(thaad_bats) == 2
        assert all(b.interceptor_config.kill_mechanism == "hit_to_kill"
                   for b in patriot_bats + thaad_bats)

    def test_tier_kind_validation(self):
        cfg, g = build_interceptor_config("arrow3")
        with np.testing.assert_raises(ValueError):
            Tier(kind="sideways", interceptor_name="arrow3", bases=[np.zeros(3)])

    def test_tier_expands_to_batteries(self):
        cfg, g = build_interceptor_config("gmd")
        tier = Tier(kind="boost", interceptor_name="gmd",
                    bases=[np.zeros(3), np.array([1.0, 0, 0])],
                    magazine_per_base=5, salvo_size=2, label="gmd")
        bats = tier.to_batteries()
        assert len(bats) == 2
        assert all(b.magazine == 5 for b in bats)
        assert all(b.salvo_size == 2 for b in bats)

    def test_layer_aggregates_tiers(self):
        cfg, g = build_interceptor_config("tamir")
        ly = Layer("point").add(Tier(kind="lower", interceptor_name="tamir",
                                     bases=[np.zeros(3)], magazine_per_base=3))
        assert ly.total_magazine == 3
        assert len(ly.batteries) == 1


class TestDistributedC2:
    def test_handoff_passes_all_under_hi_bandwidth(self):
        np.random.seed(0)
        threats = [ThreatTrack(threat_id=i, target=None, aim_point=np.zeros(3))
                   for i in range(10)]
        c2 = DistributedC2Config()
        c2.space.bandwidth_tracks_per_s = 100.0
        c2.space.p_detect = 1.0
        tasked, diag = distributed_handoff(threats, c2)
        assert diag["n_dropped"] == 0
        assert diag["n_tasked"] == 10

    def test_handoff_saturates_at_zero_bandwidth(self):
        np.random.seed(0)
        threats = [ThreatTrack(threat_id=i, target=None, aim_point=np.zeros(3))
                   for i in range(10)]
        c2 = DistributedC2Config()
        c2.space.bandwidth_tracks_per_s = 0.0
        c2.space.p_detect = 1.0
        tasked, diag = distributed_handoff(threats, c2)
        assert diag["n_tasked"] == 0
        assert diag["n_dropped"] == 10

    def test_handoff_spread_arrival_cues_fraction(self):
        np.random.seed(2)
        threats = [ThreatTrack(threat_id=i, target=None, aim_point=np.zeros(3))
                   for i in range(20)]

        def tc(t):
            return 0.5 * (t.threat_id % 10)
        c2 = DistributedC2Config()
        c2.space.bandwidth_tracks_per_s = 2.0
        c2.space.p_detect = 1.0
        c2.space.warning_latency_s = 0.0
        c2.ground_latency_s = 0.0
        tasked, diag = distributed_handoff(threats, c2, t_contact=tc)
        # Over a 5 s window at 2 tracks/s ~= 10 cues; saturation drops the rest.
        assert 0 < diag["n_tasked"] < 20
        assert diag["n_tasked"] + diag["n_dropped"] == 20

    def test_cue_latency_sets_skip_rounds(self):
        # A non-zero cue latency must delay the first engagement rounds.
        cfg, g = build_interceptor_config("arrow3")
        bats = [Battery("A", cfg, g, location=np.zeros(3), magazine=3, salvo_size=1)]
        threats = [ThreatTrack(threat_id=0, target=None, aim_point=np.zeros(3))]
        # With a huge cue latency and few rounds, nothing is engaged.
        c2 = DistributedC2Config()
        c2.space.warning_latency_s = 100.0  # > max_rounds*interval
        c2.space.p_detect = 1.0
        tasked, _ = distributed_handoff(threats, c2)
        bm = BattleManager(tasked, bats, cfg=BattleManagerConfig(max_rounds=3))
        # assess would always kill, but cue latency skips all 3 rounds.
        res = bm.run(lambda t, bi: 0.1)
        assert res.n_defeated == 0


class TestLayeredCampaign:
    def test_layered_campaign_engages_raid(self):
        np.random.seed(3)
        arch = build_architecture_from_locations(magazine_per_base=6, salvo_size=1)
        bats = arch.batteries
        raid = _raid(10, bats, seed=3)
        c2 = DistributedC2Config()
        c2.space.bandwidth_tracks_per_s = 100.0
        c2.space.p_detect = 1.0
        c2.raid_arrival_window_s = 5.0

        def assess(t, bi):
            return 0.1  # perfect interceptors

        r = run_layered_campaign(raid, arch, assess, c2=c2)
        # Every threat is cued (hi bandwidth) and defeated (perfect interceptors).
        assert r["c2_diag"]["n_tasked"] == 10
        assert r["battle"].n_defeated == 10
        assert r["battle"].n_leakage == 0

    def test_layered_campaign_saturation_drops_threats(self):
        np.random.seed(4)
        arch = build_architecture_from_locations(magazine_per_base=6, salvo_size=1)
        bats = arch.batteries
        raid = _raid(20, bats, seed=4)
        c2 = DistributedC2Config()
        c2.space.bandwidth_tracks_per_s = 1.0  # heavy saturation
        c2.space.p_detect = 1.0
        c2.raid_arrival_window_s = 2.0

        def assess(t, bi):
            return 0.1

        r = run_layered_campaign(raid, arch, assess, c2=c2)
        # Not all threats reached a battery -> some leakage from C2 saturation.
        assert r["c2_diag"]["n_tasked"] < 20
        assert r["battle"].n_leakage > 0
        # Leakage cannot exceed the number that were dropped+cued-but-missed.
        assert r["battle"].n_leakage <= 20

    def test_layered_campaign_imperfect_interceptors_leak(self):
        np.random.seed(5)
        arch = build_architecture_from_locations(magazine_per_base=10, salvo_size=2)
        bats = arch.batteries
        raid = _raid(6, bats, seed=5)
        c2 = DistributedC2Config()
        c2.space.bandwidth_tracks_per_s = 100.0
        c2.space.p_detect = 1.0
        c2.raid_arrival_window_s = 4.0

        # Interceptors that always miss (miss >> kill_radius) -> full leakage.
        def assess(t, bi):
            return 50.0

        r = run_layered_campaign(raid, arch, assess, c2=c2)
        assert r["c2_diag"]["n_tasked"] == 6
        assert r["battle"].n_defeated == 0
        assert r["battle"].n_leakage == 6


def _raid_with_targets(n, batteries, seed=0):
    """Build threats that carry a real target scenario (needed for run_engagement)."""
    from project_icarus.scenarios.target_factory import BallisticScenario
    from project_icarus.scenarios.presets import geodetic_to_ecef

    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        loc = np.asarray(batteries[i % len(batteries)].location, float)
        aim = loc + rng.normal(0, 1e3, size=3)
        target = BallisticScenario(r0=aim, v0=np.array([0.0, 800.0, 0.0]))
        tt = ThreatTrack(threat_id=i, target=target, aim_point=aim, priority=1.0)
        out.append(tt)
    return out


class TestLayeredParallelAndMetrics:
    def test_run_campaign_parallel_matches_serial(self):
        # The parallel pairwise backend must fan out / gather identically to the
        # serial loop without invoking the (expensive) 6-DOF integrator per
        # pair. We stub ``run_engagement`` so the test validates the joblib
        # merge logic in isolation and stays fast in CI.
        import project_icarus.sim.api as api
        from project_icarus.c2 import run_campaign, CampaignThreat
        from project_icarus.scenarios.presets import build_interceptor_config
        from project_icarus.guidance.law import GuidanceLaw

        cfg, g = build_interceptor_config("arrow3")
        glaw = GuidanceLaw(g)
        bats = [Battery("A", cfg, glaw, location=np.zeros(3), magazine=10, salvo_size=1)]

        # Deterministic per-pair stub so serial == parallel by construction.
        def _stub(interceptor, guidance, target, scenario, n_trials, perturbations):
            key = (id(target), id(bats[0]))
            return type("E", (), {"miss_distance": float(hash(key)) % 100})()

        saved = api.run_engagement
        api.run_engagement = _stub
        try:
            from project_icarus.scenarios.target_factory import BallisticScenario
            threats = [
                CampaignThreat(
                    target=BallisticScenario(r0=np.array([6.4e6, 0, 0]),
                                          v0=np.array([0.0, 800.0, 0.0])),
                    aim_point=np.zeros(3), priority=1.0, label=f"T{i}")
                for i in range(4)
            ]

            def builder(th, bat):
                from project_icarus.sim.api import EngagementScenario
                return EngagementScenario(engagement_end=40.0)

            ser = run_campaign(threats, bats, builder, n_trials=1, parallel=False)
            par = run_campaign(threats, bats, builder, n_trials=1, parallel=True, n_jobs=2)
        finally:
            api.run_engagement = saved

        # Same number of pairwise engagements, same doctrine outcome.
        assert len(ser.engagements) == len(par.engagements) == 4
        assert ser.battle.n_threats == par.battle.n_threats == 4
        assert ser.battle.n_leakage == par.battle.n_leakage

    def test_architecture_summary_rolls_up(self):
        arch = build_architecture_from_locations(magazine_per_base=4, salvo_size=1)
        summ = architecture_summary(arch)
        # 11 sites (7 legacy + Patriot/THAAD) across 2 layers (upper + regional).
        assert summ["n_batteries"] == 11
        assert summ["total_magazine"] == 44
        assert summ["n_layers"] == 2
        # Magazine rolls up correctly across both layers.
        assert summ["total_magazine"] == sum(
            ly["total_magazine"] for ly in summ["layers"]
        )

    def test_national_metrics_aggregates(self):
        np.random.seed(7)
        arch = build_architecture_from_locations(magazine_per_base=6, salvo_size=1)
        bats = arch.batteries
        raid = _raid(12, bats, seed=7)
        c2 = DistributedC2Config()
        c2.space.bandwidth_tracks_per_s = 100.0
        c2.space.p_detect = 1.0
        c2.raid_arrival_window_s = 3.0

        def assess(t, bi):
            return 0.1

        r = run_layered_campaign(raid, arch, assess, c2=c2)
        m = national_metrics(r)
        assert m["n_inbound"] == 12
        assert m["n_defeated"] + m["n_leakage"] == 12
        assert m["n_dropped_c2_saturation"] == 0
        assert 0.0 <= m["mean_battery_utilization"] <= 1.0
        assert m["shots_fired"] >= m["n_defeated"]

    def test_national_metrics_saturation_drop(self):
        np.random.seed(8)
        arch = build_architecture_from_locations(magazine_per_base=6, salvo_size=1)
        bats = arch.batteries
        raid = _raid(20, bats, seed=8)
        c2 = DistributedC2Config()
        c2.space.bandwidth_tracks_per_s = 1.0
        c2.space.p_detect = 1.0
        c2.raid_arrival_window_s = 2.0

        def assess(t, bi):
            return 0.1

        r = run_layered_campaign(raid, arch, assess, c2=c2)
        m = national_metrics(r)
        assert m["n_dropped_c2_saturation"] > 0
        assert m["n_leakage"] == m["n_dropped_c2_saturation"] + m["n_cued_but_missed"]


class TestNationalMap:
    def test_build_national_scene(self):
        arch = build_architecture_from_locations(magazine_per_base=4, salvo_size=1)
        bats = arch.batteries
        defended = [np.asarray(b.location, float) for b in bats[:3]]
        threats = [np.asarray(b.location, float) + np.array([1e3, 0, 0]) for b in bats[:5]]
        scene = build_national_scene(arch, defended_points=defended, threat_points=threats)
        # MultiBlock with earth + base clouds + defended + threats.
        assert "earth" in scene.keys()
        assert "defended" in scene.keys()
        assert "threats" in scene.keys()

    def test_coverage_summary_table(self):
        arch = build_architecture_from_locations(magazine_per_base=4, salvo_size=1)
        rows = coverage_summary_table(arch)
        # One row per tier across all layers.
        n_tiers = sum(len(ly.tiers) for ly in arch.layers)
        assert len(rows) == n_tiers
        for row in rows:
            assert row["total_magazine"] == row["n_bases"] * row["magazine_per_base"]


import pytest as _pytest

_pytest.importorskip("panel")


class TestDashboard:
    def test_dashboard_synthetic_run(self):
        from project_icarus.c2.dashboard import NationalDashboard
        dash = NationalDashboard()
        dash._do_reset()
        assert dash._scene_png is not None  # offscreen PyVista PNG rendered
        dash.n_threats = 12
        dash.bandwidth = 2.0
        dash._do_run()
        m = dash._metrics
        assert m["n_inbound"] == 12
        assert m["n_tasked"] + m["n_dropped_c2_saturation"] == 12
        assert dash._scene_png is not None
        # metrics + coverage panes render without error
        dash.metrics_pane()
        dash.coverage_pane()
        dash.scene_pane()
        dash.view()


class TestLayeredParallel:
    def test_layered_assess_none_serial_matches_contract(self):
        # The assess=None path must fan out via JSON specs and return metrics.
        from project_icarus.c2.layers import run_layered_campaign
        from project_icarus.c2.battle_manager import ThreatTrack, BattleManagerConfig
        from project_icarus.scenarios.target_factory import BallisticScenario
        from project_icarus.scenarios.scenario import EngagementScenario
        import numpy as np

        arch = build_architecture_from_locations(magazine_per_base=2)
        # single-battery tiny arch for speed
        from project_icarus.c2.layers import DefenseArchitecture, Layer, Tier
        from reference.locations import locations_by_designation, coordinates_to_ecef
        tiny = DefenseArchitecture(name="tiny").add(
            Layer("upper_layer").add(Tier(kind="upper", interceptor_name="arrow3",
                                          bases=[np.zeros(3)], magazine_per_base=2, label="arrow3")))

        groups = locations_by_designation()
        defended = coordinates_to_ecef(groups["defended-target"][0])
        R_EARTH = 6.371e6

        def _raid(n):
            out = []
            for i in range(n):
                r0 = np.array([0.0, 0.0, R_EARTH + 1.0e5]) + np.array([i * 1e3, 0, 0])
                v0 = 5000.0 * (np.asarray(defended, float) - r0)
                v0 = v0 / np.linalg.norm(v0)
                tc = BallisticScenario(r0=r0, v0=v0, metadata={"name": f"t{i}"})
                out.append(ThreatTrack(target=tc, threat_id=i, aim_point=defended))
            return out

        raid = _raid(2)

        def sb(track, battery):
            return EngagementScenario(
                interceptor_launch_site=np.asarray(battery.location, float),
                target_launch_site=np.asarray(track.target.r0, float),
                engagement_end=10.0)

        res = run_layered_campaign(raid, tiny, scenario_builder=sb, n_trials=1,
                                   cfg=BattleManagerConfig())
        m = national_metrics(res)
        assert m["n_inbound"] == 2
        # Each raid element reaches a battery (no saturation at default bandwidth).
        assert m["n_tasked"] == 2
