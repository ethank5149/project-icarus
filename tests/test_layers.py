import numpy as np

from src.c2 import (
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
)
from src.scenarios.presets import build_interceptor_config


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
        # 7 interceptor-launch-site records in the DB.
        assert len(arch.batteries) == 7
        assert arch.total_magazine == 28
        # All batteries carry a real interceptor config.
        for bat in arch.batteries:
            assert bat.interceptor_config is not None
            assert bat.magazine == 4

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
