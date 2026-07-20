import numpy as np
import pytest

from src.c2 import (
    BattleManager,
    BattleManagerConfig,
    ThreatTrack,
    Battery,
    greedy_allocate,
    hungarian_allocate,
    run_campaign,
    CampaignThreat,
)
from src.interceptors.config import InterceptorConfig, GuidanceConfig
from src.scenarios.target_factory import BallisticScenario, R_EARTH
from src.scenarios.scenario import EngagementScenario


def _interceptor():
    ic, gc = InterceptorConfig(name="Test", kill_radius=0.5), GuidanceConfig()
    return ic, gc


def _battery(name, magazine=5, salvo=2, loc=(R_EARTH, 0.0, 0.0)):
    ic, gc = _interceptor()
    return Battery(name=name, interceptor_config=ic, guidance_config=gc,
                   location=np.array(loc, dtype=float), magazine=magazine, salvo_size=salvo)


def _track(tid, aim=(R_EARTH, 0.0, 0.0), priority=1.0):
    return ThreatTrack(threat_id=tid, target=None, aim_point=np.array(aim, dtype=float),
                       priority=priority)


class TestAllocators:
    def test_greedy_respects_magazine(self):
        threats = [_track(0), _track(1)]
        bats = [_battery("A", magazine=3, salvo=2)]
        comm = greedy_allocate(threats, bats, lambda x: np.asarray(x, dtype=float), salvo_size=2)
        total = sum(c[2] for c in comm)
        assert total == 3  # only 3 in magazine
        assert bats[0].magazine == 0

    def test_greedy_prioritizes_high_priority(self):
        threats = [_track(0, priority=1.0), _track(1, priority=10.0)]
        bats = [_battery("A", magazine=2, salvo=1)]
        comm = greedy_allocate(threats, bats, lambda x: np.asarray(x, dtype=float), salvo_size=1)
        # single salvo of 1 must go to the higher-priority threat (id 1)
        assert comm[0][0] == 1

    def test_hungarian_minimizes_range(self):
        # Two batteries: A near threat 0, B near threat 1.
        threats = [_track(0, aim=(R_EARTH, 0.0, 0.0)),
                   _track(1, aim=(R_EARTH, 0.0, 0.0) + np.array([0, 1e6, 0]))]
        bats = [_battery("A", magazine=5, loc=(R_EARTH, 0.0, 0.0)),
                _battery("B", magazine=5, loc=(R_EARTH, 0.0, 0.0) + np.array([0, 1e6, 0]))]
        comm = hungarian_allocate(threats, bats, lambda x: np.asarray(x, dtype=float), max_per_battery=1)
        # Each threat should be served by its nearer battery.
        mapping = {(c[0], c[1]) for c in comm}
        assert (0, 0) in mapping
        assert (1, 1) in mapping


class TestBattleManager:
    def test_shoot_look_shoot_defeats_with_salvo(self):
        # assess returns miss distances; first shot kills (<= kill_radius).
        threats = [_track(0)]
        bats = [_battery("A", magazine=5, salvo=1)]
        bm = BattleManager(threats, bats, cfg=BattleManagerConfig(
            doctrine="shoot_look_shoot", allocator="greedy", salvo_size=1, max_rounds=3))

        shot = {"n": 0}
        def assess(track, bi):
            shot["n"] += 1
            return 0.1 if shot["n"] == 1 else 10.0  # first shot kills

        res = bm.run(assess)
        assert res.n_defeated == 1
        assert res.n_leakage == 0
        assert res.shots_fired == 1

    def test_leakage_when_out_of_ammo(self):
        # assess always misses; battery runs dry -> leakage.
        threats = [_track(0), _track(1)]
        bats = [_battery("A", magazine=1, salvo=1)]
        bm = BattleManager(threats, bats, cfg=BattleManagerConfig(
            doctrine="salvo", allocator="greedy", salvo_size=1, max_rounds=2))

        def assess(track, bi):
            return 50.0  # always miss

        res = bm.run(assess)
        assert res.n_defeated == 0
        assert res.n_leakage == 2
        assert res.leakage_fraction == 1.0
        assert res.shots_fired == 1  # only one interceptor available

    def test_salvo_fires_multiple_then_stops(self):
        threats = [_track(0)]
        bats = [_battery("A", magazine=5, salvo=3)]
        bm = BattleManager(threats, bats, cfg=BattleManagerConfig(
            doctrine="salvo", allocator="greedy", salvo_size=3, max_rounds=4))

        def assess(track, bi):
            return 50.0  # miss every time

        res = bm.run(assess)
        # salvo fires a full salvo (3) in one round then stops.
        assert res.shots_fired == 3
        assert res.n_leakage == 1

    def test_battery_utilization_reported(self):
        threats = [_track(0)]
        bats = [_battery("A", magazine=4, salvo=2)]
        bm = BattleManager(threats, bats, cfg=BattleManagerConfig(
            doctrine="salvo", allocator="greedy", salvo_size=2, max_rounds=1))

        def assess(track, bi):
            return 50.0

        res = bm.run(assess)
        util = res.battery_utilization
        assert util["A"] == 2.0 / 4.0


class TestCampaign:
    def test_campaign_saturation_raid(self):
        # Build a tiny raid of 3 threats aimed at one defended point.
        aim = np.array([R_EARTH, 0.0, 0.0])
        threats = []
        for i in range(3):
            r0 = np.array([R_EARTH + 1e6, float(i) * 1e5, 0.0])
            v0 = np.array([0.0, -2000.0, 0.0])
            threats.append(CampaignThreat(
                target=BallisticScenario(r0=r0, v0=v0),
                aim_point=aim, priority=1.0))

        # One big battery that should defeat everything (each engagement is cheap
        # with small n_trials; we assert structure rather than physical kill).
        batteries = [_battery("A", magazine=9, salvo=1)]

        def scenario_builder(threat, battery):
            return EngagementScenario(
                name="campaign",
                interceptor_launch_site=np.array([R_EARTH, 0.0, 0.0]),
                target_launch_site=threat.launch_site,
                threat_axis=aim - threat.launch_site,
                engagement_end=300.0,
            )

        cfg = BattleManagerConfig(doctrine="shoot_look_shoot", allocator="greedy",
                                 salvo_size=1, max_rounds=2)
        result = run_campaign(
            threats=threats, batteries=batteries,
            scenario_builder=scenario_builder, cfg=cfg,
            n_trials=8, perturbations={"position": 100.0})

        # 3 threats x 1 battery = 3 pairwise engagements precomputed.
        assert len(result.engagements) == 3
        # System metrics well-defined.
        s = result.summary()
        assert 0.0 <= s["leakage_fraction"] <= 1.0
        assert s["n_threats"] == 3
        assert s["shots_fired"] >= 3
