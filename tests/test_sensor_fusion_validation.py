"""Phase 5: Sensor/C2 fusion validation and campaign statistical tests.

Extends the Phase 5A/B/C test suite with:
* Sensor track accuracy under varying clutter densities
* M-of-N confirmation reliability statistics
* Campaign-level kill-probability regression
* C2 latency sensitivity curves
"""

import numpy as np
import pytest

from project_icarus.dynamics.coordinate_systems import geodetic_to_ecef
from project_icarus.sensors.sensor import Sensor, SensorNetwork, Track
from project_icarus.sensors.network import load_sensors_from_locations
from project_icarus.c2 import (
    BattleManager, BattleManagerConfig, ThreatTrack, Battery,
    run_campaign, CampaignThreat, run_discrete_event, C2Scenario,
)
from project_icarus.interceptors.config import InterceptorConfig, GuidanceConfig
from project_icarus.scenarios.target_factory import BallisticScenario, R_EARTH
from project_icarus.scenarios.scenario import EngagementScenario


def _ecef(lat_deg, lon_deg, alt_m=0.0):
    return np.asarray(geodetic_to_ecef(lat_deg, lon_deg, alt_m), dtype=float)


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


class TestSensorFusionValidation:
    """Phase 5 validation: track accuracy and confirmation reliability."""

    def test_track_converges_under_high_clutter(self):
        """With 5 detections/scan clutter, the fused track must still converge."""
        sensors = load_sensors_from_locations(designation=["sensor"])
        rng = np.random.default_rng(42)
        net = SensorNetwork(sensors, confirmation_hits=3, use_clutter=True, rng=rng)
        for s in net.sensors:
            s.clutter_rate = 5.0

        center = _ecef(45.0, -90.0)
        path = [center + np.array([k * 5e3, k * 1.5e3, 700e3]) for k in range(12)]
        for k, tgt in enumerate(path):
            net.scan([tgt], rcs_m2=1.0, t=float(k))

        confirmed = net.confirmed_tracks(min_hits=3)
        assert len(confirmed) >= 1
        best = min(confirmed, key=lambda t: np.linalg.norm(t.position - path[-1]))
        assert np.linalg.norm(best.position - path[-1]) < 1.0e6

    def test_confirmation_rejects_spurious_clutter_only(self):
        """Pure-clutter scans must not confirm any track."""
        sensors = load_sensors_from_locations(designation=["sensor"])
        rng = np.random.default_rng(99)
        net = SensorNetwork(sensors, confirmation_hits=3, use_clutter=True, rng=rng)
        for s in net.sensors:
            s.clutter_rate = 4.0

        for k in range(6):
            net.scan([], rcs_m2=1.0, t=float(k))
        assert net.confirmed_tracks(min_hits=3) == []

    def test_track_rmse_scales_with_range(self):
        """Position RMSE should increase with target range."""
        sensor = Sensor("S", lat_deg=40.0, lon_deg=-100.0, max_range_m=2000e3,
                        range_std_m=200.0, angle_std_deg=0.1)
        rng = np.random.default_rng(7)
        near = _ecef(40.0, -100.0) + np.array([200e3, 0.0, 300e3])
        far = _ecef(40.0, -100.0) + np.array([1600e3, 0.0, 300e3])

        def track_error(truth_pos):
            dets = [sensor.detect(truth_pos, rng=rng, t=0.0) for _ in range(40)]
            dets = [d for d in dets if d is not None]
            errs = [np.linalg.norm(d.estimated_ecef() - truth_pos) for d in dets]
            return np.sqrt(np.mean(np.square(errs))) if errs else np.inf

        near_err = track_error(near)
        far_err = track_error(far)
        assert near_err < far_err


class TestCampaignStatistics:
    """Phase 5C: campaign-level statistical validation."""

    def test_campaign_kill_probability_monotonic_with_salvo(self):
        """Larger salvo size should increase kill probability (more shots)."""
        aim = np.array([R_EARTH, 0.0, 0.0])
        threats = []
        for i in range(4):
            r0 = np.array([R_EARTH + 1.5e6, float(i) * 2e5, 0.0])
            v0 = np.array([0.0, -2200.0, 0.0])
            threats.append(CampaignThreat(
                target=BallisticScenario(r0=r0, v0=v0),
                aim_point=aim, priority=1.0))

        def scenario_builder(threat, battery):
            return EngagementScenario(
                name="camp",
                interceptor_launch_site=np.array([R_EARTH, 0.0, 0.0]),
                target_launch_site=threat.launch_site,
                threat_axis=aim - threat.launch_site,
                engagement_end=280.0,
            )

        batteries_small = [_battery("A", magazine=20, salvo=1)]
        result_small = run_campaign(
            threats=threats, batteries=batteries_small,
            scenario_builder=scenario_builder,
            cfg=BattleManagerConfig(doctrine="salvo", allocator="greedy",
                                    salvo_size=1, max_rounds=4),
            n_trials=10, perturbations={"position": 80.0},
        )

        batteries_large = [_battery("B", magazine=20, salvo=2)]
        result_large = run_campaign(
            threats=threats, batteries=batteries_large,
            scenario_builder=scenario_builder,
            cfg=BattleManagerConfig(doctrine="salvo", allocator="greedy",
                                    salvo_size=2, max_rounds=4),
            n_trials=10, perturbations={"position": 80.0},
        )

        s_small = result_small.summary()
        s_large = result_large.summary()
        assert s_small["n_threats"] == s_large["n_threats"] == 4
        assert s_large["leakage_fraction"] <= s_small["leakage_fraction"]

    def test_c2_latency_increases_leakage(self):
        """Higher C2 latency should increase leakage fraction."""
        aim = _ecef(45.0, -90.0) + np.array([0.0, 0.0, 600e3])
        r0 = _ecef(45.0, -90.0) + np.array([0.0, 0.0, 1200e3])
        v0 = np.array([1000.0, 0.0, -300.0])
        threats = [CampaignThreat(
            target=BallisticScenario(r0=r0, v0=v0),
            aim_point=aim, priority=1.0)]

        def scenario_builder(threat, battery):
            return EngagementScenario(
                name="lat",
                interceptor_launch_site=aim,
                target_launch_site=threat.launch_site,
                threat_axis=aim - threat.launch_site,
                engagement_end=60.0,
            )

        result_low = run_campaign(
            threats=threats, batteries=[_battery("A", magazine=3, salvo=1)],
            scenario_builder=scenario_builder,
            cfg=BattleManagerConfig(doctrine="shoot_look_shoot", allocator="greedy",
                                    salvo_size=1, max_rounds=3, c2_latency_s=1.0),
            n_trials=12, perturbations={"position": 50.0},
        )
        result_high = run_campaign(
            threats=threats, batteries=[_battery("B", magazine=3, salvo=1)],
            scenario_builder=scenario_builder,
            cfg=BattleManagerConfig(doctrine="shoot_look_shoot", allocator="greedy",
                                    salvo_size=1, max_rounds=3, c2_latency_s=15.0),
            n_trials=12, perturbations={"position": 50.0},
        )

        assert result_high.summary()["leakage_fraction"] >= result_low.summary()["leakage_fraction"]


class TestDiscreteEventSweep:
    """Phase 5B: discrete-event C2 loop sweeps."""

    def test_discrete_event_single_sensor_single_threat(self):
        from project_icarus.sensors.sensor import Sensor

        sensor = Sensor("S1", lat_deg=45.0, lon_deg=-90.0, max_range_m=5000e3,
                        range_std_m=300.0, angle_std_deg=0.2)
        rng = np.random.default_rng(3)
        net = SensorNetwork([sensor], confirmation_hits=2, use_clutter=False, rng=rng)

        aim = _ecef(45.0, -90.0) + np.array([0.0, 0.0, 600e3])
        threats = [_track(0, aim=aim)]
        bats = [_battery("A", magazine=6, salvo=1, loc=aim)]

        bm = BattleManager(threats, bats, cfg=BattleManagerConfig(
            doctrine="shoot_look_shoot", allocator="greedy", salvo_size=1,
            max_rounds=3, c2_latency_s=1.0))

        scenario = C2Scenario(name="de", t_start=0.0, t_end=10.0, dt=1.0)

        def truth(t):
            return [_ecef(45.0, -90.0) + np.array([0.0, 0.0, 600e3])
                    + t * np.array([1e3, 0.0, -200.0])]

        def assess(track_arg, bi):
            return 0.1

        out = run_discrete_event(scenario, net, bm, truth, assess, rcs_m2=1.0)
        assert len(out["shots"]) >= 1
        assert out["battle"].n_defeated >= 1

    def test_discrete_event_multi_battery_allocation(self):
        from project_icarus.sensors.sensor import Sensor

        sensor = Sensor("S1", lat_deg=45.0, lon_deg=-90.0, max_range_m=5000e3,
                        range_std_m=300.0, angle_std_deg=0.2)
        rng = np.random.default_rng(5)
        net = SensorNetwork([sensor], confirmation_hits=2, use_clutter=False, rng=rng)

        aim = _ecef(45.0, -90.0) + np.array([0.0, 0.0, 600e3])
        threats = [_track(0, aim=aim)]
        bats = [_battery("A", magazine=3, salvo=1, loc=aim),
                _battery("B", magazine=3, salvo=1, loc=aim + np.array([5e3, 0.0, 0.0]))]

        bm = BattleManager(threats, bats, cfg=BattleManagerConfig(
            doctrine="salvo", allocator="hungarian", salvo_size=1,
            max_rounds=2, c2_latency_s=1.0))

        scenario = C2Scenario(name="multi", t_start=0.0, t_end=10.0, dt=1.0)

        def truth(t):
            return [aim + t * np.array([1e3, 0.0, -200.0])]

        def assess(track_arg, bi):
            return 0.1

        out = run_discrete_event(scenario, net, bm, truth, assess, rcs_m2=1.0)
        assert out["battle"].shots_fired <= 6
