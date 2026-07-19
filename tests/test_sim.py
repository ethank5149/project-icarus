import numpy as np
import pytest
from src.scenarios.target_factory import (
    BallisticScenario,
    FOBSScenario,
    HGVScenario,
    SuppressedScenario,
    SwarmScenario,
    GuidedThreatConfig,
    simulate_guided_threat,
    MU_EARTH,
    R_EARTH,
)
from src.guidance.terminal_guidance import TerminalGuidance
from src.scenarios.scenario import EngagementScenario
from src.scenarios.presets import (
    get_interceptor_presets,
    get_target_presets,
    interceptor_preset,
    target_preset,
    set_interceptor_geodetic,
    set_target_geodetic,
    geodetic_to_ecef,
    ecef_to_geodetic,
    _WGS84_A,
    _WGS84_B,
)
from src.sim.api import run_engagement
from src.sim.runner import EngagementRunner
from src.interceptors.config import InterceptorConfig, GuidanceConfig
from src.guidance.law import GuidanceLaw


class TestBallisticScenario:
    def test_propagate_shape(self):
        tgt = BallisticScenario(r0=np.array([R_EARTH, 0.0, 0.0]), v0=np.array([0.0, 1000.0, 0.0]))
        state = tgt.propagate(10.0)
        assert state.shape == (6,)

    def test_energy_conservation(self):
        tgt = BallisticScenario(r0=np.array([R_EARTH, 0.0, 0.0]), v0=np.array([0.0, 1000.0, 0.0]))
        state0 = tgt.propagate(0.0)
        state10 = tgt.propagate(10.0)
        e0 = 0.5 * np.linalg.norm(state0[3:]) ** 2 - MU_EARTH / np.linalg.norm(state0[:3])
        e10 = 0.5 * np.linalg.norm(state10[3:]) ** 2 - MU_EARTH / np.linalg.norm(state10[:3])
        assert np.isclose(e0, e10, rtol=1e-2)


class TestFOBSScenario:
    def test_apoapsis(self):
        tgt = FOBSScenario.from_orbital_params(apoapsis_km=200.0, inclination_deg=0.0)
        assert tgt.apoapsis_km == 200.0

    def test_propagate(self):
        tgt = FOBSScenario(r0=np.array([R_EARTH, 0.0, 0.0]), v0=np.array([0.0, 7000.0, 0.0]))
        state = tgt.propagate(60.0)
        assert state.shape == (6,)


class TestHGVScenario:
    def test_propagate(self):
        tgt = HGVScenario.from_params(max_alt_km=80.0, lateral_range_km=2000.0)
        state = tgt.propagate(10.0)
        assert state.shape == (6,)


class TestSuppressedScenario:
    def test_propagate(self):
        tgt = SuppressedScenario(r0=np.array([R_EARTH, 0.0, 0.0]), v0=np.array([0.0, 800.0, 800.0]))
        state = tgt.propagate(30.0)
        assert state.shape == (6,)


class TestSwarmScenario:
    def test_propagate(self):
        tgt = SwarmScenario.from_params(n_payloads=3, spread_deg=2.0, range_km=500.0)
        state = tgt.propagate(10.0)
        assert state.shape == (6,)

    def test_payload_states(self):
        tgt = SwarmScenario.from_params(n_payloads=3, spread_deg=2.0, range_km=500.0)
        states = tgt.payload_states(10.0)
        assert len(states) == 3
        assert all(s.shape == (6,) for s in states)


class TestEngagementScenario:
    def test_defaults(self):
        s = EngagementScenario()
        assert s.name == "Default"
        assert s.engagement_end == 300.0


class TestSwarmScenario:
    def test_defaults(self):
        s = SwarmScenario()
        assert s.n_payloads == 3


class TestInterceptorPresets:
    def test_get_interceptor_presets(self):
        presets = get_interceptor_presets()
        assert isinstance(presets, dict)
        assert "vandenberg" in presets
        assert "cape_canaveral" in presets
        assert "kwajalein" in presets
        assert "schriever" in presets
        assert "fort_greely" in presets
        assert "clear_sfs" in presets
        assert "custom_geodetic" in presets

    def test_interceptor_preset_values(self):
        vandenberg = interceptor_preset("vandenberg")
        assert isinstance(vandenberg, np.ndarray)
        assert vandenberg.shape == (3,)
        assert np.linalg.norm(vandenberg) > R_EARTH

    def test_interceptor_preset_invalid(self):
        with pytest.raises(KeyError):
            interceptor_preset("nonexistent")

    def test_geodetic_interceptor_presets(self):
        cape = interceptor_preset("cape_canaveral")
        assert isinstance(cape, np.ndarray)
        assert cape.shape == (3,)
        assert np.linalg.norm(cape) > R_EARTH

    def test_set_interceptor_geodetic(self):
        site = set_interceptor_geodetic(34.7, -120.6, 0.0)
        assert isinstance(site, np.ndarray)
        assert site.shape == (3,)
        assert np.linalg.norm(site) > R_EARTH


class TestGeodeticConversion:
    def test_equator_zero_alt(self):
        r = geodetic_to_ecef(0.0, 0.0, 0.0)
        assert np.allclose(r, [_WGS84_A, 0.0, 0.0], atol=1.0)

    def test_north_pole(self):
        r = geodetic_to_ecef(90.0, 0.0, 0.0)
        assert np.allclose(r, [0.0, 0.0, _WGS84_B], atol=1.0)

    def test_roundtrip(self):
        lat, lon, alt = 34.7, -120.6, 100.0
        r = geodetic_to_ecef(lat, lon, alt)
        lat2, lon2, alt2 = ecef_to_geodetic(r)
        assert np.isclose(lat, lat2, atol=1e-4)
        assert np.isclose(lon, lon2, atol=1e-4)
        assert np.isclose(alt, alt2, atol=1.0)

    def test_custom_geodetic_target(self):
        preset = set_target_geodetic(34.7, -120.6, 0.0, v0=np.array([0.0, 1200.0, 0.0]))
        assert preset.name.startswith("custom_geodetic_")
        assert preset.target.r0.shape == (3,)
        assert preset.engagement.engagement_end == 300.0


class TestTargetPresets:
    def test_get_target_presets(self):
        presets = get_target_presets()
        assert isinstance(presets, dict)
        assert "ballistic_short_range" in presets
        assert "fobs_low_orbital" in presets
        assert "hgp_hypersonic_glide" in presets
        assert "suppressed_deep_dip" in presets
        assert "swarm_tight" in presets
        assert "ballistic_target_moscow" in presets
        assert "ballistic_target_beijing" in presets

    def test_target_preset_structure(self):
        preset = target_preset("ballistic_target_moscow")
        assert preset.name == "ballistic_target_moscow"
        assert isinstance(preset.target, BallisticScenario)
        assert isinstance(preset.engagement, EngagementScenario)
        assert preset.description != ""

    def test_target_preset_propagate(self):
        preset = target_preset("hgp_hypersonic_glide")
        state = preset.target.propagate(5.0)
        assert state.shape == (6,)

    def test_target_preset_invalid(self):
        with pytest.raises(KeyError):
            target_preset("nonexistent")


class TestEngagementRunner:
    def test_run_returns_result(self):
        interceptor = InterceptorConfig(name="Test", mass=1000.0)
        guidance = GuidanceLaw()
        target = BallisticScenario(r0=np.array([R_EARTH, 0.0, 0.0]), v0=np.array([0.0, 1000.0, 0.0]))
        # Launch from 100 km altitude so the coasting interceptor leaves the
        # dense lower atmosphere (the surface-skimming regime is numerically
        # stiff and slow); this keeps the smoke test fast while exercising the
        # full event-driven pipeline.
        scenario = EngagementScenario(
            interceptor_launch_site=np.array([R_EARTH + 100e3, 0.0, 0.0]),
            engagement_end=60.0,
        )
        runner = EngagementRunner(interceptor=interceptor, guidance=guidance, target=target, scenario=scenario)
        result = runner.run(n_trials=5)
        assert isinstance(result.miss_distance, float)
        assert isinstance(result.kill_assessment, bool)
        if result.monte_carlo:
            assert len(result.monte_carlo.miss_distances) == 5

    def test_v0_computed_from_geometry(self):
        from src.sim.runner import _compute_v0
        scenario = EngagementScenario(
            interceptor_launch_site=np.array([R_EARTH, 0.0, 0.0]),
            threat_axis=np.array([1.0, 0.0, 0.0]),
        )
        v0 = _compute_v0(scenario, magnitude=1500.0)
        assert np.isclose(np.linalg.norm(v0), 1500.0)
        assert np.allclose(v0, [1500.0, 0.0, 0.0])

    def test_monte_carlo_perturbation(self):
        interceptor = InterceptorConfig(name="Test", mass=1000.0)
        guidance = GuidanceLaw()
        target = BallisticScenario(r0=np.array([R_EARTH, 0.0, 0.0]), v0=np.array([0.0, 1000.0, 0.0]))
        scenario = EngagementScenario(
            interceptor_launch_site=np.array([R_EARTH + 100e3, 0.0, 0.0]),
            engagement_end=60.0,
        )
        runner = EngagementRunner(interceptor=interceptor, guidance=guidance, target=target, scenario=scenario)
        result = runner.run(n_trials=3, perturbations={"position_sigma": 50.0, "velocity_sigma": 2.0})
        assert result.monte_carlo is not None
        assert len(result.monte_carlo.miss_distances) == 3
        assert all(np.isfinite(m) for m in result.monte_carlo.miss_distances)


class TestLocationsDatabase:
    def test_locations_loaded(self):
        from reference.locations import load_locations

        recs = load_locations()
        assert len(recs) > 50
        assert all("designation" in r for r in recs)

    def test_designation_groups(self):
        from reference.locations import locations_by_designation

        groups = locations_by_designation()
        assert "interceptor-launch-site" in groups
        assert "target-launch-site" in groups
        assert "defended-target" in groups
        assert len(groups["target-launch-site"]) >= 20
        assert len(groups["defended-target"]) >= 30

    def test_coordinates_to_ecef(self):
        from reference.locations import load_locations, coordinates_to_ecef

        rec = next(r for r in load_locations() if r["name"] == "Vandenberg Space Force Base")
        r = coordinates_to_ecef(rec)
        assert r.shape == (3,)
        # Should be within a few km of the WGS84 surface (mean radius 6371 km).
        assert abs(np.linalg.norm(r) - R_EARTH) < 50e3

    def test_locations_by_name(self):
        from reference.locations import locations_by_name

        by_name = locations_by_name()
        assert "Fort Greely" in by_name
        assert by_name["Fort Greely"]["designation"] == "interceptor-launch-site"


class TestExpandedPresets:
    def test_interceptor_presets_from_yaml(self):
        presets = get_interceptor_presets()
        from reference.locations import locations_by_designation, _sanitize_key

        for rec in locations_by_designation()["interceptor-launch-site"]:
            key = _sanitize_key(rec["name"])
            assert key in presets
            assert abs(np.linalg.norm(presets[key]) - R_EARTH) < 50e3

    def test_target_launch_presets_from_yaml(self):
        presets = get_target_presets()
        from reference.locations import locations_by_designation, _sanitize_key

        for rec in locations_by_designation()["target-launch-site"]:
            key = "ballistic_target_" + _sanitize_key(rec["name"])
            assert key in presets
            assert isinstance(presets[key].target, BallisticScenario)

    def test_defended_target_presets(self):
        presets = get_target_presets()
        assert "defended_washington_dc" in presets
        assert "defended_whiteman_afb" in presets
        # Defended presets carry their geodetic aim point in target_launch_site
        # (which lies on/near the WGS84 surface, not strictly above mean radius).
        preset = presets["defended_washington_dc"]
        assert abs(np.linalg.norm(preset.engagement.target_launch_site) - R_EARTH) < 50e3

    def test_yaml_presets_propagate(self):
        preset = target_preset("ballistic_target_kozelsk")
        state = preset.target.propagate(5.0)
        assert state.shape == (6,)


class TestThreatToDefended:
    def test_curated_pairs_registered(self):
        presets = get_target_presets()
        pairs = [k for k in presets if k.startswith("threat_") and "_to_" in k]
        assert len(pairs) >= 15

    def test_pair_is_aimed_at_defended(self):
        preset = target_preset("threat_kozelsk_to_washington_dc")
        # Launch speed is ICBM-class and the aim point is set on the WGS84 surface.
        assert 5000.0 < np.linalg.norm(preset.target.v0) < 15000.0
        assert abs(np.linalg.norm(preset.engagement.target_launch_site) - R_EARTH) < 50e3
        state = preset.target.propagate(5.0)
        assert state.shape == (6,)

    def test_build_threat_to_defended_helper(self):
        from src.scenarios.presets import build_threat_to_defended

        preset = build_threat_to_defended("Tatishchevo", "Washington D.C.")
        assert isinstance(preset.target, BallisticScenario)
        assert "azimuth" in preset.description
        assert "range" in preset.description
        # Unknown names should raise.
        import pytest

        with pytest.raises(KeyError):
            build_threat_to_defended("Nonexistent Threat", "Washington D.C.")
        with pytest.raises(KeyError):
            build_threat_to_defended("Kozelsk", "Nonexistent Target")

    def test_azimuth_and_range_physically_reasonable(self):
        from src.scenarios.presets import geodetic_launch_to_target

        # Short range should give modest speed; long range higher speed.
        short = geodetic_launch_to_target(34.74, -120.57, 361, 34.05, -118.24, 305)
        long = geodetic_launch_to_target(54.02, 35.46, 490, 38.90, -77.04, 40)
        assert np.linalg.norm(short.target.v0) < np.linalg.norm(long.target.v0)
        # Long-range ICBM-class speed.
        assert np.linalg.norm(long.target.v0) > 8000.0


class TestScenarioVariants:
    """All threat families should be geodetically aimed at the defended point."""

    def test_all_scenario_types_build(self):
        from src.scenarios.presets import build_threat_to_defended

        for sc in ("ballistic", "fobs", "hgv", "suppressed", "swarm"):
            preset = build_threat_to_defended("Kozelsk", "Washington D.C.", scenario_type=sc)
            state = preset.target.propagate(5.0)
            assert state.shape == (6,)
            # Every variant carries the defended aim point on the WGS84 surface.
            assert abs(np.linalg.norm(preset.engagement.target_launch_site) - R_EARTH) < 50e3

    def test_fobs_reentry_aims_at_target(self):
        from src.scenarios.presets import build_threat_to_defended
        from src.scenarios.target_factory import FOBSScenario

        preset = build_threat_to_defended("Kozelsk", "Washington D.C.", scenario_type="fobs")
        assert isinstance(preset.target, FOBSScenario)
        st = preset.target.propagate(1100.0)
        # Reentry now steers toward the aim point (not a hardcoded fixed point).
        dist = np.linalg.norm(st[:3] - preset.engagement.target_launch_site)
        assert dist < 8000e3

    def test_suppressed_maneuver_along_threat_axis(self):
        from src.scenarios.presets import build_threat_to_defended
        from src.scenarios.target_factory import SuppressedScenario

        preset = build_threat_to_defended("Kozelsk", "Washington D.C.", scenario_type="suppressed")
        assert isinstance(preset.target, SuppressedScenario)
        # Maneuver direction lies in the horizontal plane (threat axis), not +y.
        assert abs(preset.target._maneuver_dir[2]) < 1e-9
        assert np.linalg.norm(preset.target._maneuver_dir) > 0.0

    def test_hgv_inserted_at_glide_altitude(self):
        from src.scenarios.presets import build_threat_to_defended
        from src.scenarios.target_factory import HGVScenario

        preset = build_threat_to_defended(
            "Kozelsk", "Washington D.C.", scenario_type="hgv", glide_alt_km=70.0
        )
        assert isinstance(preset.target, HGVScenario)
        alt = np.linalg.norm(preset.target.r0) - R_EARTH
        assert 60e3 < alt < 80e3

    def test_unknown_scenario_type_raises(self):
        import pytest
        from src.scenarios.presets import geodetic_launch_to_target

        with pytest.raises(ValueError):
            geodetic_launch_to_target(54.02, 35.46, 490, 38.90, -77.04, 40, scenario_type="bogus")

    def test_curated_variants_registered(self):
        presets = get_target_presets()
        suffixes = ("_fobs", "_hgv", "_suppressed")
        for suf in suffixes:
            keys = [k for k in presets if k.startswith("threat_") and k.endswith(suf)]
            assert len(keys) >= 15, f"missing {suf} variants"


class TestGuidedThreat:
    """The FOBS reentry is a real closed-loop PN-guided EOM6DOF solve, not the
    ad-hoc steering term. These tests exercise ``simulate_guided_threat``
    directly and via the FOBS scenario."""

    def _suborbital_collision_course(self):
        """A threat on a near-collision course with a ground aim point."""
        r0 = np.array([R_EARTH + 120e3, 0.0, 0.0])
        aim = np.array([R_EARTH, 0.0, 0.0])
        to_aim = aim - r0
        v0 = 6500.0 * to_aim / np.linalg.norm(to_aim)
        return r0, v0, aim

    def test_returns_six_vector(self):
        r0, v0, aim = self._suborbital_collision_course()
        times, states = simulate_guided_threat(r0, v0, aim, t_end=120.0)
        assert states.shape[0] == 6  # [r(3), v(3)]
        assert times.shape[0] == states.shape[1]
        assert np.all(np.isfinite(states))

    def test_reaches_ground(self):
        r0, v0, aim = self._suborbital_collision_course()
        times, states = simulate_guided_threat(r0, v0, aim, t_end=300.0)
        final_alt = np.linalg.norm(states[:3, -1]) - R_EARTH
        # The ground-impact terminal event stops the integration at the surface.
        assert final_alt <= 1e3

    def test_pn_converges_to_aim(self):
        r0, v0, aim = self._suborbital_collision_course()
        # Small lateral bias: PN must null it out.
        v0 = v0 + np.array([0.0, 25.0, 0.0])
        times, states = simulate_guided_threat(
            r0, v0, aim, vehicle=GuidedThreatConfig(accel_limit=80.0),
            guidance_law=TerminalGuidance(accel_limit=80.0), t_end=300.0,
        )
        miss = np.linalg.norm(states[:3, -1] - aim)
        assert miss < 5e3

    def test_pn_improves_over_ballistic(self):
        r0, v0, aim = self._suborbital_collision_course()
        v0 = v0 + np.array([0.0, 200.0, 0.0])  # larger lateral bias
        guided = simulate_guided_threat(
            r0, v0, aim, vehicle=GuidedThreatConfig(accel_limit=80.0),
            guidance_law=TerminalGuidance(accel_limit=80.0), t_end=300.0,
        )
        ballistic = simulate_guided_threat(
            r0, v0, aim, vehicle=GuidedThreatConfig(accel_limit=80.0),
            guidance_law=TerminalGuidance(accel_limit=0.0), t_end=300.0,
        )
        miss_guided = np.linalg.norm(guided[1][:3, -1] - aim)
        miss_ballistic = np.linalg.norm(ballistic[1][:3, -1] - aim)
        assert miss_guided <= miss_ballistic

    def test_fobs_reentry_is_guided_solve(self):
        from src.scenarios.presets import build_threat_to_defended

        preset = build_threat_to_defended("Kozelsk", "Washington D.C.", scenario_type="fobs")
        target = preset.target
        assert isinstance(target, FOBSScenario)
        # The guided reentry reaches the surface (ground-impact event) within the
        # engagement window rather than hanging in a virtual orbit.
        st = target.propagate(1200.0)
        assert st.shape == (6,)
        final_alt = np.linalg.norm(st[:3]) - R_EARTH
        assert final_alt < 50e3

