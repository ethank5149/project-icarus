"""Comprehensive validation tests for Sarmat physics and guidance.

These tests verify that the existing production code produces physically
realistic results BEFORE we build optimization on top of it.

A 7,400 km range ICBM should reach:
- ~200+ km altitude at burnout
- ~7+ km/s velocity at burnout
- Miss distance < 100m with optimized guidance

If these tests fail, we MUST fix the bugs before proceeding.
"""

import numpy as np
import pytest

from project_icarus.scenarios.target_factory import (
    SarmatScenario,
    _geodetic_to_ecef_simple,
    _ecef_to_geodetic,
    _ground_altitude,
    _two_body_accel,
    _enu_basis,
)
from project_icarus.reference.surface_elevation import get_surface_elevation
from project_icarus.guidance.icbm_guidance import _great_circle_azimuth


# Test constants
KOZELSK_LAT = 54.07
KOZELSK_LON = 35.73
DC_LAT = 38.90
DC_LON = -77.04
KOZELSK_ELEV = get_surface_elevation(KOZELSK_LAT, KOZELSK_LON)
DC_ELEV = get_surface_elevation(DC_LAT, DC_LON)

R0 = _geodetic_to_ecef_simple(KOZELSK_LAT, KOZELSK_LON, KOZELSK_ELEV)
R_TARGET = _geodetic_to_ecef_simple(DC_LAT, DC_LON, DC_ELEV)


class TestCoordinateSystems:
    """Validate ECEF/geodetic conversions and ENU basis."""

    def test_launch_target_coordinates(self):
        """Verify launch and target are in correct locations."""
        lat0, lon0, alt0 = _ecef_to_geodetic(R0)
        lat1, lon1, alt1 = _ecef_to_geodetic(R_TARGET)

        assert abs(lat0 - KOZELSK_LAT) < 0.01
        assert abs(lon0 - KOZELSK_LON) < 0.01
        assert abs(lat1 - DC_LAT) < 0.01
        assert abs(lon1 - DC_LON) < 0.01

    def test_range_distance(self):
        """Verify launch-target distance is ~7,400 km."""
        distance = np.linalg.norm(R0 - R_TARGET) / 1000
        assert 7000 < distance < 8000, f"Range {distance:.1f} km is outside expected [7000, 8000] km"

    def test_enu_basis_orthonormal(self):
        """ENU basis should be orthonormal."""
        lat, lon, _ = _ecef_to_geodetic(R0)
        east, north, up = _enu_basis(lat, lon)

        # Check unit vectors
        assert abs(np.linalg.norm(east) - 1.0) < 1e-6
        assert abs(np.linalg.norm(north) - 1.0) < 1e-6
        assert abs(np.linalg.norm(up) - 1.0) < 1e-6

        # Check orthogonality
        assert abs(np.dot(east, north)) < 1e-6
        assert abs(np.dot(east, up)) < 1e-6
        assert abs(np.dot(north, up)) < 1e-6

    def test_great_circle_azimuth_direction(self):
        """Azimuth should point toward target, not away."""
        az_deg = _great_circle_azimuth(R0, R_TARGET)
        assert 0 <= az_deg < 360, f"Azimuth {az_deg} should be in [0, 360)"

        # DC is WNW from Kozelsk, so azimuth should be ~310-315 degrees
        assert 280 < az_deg < 340, f"Azimuth {az_deg:.1f}° should be WNW (~310-315°)"


class TestSarmatScenarioPhysics:
    """Validate SarmatScenario produces realistic ICBM performance."""

    @pytest.fixture
    def scenario(self):
        """Create a basic SarmatScenario."""
        scenario = SarmatScenario(
            r0=R0.copy(),
            v0=np.zeros(3),
            use_j2=True,
        )
        return scenario

    def test_initial_mass(self, scenario):
        """Verify initial mass matches RS-28 specs."""
        mass = scenario._initial_mass()
        assert 200000 < mass < 210000, f"Initial mass {mass:.0f} kg should be ~208,100 kg"

    def test_stage_thrust_levels(self, scenario):
        """Verify stage thrust matches RS-28 published specs from authoritative sources."""
        stages = scenario._SARMAT_STAGES

        # Stage 1: PDU-99 derived from RD-274 (~4,952 kN)
        assert abs(stages[0]["thrust"] - 5.0e6) < 0.5e6

        # Stage 2: RD-250 class (~1,200 kN estimated)
        assert abs(stages[1]["thrust"] - 1.2e6) < 0.2e6

        # Stage 3: Four RS-99 engines, "over 100 tons thrust" (~1,000+ kN)
        assert abs(stages[2]["thrust"] - 1.0e6) < 0.2e6

    def test_stage_burn_times(self, scenario):
        """Verify stage burn times sum to ~360s."""
        stages = scenario._SARMAT_STAGES
        total_burn = sum(s["t_end"] - s["t_start"] for s in stages)
        assert 300 < total_burn < 420, f"Total burn {total_burn:.0f}s should be ~360s"

    def test_mass_flow_rates(self, scenario):
        """Verify mass flow rates are physically consistent."""
        stages = scenario._SARMAT_STAGES

        for i, stage in enumerate(stages):
            burn_time = stage["t_end"] - stage["t_start"]
            propellant_mass = abs(stage["m_dot"]) * burn_time

            # Stage 1: ~129t propellant (140t wet - 11.34t dry)
            if i == 0:
                assert 120e3 < propellant_mass < 135e3

            # Stage 2: ~44t propellant (48t wet - 3.6t dry)
            elif i == 1:
                assert 40e3 < propellant_mass < 48e3

            # Stage 3: ~8t propellant (10.1t wet - 2.16t dry)
            elif i == 2:
                assert 7e3 < propellant_mass < 10e3

    def test_thrust_direction_points_roughly_toward_target(self, scenario):
        """At t=5s, thrust should point roughly toward DC, not away."""
        thrust_dir = scenario._current_thrust_dir(5.0, R0 + np.array([0, 0, 1000.0]), np.zeros(3))

        to_target = R_TARGET - R0
        to_target_dir = to_target / np.linalg.norm(to_target)

        # Should have positive projection toward target
        dot_product = np.dot(thrust_dir, to_target_dir)
        assert dot_product > 0.3, \
            f"Thrust direction dot product with target {dot_product:.3f} should be > 0.3"

    def test_trajectory_reaches_space(self, scenario):
        """ICBM should reach at least 100km altitude (Karman line)."""
        times, states = scenario._integrate_full()

        max_alt = max(_ground_altitude(state[:3]) for state in states)
        assert max_alt > 100e3, f"Max altitude {max_alt/1e3:.1f} km should exceed 100 km"

    def test_trajectory_reaches_orbital_velocity(self, scenario):
        """ICBM should reach at least 7 km/s (orbital velocity ~7.9 km/s)."""
        times, states = scenario._integrate_full()

        max_speed = max(np.linalg.norm(state[3:6]) for state in states)
        assert max_speed > 7e3, f"Max speed {max_speed/1e3:.1f} km/s should exceed 7 km/s"

    def test_burnout_conditions(self, scenario):
        """At burnout, altitude and speed should be in realistic ranges."""
        times, states = scenario._integrate_full()

        # Find burnout state (last boost stage ends at 342s)
        burnout_idx = np.argmax(times >= 342.0)
        if burnout_idx >= len(states):
            burnout_idx = -1

        burnout_state = states[burnout_idx]
        burnout_alt = _ground_altitude(burnout_state[:3])
        burnout_speed = np.linalg.norm(burnout_state[3:6])

        # Realistic ICBM burnout: 150-300km altitude, 6-8 km/s
        assert 100e3 < burnout_alt < 400e3, \
            f"Burnout altitude {burnout_alt/1e3:.1f} km should be in [100, 400] km"
        assert 6e3 < burnout_speed < 9e3, \
            f"Burnout speed {burnout_speed/1e3:.1f} km/s should be in [6, 9] km/s"


class TestTrajectoryOptimization:
    """Validate that trajectory optimization produces realistic results."""

    def test_default_guidance_miss_distance(self):
        """Default guidance should produce miss < 1000km (not 7000km)."""
        from project_icarus.optimization.direct_trajectory_optimizer import compute_miss_distance

        params = [np.radians(45.0), np.radians(25.0), 100.0, 6800.0]
        miss = compute_miss_distance(params)

        # If it's > 1000km, the guidance is fundamentally broken
        assert miss < 1000e3, \
            f"Default guidance miss {miss/1e3:.1f} km is too large - guidance logic is broken"

    def test_trajectory_optimization_can_improve(self):
        """Optimization should be able to reduce miss distance."""
        from project_icarus.optimization.direct_trajectory_optimizer import compute_miss_distance

        default_params = [np.radians(45.0), np.radians(25.0), 100.0, 6800.0]
        default_miss = compute_miss_distance(default_params)

        # Try different parameters
        test_params = [
            [np.radians(55.0), np.radians(20.0), 120.0, 7000.0],
            [np.radians(50.0), np.radians(15.0), 150.0, 7200.0],
            [np.radians(60.0), np.radians(10.0), 180.0, 7500.0],
        ]

        best_miss = default_miss
        for params in test_params:
            miss = compute_miss_distance(params)
            best_miss = min(best_miss, miss)

        # At least one parameter set should improve on default
        assert best_miss < default_miss * 0.9, \
            f"Could not improve on default miss {default_miss/1e3:.1f} km"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
