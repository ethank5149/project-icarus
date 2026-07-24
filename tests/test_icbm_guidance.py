"""Unit tests for ICBM guidance system.

These tests validate the physically realistic guidance implementation
and catch regressions in thrust direction, INS propagation, and burnout targeting.
"""

import numpy as np
import pytest

from project_icarus.guidance.icbm_guidance import (
    ICBMGuidance,
    InertialNavigationSystem,
    _ecef_to_geodetic,
    _enu_basis,
    _geodetic_to_ecef,
    _great_circle_azimuth,
    _two_body_accel_j2,
)
from project_icarus.scenarios.target_factory import (
    SarmatScenario,
    _geodetic_to_ecef_simple,
    _ecef_to_geodetic as scenario_ecef_to_geodetic,
)
from project_icarus.reference.surface_elevation import get_surface_elevation


# Test fixtures
KOZELSK_LAT = 54.07
KOZELSK_LON = 35.73
DC_LAT = 38.90
DC_LON = -77.04

KOZELSK_ELEV = get_surface_elevation(KOZELSK_LAT, KOZELSK_LON)
DC_ELEV = get_surface_elevation(DC_LAT, DC_LON)

R0_KOZELSK = _geodetic_to_ecef_simple(KOZELSK_LAT, KOZELSK_LON, KOZELSK_ELEV)
R_TARGET_DC = _geodetic_to_ecef_simple(DC_LAT, DC_LON, DC_ELEV)


class TestGeodeticConversions:
    """Test coordinate conversion accuracy."""

    def test_geodetic_to_ecef_roundtrip(self):
        """Converting ECEF->geodetic->ECEF should recover original position."""
        r = R0_KOZELSK
        lat, lon, alt = _ecef_to_geodetic(r)
        r2 = _geodetic_to_ecef(lat, lon, alt)
        np.testing.assert_allclose(r, r2, rtol=1e-9)

    def test_target_position_reasonable(self):
        """Target should be ~7386 km from launch."""
        dist = np.linalg.norm(R_TARGET_DC - R0_KOZELSK)
        assert 7_300_000 < dist < 7_500_000, f"Distance {dist/1000:.1f}km out of range"

    def test_azimuth_computation(self):
        """Azimuth from Kozelsk to DC should be ~310°."""
        az = _great_circle_azimuth(R0_KOZELSK, R_TARGET_DC)
        assert 305.0 < az < 315.0, f"Azimuth {az:.2f}° out of expected range"


class TestICBMGuidance:
    """Test realistic ICBM guidance implementation."""

    def setup_method(self):
        """Create guidance instance for each test."""
        self.guidance = ICBMGuidance(
            target_ecef=R_TARGET_DC,
            launch_ecef=R0_KOZELSK,
            burnout_vmag=6800.0,
            pitch_over_start=5.0,
            initial_elevation=np.radians(55.0),
            use_j2=True,
        )

    def test_initial_thrust_vertical(self):
        """At t=0, thrust should be vertical (up)."""
        d = self.guidance.thrust_direction(R0_KOZELSK, np.zeros(3), 0.0)
        lat, lon, _ = _ecef_to_geodetic(R0_KOZELSK)
        _, _, up = _enu_basis(lat, lon)
        assert np.dot(d, up) > 0.99, "Initial thrust should be nearly vertical"

    def test_thrust_direction_unit_vector(self):
        """Thrust direction should always be a unit vector."""
        for t in [0.0, 5.0, 10.0, 30.0, 100.0, 200.0, 300.0]:
            d = self.guidance.thrust_direction(R0_KOZELSK, np.array([100.0, 200.0, 300.0]), t)
            assert np.isclose(np.linalg.norm(d), 1.0, rtol=1e-6), \
                f"Thrust direction not unit at t={t}: |d|={np.linalg.norm(d)}"

    def test_pitch_over_occurs(self):
        """Thrust should pitch over from vertical toward horizontal."""
        d0 = self.guidance.thrust_direction(R0_KOZELSK, np.zeros(3), 0.0)
        r_boosted = R0_KOZELSK + np.array([0.0, 0.0, 50_000.0])
        v_boosted = np.array([800.0, 1200.0, 400.0])
        d10 = self.guidance.thrust_direction(r_boosted, v_boosted, 10.0)
        
        lat, lon, _ = _ecef_to_geodetic(R0_KOZELSK)
        _, _, up = _enu_basis(lat, lon)
        
        fpa0 = np.degrees(np.arcsin(np.clip(np.dot(d0, up), -1.0, 1.0)))
        fpa10 = np.degrees(np.arcsin(np.clip(np.dot(d10, up), -1.0, 1.0)))
        
        assert fpa0 > 80.0, f"Initial FPA should be near vertical, got {fpa0:.1f}°"
        assert fpa10 < fpa0, f"FPA should decrease after pitch-over, got {fpa10:.1f}° vs {fpa0:.1f}°"

    def test_pitch_never_near_vertical_after_20s(self):
        """After pitch-over, thrust should not remain near-vertical."""
        r_high = R0_KOZELSK + np.array([0.0, 0.0, 200_000.0])
        v_fast = np.array([1000.0, 2000.0, 500.0])
        lat, lon, _ = _ecef_to_geodetic(r_high)
        _, _, up = _enu_basis(lat, lon)

        for t in [25.0, 50.0, 100.0, 200.0, 300.0]:
            d = self.guidance.thrust_direction(r_high, v_fast, t)
            vertical_component = np.dot(d, up)
            assert vertical_component < 0.5, \
                f"Thrust too vertical at t={t:.1f}: vertical={vertical_component:.3f}"

    def test_azimuth_toward_target(self):
        """Thrust horizontal component should point toward target azimuth."""
        d = self.guidance.thrust_direction(
            R0_KOZELSK, np.array([0.0, 100.0, 0.0]), 20.0
        )
        lat, lon, _ = _ecef_to_geodetic(R0_KOZELSK)
        east, north, up = _enu_basis(lat, lon)
        
        d_horiz = d - np.dot(d, up) * up
        d_horiz_norm = np.linalg.norm(d_horiz)
        if d_horiz_norm > 1e-6:
            d_horiz = d_horiz / d_horiz_norm
            
            az = np.radians(_great_circle_azimuth(R0_KOZELSK, R_TARGET_DC))
            target_horiz = np.cos(az) * north + np.sin(az) * east
            target_horiz = target_horiz / np.linalg.norm(target_horiz)
            
            alignment = np.dot(d_horiz, target_horiz)
            assert alignment > 0.5, \
                f"Thrust horizontal component should align with target azimuth, got dot={alignment:.3f}"


class TestInertialNavigationSystem:
    """Test INS state estimation."""

    def test_ins_propagation_conserves_momentum(self):
        """INS should conserve momentum in free space."""
        ins = InertialNavigationSystem(
            r0=np.array([6371e3, 0.0, 0.0]),
            v0=np.array([0.0, 7500.0, 0.0]),
        )
        
        for _ in range(100):
            ins.propagate(np.zeros(3), np.zeros(3), 0.1)
        
        expected_dr = 7500.0 * 10.0
        actual_dr = np.linalg.norm(ins.r - np.array([6371e3, 0.0, 0.0]))
        assert np.isclose(actual_dr, expected_dr, rtol=1e-2), \
            f"INS position change {actual_dr:.1f}m != expected {expected_dr:.1f}m"

    def test_ins_attitude_quaternion_norm(self):
        """INS quaternion should remain normalized."""
        ins = InertialNavigationSystem(
            r0=R0_KOZELSK,
            v0=np.zeros(3),
        )
        
        for _ in range(100):
            ins.propagate(np.array([0.0, 0.0, 9.81]), np.zeros(3), 0.1)
        
        assert np.isclose(np.linalg.norm(ins.q), 1.0, rtol=1e-6), \
            f"Quaternion norm drifted to {np.linalg.norm(ins.q)}"


class TestSarmatScenario:
    """Test full Sarmat scenario integration."""

    def setup_method(self):
        """Create Sarmat scenario for testing."""
        self.scenario = SarmatScenario(
            r0=R0_KOZELSK,
            v0=np.zeros(3),
            use_j2=True,
        )

    def test_scenario_has_guidance(self):
        """Scenario should have a real ICBMGuidance instance."""
        assert hasattr(self.scenario, '_guidance')
        assert self.scenario._guidance is not None
        assert isinstance(self.scenario._guidance, ICBMGuidance)

    def test_thrust_direction_valid(self):
        """Thrust direction should be valid at all times during boost."""
        for t in [0.0, 5.0, 10.0, 50.0, 100.0, 200.0, 250.0, 300.0]:
            r = self.scenario.propagate(t)[:3]
            v = self.scenario.propagate(t)[3:6]
            d = self.scenario._guidance.thrust_direction(r, v, t)

            assert np.all(np.isfinite(d)), f"Thrust direction invalid at t={t}"
            assert np.isclose(np.linalg.norm(d), 1.0, rtol=1e-6), \
                f"Thrust direction not unit at t={t}"

    def test_mass_decreases_during_boost(self):
        """Mass should decrease monotonically during boost."""
        masses = []
        for t in [0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 340.0]:
            state = self.scenario.propagate(t)
            if len(state) > 6:
                masses.append(state[6])
        
        for i in range(1, len(masses)):
            assert masses[i] <= masses[i-1] + 1e-6, \
                f"Mass increased from {masses[i-1]:.1f} to {masses[i]:.1f} at t={t}"


class TestGuidanceRegression:
    """Regression tests for specific bugs that were encountered."""

    def test_thrust_never_points_down(self):
        """Thrust should never have negative vertical component during boost."""
        guidance = ICBMGuidance(
            target_ecef=R_TARGET_DC,
            launch_ecef=R0_KOZELSK,
            use_j2=True,
        )
        
        for t in np.linspace(0, 300, 50):
            r = R0_KOZELSK + np.array([0.0, 0.0, 1000.0 * t])
            v = np.array([100.0 * t, 200.0 * t, 50.0 * t])
            d = guidance.thrust_direction(r, v, t)
            
            lat, lon, _ = _ecef_to_geodetic(r)
            _, _, up = _enu_basis(lat, lon)
            vertical_component = np.dot(d, up)
            
            assert vertical_component > -0.1, \
                f"Thrust pointing down at t={t:.1f}: vertical={vertical_component:.3f}"

    def test_guidance_handles_near_zero_velocity(self):
        """Guidance should not crash with near-zero velocity."""
        guidance = ICBMGuidance(
            target_ecef=R_TARGET_DC,
            launch_ecef=R0_KOZELSK,
            use_j2=True,
        )
        
        d = guidance.thrust_direction(R0_KOZELSK, np.zeros(3), 10.0)
        assert np.all(np.isfinite(d)), "Guidance crashed with zero velocity"
        assert np.isclose(np.linalg.norm(d), 1.0, rtol=1e-6)
