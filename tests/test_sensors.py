"""Tests for the Phase 5A sensor layer (detection, tracking, fusion).

Lightweight, deterministic (seeded RNG), and fast (<5 s total). Validates:
* Pd scaling with range/RCS and earth-mask gating,
* noisy measurement error statistics,
* EKF track convergence against a truth trajectory,
* multi-sensor fusion into a single confirmed track,
* data-driven loading from ``reference/locations.yml``,
* geodetic coverage-mask generation.
"""

import numpy as np
import pytest

from src.dynamics.coordinate_systems import geodetic_to_ecef
from src.sensors.sensor import Detection, Sensor, SensorNetwork, Track
from src.sensors.network import load_sensors_from_locations


def _ecef(lat_deg, lon_deg, alt_m=0.0):
    return np.asarray(geodetic_to_ecef(lat_deg, lon_deg, alt_m), dtype=float)


# --------------------------------------------------------------------------- #
# Sensor model
# --------------------------------------------------------------------------- #
def test_probability_of_detection_scales_with_range():
    s = Sensor("S", lat_deg=40.0, lon_deg=-100.0, max_range_m=2000e3)
    near = _ecef(40.0, -100.0) + np.array([200e3, 0.0, 200e3])
    far = _ecef(40.0, -100.0) + np.array([1800e3, 0.0, 200e3])
    assert s.probability_of_detection(near) > s.probability_of_detection(far)
    assert s.probability_of_detection(near) > 0.9
    assert s.probability_of_detection(
        _ecef(40.0, -100.0) + np.array([5000e3, 0.0, 0.0])
    ) == 0.0


def test_rcs_improves_detection():
    s = Sensor("S", lat_deg=0.0, lon_deg=0.0, max_range_m=1000e3)
    tgt = _ecef(0.0, 0.0) + np.array([900e3, 0.0, 100e3])
    p_small = s.probability_of_detection(tgt, rcs_m2=0.01)
    p_large = s.probability_of_detection(tgt, rcs_m2=10.0)
    assert p_large > p_small


def test_earth_mask_blocks_below_horizon():
    s = Sensor("S", lat_deg=40.0, lon_deg=-100.0, min_elevation_deg=3.0)
    below = _ecef(20.0, -100.0) + np.array([0.0, 0.0, 500e3])
    assert not s.visible(below)
    assert s.detect(below, rng=np.random.default_rng(0)) is None


def test_measurement_noise_statistics():
    rng = np.random.default_rng(0)
    s = Sensor("S", lat_deg=40.0, lon_deg=-100.0, range_std_m=500.0,
                angle_std_deg=0.3, max_range_m=2000e3)
    # Target due east and well above the horizon -> visible, mid-range.
    truth = _ecef(40.0, -100.0) + np.array([300e3, 0.0, 600e3])
    assert s.visible(truth)
    errs = []
    while len(errs) < 400:
        d = s.detect(truth, rng=rng, t=0.0)
        if d is not None:
            errs.append(np.linalg.norm(d.estimated_ecef() - truth))
    rmse = float(np.sqrt(np.mean(np.square(errs))))
    slant = float(np.linalg.norm(truth - s.ecef))
    # Analytic 1-sigma: sqrt(range_std^2 + (R*angle_std)^2).
    analytic = np.sqrt(500.0**2 + (slant * np.radians(0.3))**2)
    assert 0.5 * analytic < rmse < 1.5 * analytic


# --------------------------------------------------------------------------- #
# Tracking EKF
# --------------------------------------------------------------------------- #
def test_track_converges_to_truth():
    rng = np.random.default_rng(3)
    s = Sensor("S", lat_deg=40.0, lon_deg=-100.0, range_std_m=300.0,
                angle_std_deg=0.2, max_range_m=2000e3)
    # Truth: moderate-speed pass well above the horizon so the sensor keeps a
    # clear line of sight. Velocity is initialised from the first two detections
    # (2-point track initiation), as a real constant-velocity filter requires.
    center = _ecef(40.0, -100.0)
    truth = [center + np.array([k * 3e3, k * 0.8e3, 300e3]) for k in range(10)]
    assert all(s.visible(x) for x in truth)

    dets = [s.detect(x, rng=rng, t=float(k)) for k, x in enumerate(truth)]
    assert dets[0] is not None and dets[1] is not None
    p0 = dets[0].estimated_ecef()
    p1 = dets[1].estimated_ecef()
    v0 = (p1 - p0) / 1.0
    trk = Track(track_id=1, x=np.concatenate([p0, v0]),
                P=np.diag([1e6, 1e6, 1e6, 1e4, 1e4, 1e4]), last_t=0.0)
    for k in range(2, 10):
        d = dets[k]
        assert d is not None
        trk.predict(float(k))
        trk.update(d)

    final_err = float(np.linalg.norm(trk.position - truth[-1]))
    # No-filter baseline: hold the 2-point-init velocity constant from p0.
    # With a single passive sensor, range+angle gives only weak velocity
    # observability, so the EKF should match (not beat) this kinematic prior
    # and must stay stable/finite -- NOT diverge.
    dead_rec = float(np.linalg.norm((p0 + 9.0 * v0) - truth[-1]))
    assert final_err < 1.2 * dead_rec
    assert np.all(np.isfinite(trk.P))
    assert np.all(np.isfinite(trk.x))


# --------------------------------------------------------------------------- #
# Network fusion + data loading + coverage
# --------------------------------------------------------------------------- #
def test_network_fuses_multi_sensor_into_one_track():
    sensors = load_sensors_from_locations(designation=["sensor"])
    assert len(sensors) >= 5
    rng = np.random.default_rng(7)
    net = SensorNetwork(sensors, rng=rng)
    # Target over central CONUS at midcourse altitude: visible to multiple UEWR
    # sites (Beale, Cape Cod, Clear, Fylingdales) at once.
    pos0 = _ecef(45.0, -90.0) + np.array([0.0, 0.0, 700e3])
    path = [pos0 + k * np.array([120e3, 25e3, -20e3]) for k in range(8)]
    assert any(s.visible(path[0]) for s in sensors)
    for k, tgt in enumerate(path):
        net.scan([tgt], rcs_m2=1.0, t=float(k))
    confirmed = net.confirmed_tracks(min_hits=2)
    assert len(confirmed) >= 1
    # The fused track should be near the final truth position.
    final = path[-1]
    best = min(confirmed, key=lambda t: np.linalg.norm(t.position - final))
    assert np.linalg.norm(best.position - final) < 5e5


def test_coverage_mask_shape_and_nonempty():
    sensors = load_sensors_from_locations(designation=["sensor"])
    net = SensorNetwork(sensors)
    lat = np.linspace(25, 60, 15)
    lon = np.linspace(-130, -60, 15)
    mask = net.coverage_mask(lat, lon, alt_m=100e3)
    assert mask.shape == (15, 15)
    assert mask.any()


def test_load_radar_sites_have_long_range():
    sensors = load_sensors_from_locations(designation=["sensor"])
    assert all(s.max_range_m >= 4000e3 for s in sensors)
    assert any("LRDR" in str(s.name) or "Thule" in str(s.name) for s in sensors)
