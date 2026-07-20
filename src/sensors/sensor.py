"""Probabilistic sensor detection, tracking, and network fusion.

Implements the Phase 5A baseline sensor layer for Project Icarus:

* ``Sensor``            - single-site radar/IR detection model (Pd vs range/RCS/alt)
* ``Detection``        - a noisy line-of-sight (range/az/el) measurement
* ``Track``            - an EKF maintainer over successive detections
* ``SensorNetwork``    - aggregates sites, produces detections, fuses into tracks
* ``coverage``         - geodetic detection-footprint maps over ``locations.yml``

All models are analytic/OSINT-approximate and fully synthetic; no controlled
data is used. The module depends only on numpy and reuses the project's
geodetic helpers so it slots into the existing ECEF coordinate frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from src.dynamics.coordinate_systems import ecef_to_geodetic, geodetic_to_ecef


@dataclass
class Sensor:
    """A single detection site (radar or IR).

    Parameters
    ----------
    name
        Site identifier.
    lat_deg, lon_deg, alt_m
        Geodetic location of the sensor.
    max_range_m
        Kinematic detection range (m) at the reference RCS.
    reference_rcs_m2
        RCS (m^2) the ``max_range_m`` is quoted for.
    min_elevation_deg
        Earth-mask: targets below this elevation (deg) are not seen.
    range_std_m, angle_std_deg
        Measurement noise (1-sigma) for range and angles.
    p_fa
        Single-look false-alarm probability (Poisson clutter rate per look).
    """

    name: str
    lat_deg: float
    lon_deg: float
    alt_m: float = 0.0
    max_range_m: float = 2000e3
    reference_rcs_m2: float = 1.0
    min_elevation_deg: float = 3.0
    range_std_m: float = 500.0
    angle_std_deg: float = 0.3
    p_fa: float = 1e-6

    @property
    def ecef(self) -> np.ndarray:
        return np.asarray(
            geodetic_to_ecef(self.lat_deg, self.lon_deg, self.alt_m),
            dtype=float,
        )

    def probability_of_detection(
        self, target_ecef: np.ndarray, rcs_m2: float = 1.0
    ) -> float:
        """Range/RCS-scaled detection probability (logistic ramp near max)."""
        target_ecef = np.asarray(target_ecef, dtype=float)
        slant = float(np.linalg.norm(target_ecef - self.ecef))
        if slant <= 1.0:
            return 1.0
        # RCS scaling: range scales as RCS^(1/4) (radar equation).
        rcs_factor = (abs(rcs_m2) / self.reference_rcs_m2) ** 0.25
        effective_range = self.max_range_m * rcs_factor
        if slant >= effective_range:
            return 0.0
        # Smooth logistic transition over the final 15% of the range.
        margin = (effective_range - slant) / (0.15 * effective_range)
        p = 1.0 / (1.0 + np.exp(-margin))
        return float(min(0.999, max(1e-4, p)))

    def elevation_deg(self, target_ecef: np.ndarray) -> float:
        """Local elevation angle (deg) of ``target_ecef`` above the sensor horizon."""
        target_ecef = np.asarray(target_ecef, dtype=float)
        los = target_ecef - self.ecef
        slant = float(np.linalg.norm(los))
        if slant <= 1.0:
            return 90.0
        lat, lon, _ = ecef_to_geodetic(self.ecef)
        up = np.array([
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ])
        return float(np.degrees(np.arcsin(np.clip(np.dot(los, up) / slant, -1, 1))))

    def visible(self, target_ecef: np.ndarray) -> bool:
        return self.elevation_deg(target_ecef) >= self.min_elevation_deg

    def detect(
        self,
        target_ecef: np.ndarray,
        rcs_m2: float = 1.0,
        rng: Optional[np.random.Generator] = None,
        t: float = 0.0,
    ) -> Optional["Detection"]:
        """Sample a detection. Returns ``None`` on miss/earth-mask."""
        rng = rng or np.random.default_rng()
        if not self.visible(target_ecef):
            return None
        p = self.probability_of_detection(target_ecef, rcs_m2)
        if rng.random() > p:
            return None
        return self._noisy_measurement(target_ecef, rng, t)

    def _noisy_measurement(
        self, target_ecef: np.ndarray, rng: np.random.Generator, t: float
    ) -> "Detection":
        target_ecef = np.asarray(target_ecef, dtype=float)
        los = target_ecef - self.ecef
        slant = float(np.linalg.norm(los))
        los_hat = los / slant
        lat, lon, _ = ecef_to_geodetic(self.ecef)
        up = np.array([
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ])
        east = np.array([-np.sin(lon), np.cos(lon), 0.0])
        north = np.cross(up, east)
        az = np.degrees(np.arctan2(np.dot(los_hat, east), np.dot(los_hat, north)))
        el = np.degrees(np.arcsin(np.clip(np.dot(los_hat, up), -1, 1)))
        r_std = self.range_std_m
        a_std = np.radians(self.angle_std_deg)
        r_meas = slant + rng.normal(0.0, r_std)
        az_meas = az + np.degrees(rng.normal(0.0, a_std))
        el_meas = el + np.degrees(rng.normal(0.0, a_std))
        return Detection(
            sensor_name=self.name,
            sensor_ecef=self.ecef.copy(),
            t=t,
            range_m=r_meas,
            azimuth_deg=az_meas,
            elevation_deg=el_meas,
            range_std_m=r_std,
            angle_std_rad=a_std,
        )


@dataclass
class Detection:
    """A single noisy line-of-sight measurement in the sensor ENU frame."""

    sensor_name: str
    sensor_ecef: np.ndarray
    t: float
    range_m: float
    azimuth_deg: float
    elevation_deg: float
    range_std_m: float
    angle_std_rad: float

    def los_hat(self) -> np.ndarray:
        """Unit line-of-sight vector in ECEF from sensor to target."""
        a = np.radians(self.azimuth_deg)
        e = np.radians(self.elevation_deg)
        lat, lon, _ = ecef_to_geodetic(self.sensor_ecef)
        up = np.array([
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ])
        east = np.array([-np.sin(lon), np.cos(lon), 0.0])
        north = np.cross(up, east)
        return (
            np.cos(e) * (np.cos(a) * north + np.sin(a) * east)
            + np.sin(e) * up
        )

    def estimated_ecef(self) -> np.ndarray:
        return self.sensor_ecef + self.range_m * self.los_hat()


@dataclass
class Track:
    """Persisting EKF tracker over a sequence of detections.

    State is 6-DOF position/velocity in ECEF. Measurement is the 3-D
    sensor-frame LOS vector (range, east, north) expressed relative to the
    sensor, linearised about the current estimate.
    """

    track_id: int
    x: np.ndarray  # 6-vector [rx, ry, rz, vx, vy, vz]
    P: np.ndarray  # 6x6 covariance
    last_t: float
    hits: int = 1
    misses: int = 0

    # Process noise (m^2/s^3 position, m^2/s velocity) - light constant-velocity.
    q_pos: float = 1e2
    q_vel: float = 1e-2

    def predict(self, t: float):
        dt = max(0.0, t - self.last_t)
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        self.x = F @ self.x
        Q = np.zeros((6, 6))
        Q[0, 0] = Q[1, 1] = Q[2, 2] = self.q_pos * dt**3 / 3
        Q[0, 3] = Q[3, 0] = self.q_pos * dt**2 / 2
        Q[1, 4] = Q[4, 1] = self.q_pos * dt**2 / 2
        Q[2, 5] = Q[5, 2] = self.q_pos * dt**2 / 2
        Q[3, 3] = Q[4, 4] = Q[5, 5] = self.q_vel * dt
        self.P = F @ self.P @ F.T + Q
        self.last_t = t

    def update(self, det: Detection):
        lat, lon, _ = ecef_to_geodetic(det.sensor_ecef)
        up = np.array([
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ])
        east = np.array([-np.sin(lon), np.cos(lon), 0.0])
        north = np.cross(up, east)
        s_ecef = det.sensor_ecef

        # Measurement model h(x) = (range, azimuth, elevation) in the sensor
        # ENU frame; measured values come from the noisy detection.
        def h(p):
            d = p - s_ecef
            Rr = float(np.linalg.norm(d))
            los = d / Rr
            az = np.arctan2(np.dot(los, east), np.dot(los, north))
            el = np.arcsin(np.clip(np.dot(los, up), -1.0, 1.0))
            return np.array([Rr, az, el])

        z = np.array([
            det.range_m,
            np.radians(det.azimuth_deg),
            np.radians(det.elevation_deg),
        ])
        hx = h(self.x[:3])
        # Finite-difference Jacobian d(h)/d(position).
        eps = 1.0
        H = np.zeros((3, 6))
        for i in range(3):
            dp = self.x[:3].copy()
            dp[i] += eps
            H[:3, i] = (h(dp) - hx) / eps

        Rn = np.diag([
            det.range_std_m**2,
            (det.range_m * det.angle_std_rad) ** 2,
            (det.range_m * det.angle_std_rad) ** 2,
        ])
        S = H @ self.P @ H.T + Rn
        K = self.P @ H.T @ np.linalg.inv(S)
        y = z - hx
        # Wrap angular innovations to [-pi, pi].
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
        y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        self.hits += 1

    @property
    def position(self) -> np.ndarray:
        return self.x[:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:]

    @property
    def pos_cov(self) -> np.ndarray:
        return self.P[:3, :3]


class SensorNetwork:
    """Aggregates sensor sites, generates detections, and fuses them into tracks.

    The network is driven by ground-truth target states (ECEF position) at a
    sequence of times; it returns per-scan detections and maintains a fused
    track table via nearest-neighbour association.
    """

    def __init__(
        self,
        sensors: Sequence[Sensor],
        association_gate_sigma: float = 3.0,
        drop_after_misses: int = 3,
        rng: Optional[np.random.Generator] = None,
    ):
        self.sensors = list(sensors)
        self.gate = association_gate_sigma
        self.drop_after = drop_after_misses
        self._rng = rng or np.random.default_rng()
        self.tracks: List[Track] = []
        self._next_id = 1
        self.all_detections: List[Detection] = []

    def scan(
        self,
        targets: Sequence[np.ndarray],
        rcs_m2: float = 1.0,
        t: float = 0.0,
    ) -> List[Detection]:
        """Collect detections across all sites for the given target states."""
        dets: List[Detection] = []
        for tgt in targets:
            for s in self.sensors:
                d = s.detect(tgt, rcs_m2, self._rng, t)
                if d is not None:
                    dets.append(d)
                    self.all_detections.append(d)
        self._fuse(dets, t)
        return dets

    def _fuse(self, dets: List[Detection], t: float):
        for trk in self.tracks:
            trk.predict(t)
        for d in dets:
            cand = None
            best = np.inf
            for trk in self.tracks:
                res = np.linalg.norm(d.estimated_ecef() - trk.position)
                if res < best:
                    best = res
                    cand = trk
            if cand is not None and best < self.gate * 1e6:
                cand.update(d)
            else:
                self.tracks.append(
                    Track(
                        track_id=self._next_id,
                        x=np.concatenate([d.estimated_ecef(), np.zeros(3)]),
                        P=np.diag([1e6, 1e6, 1e6, 1e4, 1e4, 1e4]),
                        last_t=t,
                    )
                )
                self._next_id += 1
        # Age tracks that received no detection this scan.
        if dets:
            for trk in list(self.tracks):
                trk.misses += 1
                if trk.misses > self.drop_after:
                    self.tracks.remove(trk)

    def confirmed_tracks(self, min_hits: int = 2) -> List[Track]:
        return [t for t in self.tracks if t.hits >= min_hits]

    def coverage_mask(
        self,
        grid_lat: np.ndarray,
        grid_lon: np.ndarray,
        alt_m: float = 100e3,
    ) -> np.ndarray:
        """Boolean mask: which grid cells are visible to >=1 sensor.

        ``grid_lat``/``grid_lon`` are 1-D arrays; returns a 2-D
        (len(lat), len(lon)) visibility mask at a fixed altitude ``alt_m``.
        """
        lat = np.asarray(grid_lat, dtype=float)
        lon = np.asarray(grid_lon, dtype=float)
        mask = np.zeros((lat.size, lon.size), dtype=bool)
        for i, la in enumerate(lat):
            for j, lo in enumerate(lon):
                ecef = np.asarray(geodetic_to_ecef(la, lo, alt_m), dtype=float)
                if any(s.visible(ecef) for s in self.sensors):
                    mask[i, j] = True
        return mask
