"""
GPU-accelerated batch utilities for full-fidelity operations.

This module provides CuPy-accelerated wrappers around CPU physics functions
for embarrassingly parallel workloads.  All operations preserve exact CPU
physics fidelity — the GPU is used only for vectorization, not for replacing
adaptive integration or event detection.

Typical use cases:
- Batch geodetic/ECEF coordinate conversions over thousands of sites
- Vectorized sensor coverage mapping
- Batch Monte Carlo result aggregation and statistical analysis
- GPR/surrogate training acceleration (future)
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import cupy as cp
    _HAVE_CUDA = True
except ImportError:
    _HAVE_CUDA = False


def is_gpu_available() -> bool:
    """Check if CUDA GPU acceleration is available."""
    if not _HAVE_CUDA:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Geodetic / ECEF batch operations
# ---------------------------------------------------------------------------

def batch_geodetic_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray,
                           alt_m: np.ndarray = None) -> np.ndarray:
    """Vectorized geodetic-to-ECEF conversion on GPU.

    Parameters
    ----------
    lat_deg : (N,) array-like
        Latitude in degrees.
    lon_deg : (N,) array-like
        Longitude in degrees.
    alt_m : (N,) array-like, optional
        Altitude in metres.  Defaults to 0.

    Returns
    -------
    ecef : (N, 3) ndarray
        ECEF coordinates in metres.
    """
    if alt_m is None:
        alt_m = np.zeros_like(lat_deg)

    if is_gpu_available():
        lat = cp.asarray(lat_deg, dtype=cp.float64)
        lon = cp.asarray(lon_deg, dtype=cp.float64)
        alt = cp.asarray(alt_m, dtype=cp.float64)
        return cp.asnumpy(_batch_geodetic_to_ecef_gpu(lat, lon, alt))

    # CPU fallback
    return _batch_geodetic_to_ecef_cpu(lat_deg, lon_deg, alt_m)


def _batch_geodetic_to_ecef_cpu(lat_deg, lon_deg, alt_m):
    """CPU reference implementation using project_icarus physics."""
    from ..scenarios.presets import geodetic_to_ecef as cpu_geodetic_to_ecef
    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    alt = np.asarray(alt_m, dtype=float)
    ecef = np.empty((lat.size, 3), dtype=float)
    for i in range(lat.size):
        ecef[i] = cpu_geodetic_to_ecef(lat[i], lon[i], alt[i])
    return ecef


def _batch_geodetic_to_ecef_gpu(lat, lon, alt):
    """GPU implementation of geodetic-to-ECEF."""
    _WGS84_A = 6378137.0
    _WGS84_F = 1.0 / 298.257223563
    _WGS84_E2 = (_WGS84_A**2 - (_WGS84_A * (1.0 - _WGS84_F))**2) / _WGS84_A**2

    lat_rad = cp.radians(lat)
    lon_rad = cp.radians(lon)
    sin_lat = cp.sin(lat_rad)
    cos_lat = cp.cos(lat_rad)
    sin_lon = cp.sin(lon_rad)
    cos_lon = cp.cos(lon_rad)
    N = _WGS84_A / cp.sqrt(1.0 - _WGS84_E2 * sin_lat**2)
    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1.0 - _WGS84_E2) + alt) * sin_lat
    return cp.stack([x, y, z], axis=1)


# ---------------------------------------------------------------------------
# Batch ECEF distance / coverage operations
# ---------------------------------------------------------------------------

def batch_ecef_distances(sites_ecef: np.ndarray, targets_ecef: np.ndarray) -> np.ndarray:
    """Compute pairwise ECEF distances between sites and targets.

    Parameters
    ----------
    sites_ecef : (N, 3) array-like
        Site positions in ECEF.
    targets_ecef : (M, 3) array-like
        Target positions in ECEF.

    Returns
    -------
    distances : (N, M) ndarray
        Pairwise distances in metres.
    """
    sites = np.asarray(sites_ecef, dtype=float)
    targets = np.asarray(targets_ecef, dtype=float)

    if is_gpu_available():
        d_sites = cp.asarray(sites)
        d_targets = cp.asarray(targets)
        diff = d_sites[:, None, :] - d_targets[None, :, :]
        dist = cp.linalg.norm(diff, axis=2)
        return cp.asnumpy(dist)

    # CPU fallback
    diff = sites[:, None, :] - targets[None, :, :]
    return np.linalg.norm(diff, axis=2)


def batch_coverage_mask(sites_ecef: np.ndarray, targets_ecef: np.ndarray,
                        max_range: float) -> np.ndarray:
    """Compute boolean coverage mask: which targets are within range of which sites.

    Parameters
    ----------
    sites_ecef : (N, 3) array-like
        Site positions in ECEF.
    targets_ecef : (M, 3) array-like
        Target positions in ECEF.
    max_range : float
        Maximum detection range in metres.

    Returns
    -------
    mask : (N, M) ndarray of bool
        True if target j is within range of site i.
    """
    dist = batch_ecef_distances(sites_ecef, targets_ecef)
    return dist <= max_range


# ---------------------------------------------------------------------------
# Batch Monte Carlo result analysis
# ---------------------------------------------------------------------------

def batch_miss_distance_stats(miss_distances: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute statistics across multiple Monte Carlo result sets.

    Parameters
    ----------
    miss_distances : list of (N,) array-like
        Miss distance arrays from different campaigns/configurations.

    Returns
    -------
    stats : dict
        Dictionary with 'mean', 'std', 'p50', 'p90', 'p99', 'max' arrays.
    """
    stacked = np.asarray(miss_distances, dtype=float)  # (n_configs, n_trials)
    return {
        "mean": np.nanmean(stacked, axis=1),
        "std": np.nanstd(stacked, axis=1),
        "p50": np.nanpercentile(stacked, 50, axis=1),
        "p90": np.nanpercentile(stacked, 90, axis=1),
        "p99": np.nanpercentile(stacked, 99, axis=1),
        "max": np.nanmax(stacked, axis=1),
    }


def batch_kill_probability(kill_assessments: List[np.ndarray]) -> np.ndarray:
    """Compute kill probability across multiple Monte Carlo result sets.

    Parameters
    ----------
    kill_assessments : list of (N,) array-like
        Boolean kill assessment arrays from different campaigns/configurations.

    Returns
    -------
    p_kill : (n_configs,) ndarray
        Kill probability for each configuration.
    """
    stacked = np.asarray(kill_assessments, dtype=bool)
    return np.nanmean(stacked, axis=1)


# ---------------------------------------------------------------------------
# Batch sensor model operations (future)
# ---------------------------------------------------------------------------

def batch_sensor_detection_probability(ranges: np.ndarray, rcs: np.ndarray,
                                       pd_params: Dict[str, float]) -> np.ndarray:
    """Vectorized sensor detection probability calculation.

    Computes Pd vs range/RCS for multiple sensor-target pairs in parallel.

    Parameters
    ----------
    ranges : (N, M) array-like
        Slant ranges from N sensors to M targets.
    rcs : (M,) array-like
        Radar cross-section of each target.
    pd_params : dict
        Sensor detection parameters (range_max, rcs_min, etc.).

    Returns
    -------
    pd : (N, M) ndarray
        Detection probabilities.
    """
    ranges = np.asarray(ranges, dtype=float)
    rcs = np.asarray(rcs, dtype=float)

    if is_gpu_available():
        d_ranges = cp.asarray(ranges)
        d_rcs = cp.asarray(rcs)
        pd = _batch_pd_gpu(d_ranges, d_rcs, pd_params)
        return cp.asnumpy(pd)

    return _batch_pd_cpu(ranges, rcs, pd_params)


def _batch_pd_cpu(ranges, rcs, params):
    """CPU reference: logistic Pd model."""
    range_max = params.get("range_max", 500e3)
    rcs_min = params.get("rcs_min", 0.1)
    rcs_max = params.get("rcs_max", 10.0)
    pd_max = params.get("pd_max", 0.9)

    range_factor = np.clip(1.0 - ranges / range_max, 0.0, 1.0)
    rcs_factor = np.clip((rcs - rcs_min) / (rcs_max - rcs_min), 0.0, 1.0)
    rcs_factor = rcs_factor[None, :]  # broadcast over sensors

    pd = pd_max * range_factor * rcs_factor
    return np.clip(pd, 0.0, 1.0)


def _batch_pd_gpu(d_ranges, d_rcs, params):
    """GPU implementation of logistic Pd model."""
    range_max = params.get("range_max", 500e3)
    rcs_min = params.get("rcs_min", 0.1)
    rcs_max = params.get("rcs_max", 10.0)
    pd_max = params.get("pd_max", 0.9)

    range_factor = cp.clip(1.0 - d_ranges / range_max, 0.0, 1.0)
    rcs_factor = cp.clip((d_rcs - rcs_min) / (rcs_max - rcs_min), 0.0, 1.0)
    rcs_factor = rcs_factor[None, :]

    pd = pd_max * range_factor * rcs_factor
    return cp.clip(pd, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def gpu_array(x: np.ndarray) -> Any:
    """Transfer array to GPU if available, else return CPU array."""
    if is_gpu_available():
        return cp.asarray(x, dtype=cp.float64)
    return np.asarray(x, dtype=float)


def to_cpu(x: Any) -> np.ndarray:
    """Transfer array to CPU if on GPU."""
    if _HAVE_CUDA and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)
