from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import numpy as np

from .target_factory import (
    BallisticScenario,
    FOBSScenario,
    HGVScenario,
    SuppressedScenario,
    SwarmScenario,
    MU_EARTH,
    R_EARTH,
)
from .scenario import EngagementScenario


# ---------------------------------------------------------------------------
# Geodetic / WGS84 helpers
# ---------------------------------------------------------------------------

# WGS84 ellipsoid parameters
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_B = _WGS84_A * (1.0 - _WGS84_F)
_WGS84_E2 = (_WGS84_A**2 - _WGS84_B**2) / _WGS84_A**2


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
    """Convert geodetic (WGS84) latitude/longitude/height to ECEF position."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat**2)
    x = (N + alt_m) * cos_lat * cos_lon
    y = (N + alt_m) * cos_lat * sin_lon
    z = (N * (1.0 - _WGS84_E2) + alt_m) * sin_lat
    return np.array([x, y, z])


def ecef_to_geodetic(r_ecef: np.ndarray) -> Tuple[float, float, float]:
    """Convert ECEF position to geodetic (lat_deg, lon_deg, alt_m) using WGS84."""
    x, y, z = r_ecef
    lon = np.degrees(np.arctan2(y, x))
    p = np.sqrt(x**2 + y**2)
    
    if p < 1e-6:
        lat = 90.0 if z > 0 else -90.0
        return float(lat), float(lon), float(abs(z) - _WGS84_B)
    
    # Bowring's method
    b = _WGS84_B
    e_prime2 = _WGS84_E2 / (1.0 - _WGS84_E2)
    theta = np.arctan2(z * _WGS84_A, p * b)
    lat = np.arctan2(
        z + e_prime2 * b * np.sin(theta)**3,
        p - _WGS84_E2 * _WGS84_A * np.cos(theta)**3,
    )
    
    sin_lat = np.sin(lat)
    N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat**2)
    alt = p / np.cos(lat) - N
    
    return float(np.degrees(lat)), float(lon), float(alt)


# ---------------------------------------------------------------------------
# Interceptor launch site presets (USA / US Indo-Pacific)
# ---------------------------------------------------------------------------

_INTERCEPTOR_PRESETS: Dict[str, np.ndarray] = {
    # --- Geodetic presets ---
    # Vandenberg SFB, CA (~34.7°N, 120.6°W, 0 m)
    "vandenberg": geodetic_to_ecef(34.7, -120.6, 0.0),
    # Cape Canaveral SFS, FL (~28.4°N, 80.6°W, 0 m)
    "cape_canaveral": geodetic_to_ecef(28.4, -80.6, 0.0),
    # Kwajalein Atoll, Marshall Islands (~8.7°N, 167.7°E, 0 m)
    "kwajalein": geodetic_to_ecef(8.7, 167.7, 0.0),
    # Schriever SFB, CO (~38.8°N, 104.5°W, 0 m)
    "schriever": geodetic_to_ecef(38.8, -104.5, 0.0),
    # Fort Greely, AK (~63.9°N, 147.6°W, 0 m)
    "fort_greely": geodetic_to_ecef(63.9, -147.6, 0.0),
    # Clear SFS, AK (~64.3°N, 149.2°W, 0 m)
    "clear_sfs": geodetic_to_ecef(64.3, -149.2, 0.0),
    # Custom geodetic entry point (lat, lon, alt)
    "custom_geodetic": np.zeros(3),
}


def get_interceptor_presets() -> Dict[str, np.ndarray]:
    """Return a copy of available interceptor launch site presets."""
    return {k: v.copy() for k, v in _INTERCEPTOR_PRESETS.items()}


def interceptor_preset(name: str) -> np.ndarray:
    """Return a single interceptor launch site preset by name."""
    if name not in _INTERCEPTOR_PRESETS:
        raise KeyError(f"Unknown interceptor preset: {name}. Available: {list(_INTERCEPTOR_PRESETS.keys())}")
    return _INTERCEPTOR_PRESETS[name].copy()


def set_interceptor_geodetic(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
    """Build a custom interceptor launch site from geodetic coordinates."""
    return geodetic_to_ecef(lat_deg, lon_deg, alt_m)


# ---------------------------------------------------------------------------
# Target launch presets
# ---------------------------------------------------------------------------

@dataclass
class TargetPreset:
    """Convenience wrapper bundling a target scenario and engagement metadata."""
    name: str
    target: Any  # TargetScenario
    engagement: EngagementScenario
    description: str = ""


# Pre-built target presets ------------------------------------------------

_TARGET_PRESETS: Dict[str, TargetPreset] = {}


def _register(name: str, target, engagement, description=""):
    _TARGET_PRESETS[name] = TargetPreset(
        name=name,
        target=target,
        engagement=engagement,
        description=description,
    )


# --- Ballistic family ---

_register(
    "ballistic_short_range",
    BallisticScenario(
        r0=np.array([R_EARTH, 0.0, 0.0]),
        v0=np.array([0.0, 800.0, 0.0]),
    ),
    EngagementScenario(engagement_end=120.0),
    "Short-range ballistic, ~45° elevation",
)

_register(
    "ballistic_medium_range",
    BallisticScenario(
        r0=np.array([R_EARTH, 0.0, 0.0]),
        v0=np.array([0.0, 1500.0, 500.0]),
    ),
    EngagementScenario(engagement_end=300.0),
    "Medium-range ballistic, mixed azimuth/elevation",
)

_register(
    "ballistic_long_range",
    BallisticScenario(
        r0=np.array([R_EARTH, 0.0, 0.0]),
        v0=np.array([0.0, 2500.0, 1000.0]),
    ),
    EngagementScenario(engagement_end=600.0),
    "Long-range ballistic, high loft trajectory",
)

_register(
    "ballistic_sea_launched",
    BallisticScenario(
        r0=np.array([R_EARTH + 0.0, 800e3, 0.0]),
        v0=np.array([0.0, 1200.0, 0.0]),
    ),
    EngagementScenario(engagement_end=300.0),
    "Sea-launched ballistic, 800 km offset",
)

_register(
    "ballistic_high_latitude",
    BallisticScenario(
        r0=np.array([R_EARTH * np.cos(np.radians(70.0)), 0.0, R_EARTH * np.sin(np.radians(70.0))]),
        v0=np.array([0.0, 1200.0, 0.0]),
    ),
    EngagementScenario(engagement_end=300.0),
    "High-latitude (70°N) ballistic launch",
)

# --- Geodetic ballistic presets ---

_register(
    "ballistic_from_vandenberg",
    BallisticScenario(
        r0=geodetic_to_ecef(34.7, -120.6, 0.0),
        v0=np.array([0.0, 1200.0, 0.0]),
    ),
    EngagementScenario(engagement_end=300.0),
    "Ballistic target from Vandenberg (34.7°N, 120.6°W)",
)

_register(
    "ballistic_from_kwajalein",
    BallisticScenario(
        r0=geodetic_to_ecef(8.7, 167.7, 0.0),
        v0=np.array([0.0, 1000.0, 0.0]),
    ),
    EngagementScenario(engagement_end=300.0),
    "Ballistic target from Kwajalein (8.7°N, 167.7°E)",
)

_register(
    "ballistic_from_svalbard",
    BallisticScenario(
        r0=geodetic_to_ecef(78.2, 15.6, 0.0),
        v0=np.array([0.0, 900.0, 0.0]),
    ),
    EngagementScenario(engagement_end=300.0),
    "Ballistic target from Svalbard (78.2°N, 15.6°E)",
)

# --- FOBS family ---

_register(
    "fobs_low_orbital",
    FOBSScenario.from_orbital_params(apoapsis_km=150.0, inclination_deg=20.0),
    EngagementScenario(engagement_end=900.0),
    "Low-orbital FOBS, 150 km apoapsis, 20° inclination",
)

_register(
    "fobs_high_orbital",
    FOBSScenario.from_orbital_params(apoapsis_km=400.0, inclination_deg=45.0),
    EngagementScenario(engagement_end=1800.0),
    "High-orbital FOBS, 400 km apoapsis, 45° inclination",
)

_register(
    "fobs_polar",
    FOBSScenario.from_orbital_params(apoapsis_km=300.0, inclination_deg=90.0),
    EngagementScenario(engagement_end=1500.0),
    "Polar FOBS, 300 km apoapsis, 90° inclination",
)

# --- HGV family ---

_register(
    "hgp_hypersonic_glide",
    HGVScenario.from_params(max_alt_km=70.0, lateral_range_km=1500.0, speed_mach=10.0),
    EngagementScenario(engagement_end=600.0),
    "Hypersonic glide vehicle, 70 km max alt, Mach 10",
)

_register(
    "hgp_skip_glide",
    HGVScenario.from_params(max_alt_km=90.0, lateral_range_km=3000.0, speed_mach=12.0),
    EngagementScenario(engagement_end=900.0),
    "Skip-glide HGV, 90 km max alt, Mach 12, 3000 km range",
)

_register(
    "hgp_maneuvering",
    HGVScenario.from_params(max_alt_km=60.0, lateral_range_km=800.0, speed_mach=8.0),
    EngagementScenario(engagement_end=400.0),
    "Maneuvering HGV, 60 km max alt, Mach 8",
)

# --- Suppressed / evasive family ---

_register(
    "suppressed_deep_dip",
    SuppressedScenario(
        r0=np.array([R_EARTH, 0.0, 0.0]),
        v0=np.array([0.0, 700.0, 700.0]),
        dip_alt_km=30.0,
        midcourse_maneuver_mag=80.0,
        maneuver_interval=20.0,
    ),
    EngagementScenario(engagement_end=300.0),
    "Suppressed target with deep dip to 30 km",
)

_register(
    "suppressed_high_altitude",
    SuppressedScenario(
        r0=np.array([R_EARTH + 20e3, 0.0, 0.0]),
        v0=np.array([0.0, 900.0, 900.0]),
        dip_alt_km=60.0,
        midcourse_maneuver_mag=40.0,
        maneuver_interval=25.0,
    ),
    EngagementScenario(engagement_end=300.0),
    "Suppressed target starting from 20 km altitude",
)

# --- Swarm family ---

_register(
    "swarm_tight",
    SwarmScenario.from_params(n_payloads=3, spread_deg=1.0, range_km=500.0),
    EngagementScenario(engagement_end=300.0),
    "Tight swarm, 3 payloads, 1° spread",
)

_register(
    "swarm_wide",
    SwarmScenario.from_params(n_payloads=5, spread_deg=5.0, range_km=1200.0),
    EngagementScenario(engagement_end=500.0),
    "Wide swarm, 5 payloads, 5° spread, 1200 km range",
)

_register(
    "swarm_sea_launched",
    SwarmScenario.from_params(n_payloads=4, spread_deg=2.0, range_km=800.0),
    EngagementScenario(engagement_end=400.0),
    "Sea-launched swarm, 4 payloads, 2° spread",
)

# --- Geodetic target presets: Russia / China / regional threats ---

# Russia
_register(
    "ballistic_target_moscow",
    BallisticScenario(
        r0=geodetic_to_ecef(55.8, 37.6, 0.0),
        v0=np.array([0.0, 1000.0, 0.0]),
    ),
    EngagementScenario(engagement_end=300.0),
    "Ballistic target from Moscow region (55.8°N, 37.6°E)",
)

_register(
    "ballistic_target_novosibirsk",
    BallisticScenario(
        r0=geodetic_to_ecef(55.0, 82.9, 0.0),
        v0=np.array([0.0, 900.0, 0.0]),
    ),
    EngagementScenario(engagement_end=300.0),
    "Ballistic target from Novosibirsk region (55.0°N, 82.9°E)",
)

_register(
    "ballistic_target_vladivostok",
    BallisticScenario(
        r0=geodetic_to_ecef(43.1, 131.9, 0.0),
        v0=np.array([0.0, 800.0, 0.0]),
    ),
    EngagementScenario(engagement_end=300.0),
    "Ballistic target from Vladivostok region (43.1°N, 131.9°E)",
)

_register(
    "ballistic_target_murmansk",
    BallisticScenario(
        r0=geodetic_to_ecef(68.9, 33.1, 0.0),
        v0=np.array([0.0, 700.0, 0.0]),
    ),
    EngagementScenario(engagement_end=300.0),
    "Ballistic target from Murmansk region (68.9°N, 33.1°E)",
)

_register(
    "ballistic_target_yakutsk",
    BallisticScenario(
        r0=geodetic_to_ecef(62.0, 129.7, 0.0),
        v0=np.array([0.0, 850.0, 0.0]),
    ),
    EngagementScenario(engagement_end=300.0),
    "Ballistic target from Yakutsk region (62.0°N, 129.7°E)",
)

# China
_register(
    "ballistic_target_beijing",
    BallisticScenario(
        r0=geodetic_to_ecef(39.9, 116.4, 0.0),
        v0=np.array([0.0, 1000.0, 0.0]),
    ),
    EngagementScenario(engagement_end=300.0),
    "Ballistic target from Beijing region (39.9°N, 116.4°E)",
)

_register(
    "ballistic_target_shanghai",
    BallisticScenario(
        r0=geodetic_to_ecef(31.2, 121.5, 0.0),
        v0=np.array([0.0, 900.0, 0.0]),
    ),
    EngagementScenario(engagement_end=300.0),
    "Ballistic target from Shanghai region (31.2°N, 121.5°E)",
)

_register(
    "ballistic_target_xian",
    BallisticScenario(
        r0=geodetic_to_ecef(34.3, 108.9, 0.0),
        v0=np.array([0.0, 950.0, 0.0]),
    ),
    EngagementScenario(engagement_end=300.0),
    "Ballistic target from Xi'an region (34.3°N, 108.9°E)",
)

_register(
    "ballistic_target_chengdu",
    BallisticScenario(
        r0=geodetic_to_ecef(30.6, 104.1, 0.0),
        v0=np.array([0.0, 850.0, 0.0]),
    ),
    EngagementScenario(engagement_end=300.0),
    "Ballistic target from Chengdu region (30.6°N, 104.1°E)",
)

_register(
    "ballistic_target_urumqi",
    BallisticScenario(
        r0=geodetic_to_ecef(43.8, 87.6, 0.0),
        v0=np.array([0.0, 800.0, 0.0]),
    ),
    EngagementScenario(engagement_end=300.0),
    "Ballistic target from Urumqi region (43.8°N, 87.6°E)",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_target_presets() -> Dict[str, TargetPreset]:
    """Return a copy of all registered target presets."""
    return dict(_TARGET_PRESETS)


def target_preset(name: str) -> TargetPreset:
    """Return a single target preset by name."""
    if name not in _TARGET_PRESETS:
        raise KeyError(f"Unknown target preset: {name}. Available: {list(_TARGET_PRESETS.keys())}")
    preset = _TARGET_PRESETS[name]
    return TargetPreset(
        name=preset.name,
        target=preset.target,
        engagement=EngagementScenario(**preset.engagement.__dict__),
        description=preset.description,
    )


def set_target_geodetic(
    lat_deg: float,
    lon_deg: float,
    alt_m: float = 0.0,
    v0: Optional[np.ndarray] = None,
    engagement_end: float = 300.0,
) -> TargetPreset:
    """Build a custom ballistic target preset from geodetic coordinates."""
    if v0 is None:
        v0 = np.array([0.0, 1000.0, 0.0])
    r0 = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    target = BallisticScenario(r0=r0, v0=v0)
    engagement = EngagementScenario(engagement_end=engagement_end)
    return TargetPreset(
        name=f"custom_geodetic_{lat_deg}_{lon_deg}",
        target=target,
        engagement=engagement,
        description=f"Custom geodetic target ({lat_deg}°, {lon_deg}°, {alt_m} m)",
    )
