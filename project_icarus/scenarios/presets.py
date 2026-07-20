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
    DecoyThreatScenario,
    CruiseMissileScenario,
    MU_EARTH,
    R_EARTH,
)
from .scenario import EngagementScenario
from ..interceptors.config import InterceptorConfig, GuidanceConfig
from ..dynamics.thrust import StageSpec


# ---------------------------------------------------------------------------
# Geodetic / WGS84 helpers
# ---------------------------------------------------------------------------

# WGS84 ellipsoid parameters
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_B = _WGS84_A * (1.0 - _WGS84_F)
_WGS84_E2 = (_WGS84_A**2 - _WGS84_B**2) / _WGS84_A**2

# reference/locations.yml stores site altitudes in FEET; geodetic_to_ecef wants m.
_FT_TO_M = 0.3048


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


def _enu_basis(lat_deg: float, lon_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local East/North/Up unit vectors (ECEF) at a geodetic point."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    east = np.array([-sin_lon, cos_lon, 0.0])
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    return east, north, up


def _great_circle_azimuth_range_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> Tuple[float, float]:
    """Great-circle initial azimuth (deg, 0=N clockwise) and distance (km)."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlam = np.radians(lon2 - lon1)
    sin_az = np.sin(dlam) * np.cos(phi2)
    cos_az = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
    azimuth = (np.degrees(np.arctan2(sin_az, cos_az)) + 360.0) % 360.0
    # Angular separation via spherical law of cosines.
    cos_d = np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(dlam)
    cos_d = np.clip(cos_d, -1.0, 1.0)
    dist_km = np.arccos(cos_d) * (R_EARTH / 1000.0)
    return float(azimuth), float(dist_km)


def launch_to_target_velocity(
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    az_deg: float,
    range_km: float,
    launch_el_deg: float = 45.0,
) -> Tuple[np.ndarray, float]:
    """Compute a launch ``v0`` (ECEF) aimed at ``az_deg`` over ``range_km``.

    Uses the flat-Earth maximum-range relation ``R = v0^2 * sin(2*el) / g``
    evaluated at the local gravitational acceleration, then rotates the
    resulting velocity vector into the local East/North/Up frame. Returns
    ``(v0_ecef, speed)``.
    """
    el = np.radians(launch_el_deg)
    rmag = R_EARTH + alt_m
    g = MU_EARTH / rmag**2
    sin_2el = np.sin(2.0 * el)
    if sin_2el < 1e-6:
        raise ValueError("Launch elevation too close to 0/90 deg.")
    speed = np.sqrt(g * (range_km * 1000.0) / sin_2el)
    az = np.radians(az_deg)
    east, north, up = _enu_basis(lat_deg, lon_deg)
    horizontal = np.cos(az) * north + np.sin(az) * east
    direction = np.cos(el) * horizontal + np.sin(el) * up
    direction = direction / np.linalg.norm(direction)
    return direction * speed, float(speed)


def geodetic_launch_to_target(
    launch_lat: float,
    launch_lon: float,
    launch_alt: float,
    target_lat: float,
    target_lon: float,
    target_alt: float = 0.0,
    launch_el_deg: float = 45.0,
    scenario_type: str = "ballistic",
    use_j2: bool = True,
    engagement_end: float = 1200.0,
    **kwargs,
) -> TargetPreset:
    """Build a :class:`TargetPreset` launching from a site toward a defended point.

    The trajectory is aimed along the great-circle azimuth to the target. The
    ``scenario_type`` selects the threat profile:

    - ``"ballistic"``  : ballistic arc; speed from range via flat-Earth max-range.
    - ``"fobs"``       : fractional-orbit boost to ``apoapsis_km`` then deorbit
                         steered toward the aim point (reentry phase fixed to use
                         the true ground aim, not a hardcoded point).
    - ``"hgv"``        : hypersonic glide inserted at ``glide_alt_km`` with shallow
                         flight-path angle and ``speed_mach`` (default Mach 12).
    - ``"suppressed"`` : ballistic arc with midcourse jinking along the threat axis.
    - ``"swarm"``      : ``n_payloads`` ballistic RVs spread around the bus track.

    This gives physically meaningful threat-vs-defended pairs for trajectory
    modeling across all scenario families instead of the fixed ``+y`` velocity
    used by the YAML-derived presets.
    """
    scenario_type = scenario_type.lower()
    az, rng = _great_circle_azimuth_range_km(
        launch_lat, launch_lon, target_lat, target_lon
    )
    r0 = geodetic_to_ecef(launch_lat, launch_lon, launch_alt)
    target_launch = geodetic_to_ecef(target_lat, target_lon, target_alt)

    if scenario_type == "ballistic":
        v0, speed = launch_to_target_velocity(
            launch_lat, launch_lon, launch_alt, az, rng, launch_el_deg
        )
        target = BallisticScenario(r0=r0, v0=v0, use_j2=use_j2)
        speed_note = f"v0 {speed:.0f} m/s"

    elif scenario_type == "fobs":
        apoapsis_km = kwargs.get("apoapsis_km", 200.0)
        # Boost to a low orbital speed; the deorbit/steer happens in propagate().
        rmag = np.linalg.norm(r0)
        r_apo = R_EARTH + apoapsis_km * 1e3
        v_mag = np.sqrt(MU_EARTH * (2.0 / rmag - 1.0 / r_apo))
        v0, _ = launch_to_target_velocity(
            launch_lat, launch_lon, launch_alt, az, rng, launch_el_deg=5.0
        )
        v0 = v0 / np.linalg.norm(v0) * v_mag
        target = FOBSScenario(
            r0=r0, v0=v0, apoapsis_km=apoapsis_km
        )
        speed_note = f"orbital v0 {v_mag:.0f} m/s, apoapsis {apoapsis_km:.0f} km"

    elif scenario_type == "hgv":
        glide_alt_km = kwargs.get("glide_alt_km", 70.0)
        speed_mach = kwargs.get("speed_mach", 12.0)
        # Insert at glide altitude along the great-circle azimuth, shallow FPA.
        r0 = geodetic_to_ecef(launch_lat, launch_lon, launch_alt + glide_alt_km * 1000.0)
        v0, _ = launch_to_target_velocity(
            launch_lat, launch_lon, launch_alt + glide_alt_km * 1000.0,
            az, rng, launch_el_deg=kwargs.get("glide_el_deg", 1.0),
        )
        speed = speed_mach * 300.0
        v0 = v0 / np.linalg.norm(v0) * speed
        target = HGVScenario(
            r0=r0, v0=v0, max_alt_km=glide_alt_km,
            lateral_range_km=rng,
        )
        speed_note = f"glide v0 {speed:.0f} m/s (Mach {speed_mach:.0f}), alt {glide_alt_km:.0f} km"

    elif scenario_type == "suppressed":
        v0, speed = launch_to_target_velocity(
            launch_lat, launch_lon, launch_alt, az, rng, launch_el_deg
        )
        target = SuppressedScenario(
            r0=r0, v0=v0,
            dip_alt_km=kwargs.get("dip_alt_km", 50.0),
            midcourse_maneuver_mag=kwargs.get("midcourse_maneuver_mag", 50.0),
            maneuver_interval=kwargs.get("maneuver_interval", 30.0),
        )
        speed_note = f"v0 {speed:.0f} m/s (midcourse jink along threat axis)"

    elif scenario_type == "swarm":
        v0, speed = launch_to_target_velocity(
            launch_lat, launch_lon, launch_alt, az, rng, launch_el_deg
        )
        n_payloads = kwargs.get("n_payloads", 3)
        spread_deg = kwargs.get("spread_deg", 1.0)
        target = SwarmScenario(
            bus_r0=r0, bus_v0=v0, n_payloads=n_payloads, spread_deg=spread_deg
        )
        speed_note = f"bus v0 {speed:.0f} m/s, {n_payloads} RVs, {spread_deg:.1f}° spread"

    else:
        raise ValueError(
            f"Unknown scenario_type '{scenario_type}'. "
            "Expected one of: ballistic, fobs, hgv, suppressed, swarm."
        )

    engagement = EngagementScenario(
        engagement_end=engagement_end,
        target_launch_site=target_launch,
    )
    name = (
        f"{scenario_type}_launch_to_target_"
        f"{launch_lat:.2f}_{launch_lon:.2f}_->_{target_lat:.2f}_{target_lon:.2f}"
    )
    desc = (
        f"{scenario_type.upper()} launch from ({launch_lat:.2f}, {launch_lon:.2f}) toward "
        f"({target_lat:.2f}, {target_lon:.2f}): azimuth {az:.1f}°, range {rng:.0f} km, "
        f"{speed_note}"
    )
    return TargetPreset(
        name=name,
        target=target,
        engagement=engagement,
        description=desc,
    )


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


def _load_interceptor_presets_from_locations() -> Dict[str, np.ndarray]:
    """Augment interceptor presets from ``reference/locations.yml``.

    Sites tagged ``designation: interceptor-launch-site`` are merged in using
    their exact database coordinates, overriding any rough geodetic estimates
    already present and adding the real-world bases (GMD, Aegis Ashore, THAAD).
    """
    from reference.locations import locations_by_designation, coordinates_to_ecef, _sanitize_key

    presets: Dict[str, np.ndarray] = {}
    groups = locations_by_designation()
    for rec in groups.get("interceptor-launch-site", []):
        presets[_sanitize_key(rec["name"])] = coordinates_to_ecef(rec)
    return presets


_INTERCEPTOR_PRESETS.update(_load_interceptor_presets_from_locations())

# The YAML database carries survey-grade coordinates for some sites that also
# have rough estimates above (e.g. Vandenberg). Keep the precise values and
# drop the redundant duplicate key the YAML introduced for shared facilities.
_INTERCEPTOR_PRESETS["vandenberg"] = _INTERCEPTOR_PRESETS.get(
    "vandenberg_space_force_base", _INTERCEPTOR_PRESETS["vandenberg"]
)


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

# --- Cruise missile family ---

_register(
    "cruise_short_range",
    CruiseMissileScenario.from_params(launch_alt_km=0.0, range_km=300.0, speed_mach=0.8),
    EngagementScenario(engagement_end=300.0),
    "Short-range cruise missile, 100 m terrain-following alt, Mach 0.8",
)

_register(
    "cruise_medium_range",
    CruiseMissileScenario.from_params(launch_alt_km=0.0, range_km=800.0, speed_mach=0.85),
    EngagementScenario(engagement_end=500.0),
    "Medium-range cruise missile, Mach 0.85, 800 km range",
)

_register(
    "cruise_sea_launched",
    CruiseMissileScenario.from_params(launch_alt_km=0.0, range_km=500.0, speed_mach=0.8),
    EngagementScenario(engagement_end=400.0),
    "Sea-launched cruise missile, 100 m terrain-following",
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

# --- Advanced threat family (2C): agile RV + signature-evolving decoys ---

_register(
    "maneuvering_rv",
    SuppressedScenario(
        r0=np.array([R_EARTH, 0.0, 0.0]),
        v0=np.array([0.0, 1100.0, 600.0]),
        dip_alt_km=40.0,
        midcourse_maneuver_mag=140.0,
        maneuver_interval=12.0,
    ),
    EngagementScenario(engagement_end=350.0),
    "Agile RV with frequent high-magnitude midcourse jinks (evasive)",
)

_register(
    "advanced_decoy_threat",
    DecoyThreatScenario(
        rv=BallisticScenario(
            r0=np.array([R_EARTH, 0.0, 0.0]),
            v0=np.array([0.0, 1300.0, 700.0]),
        ),
        decoys=[
            # Signature-evolving balloon decoy: tracks the RV's RCS/IR closely
            # so the seeker/discriminator must use micro-motion + Doppler to
            # separate the real RV from the clutter.
            dict(mass=45.0, area=0.12, cd=0.55, radar_rcs_bias=0.9,
                 ir_bias=0.85, release_altitude=55e3),
            dict(mass=55.0, area=0.10, cd=0.50, radar_rcs_bias=1.1,
                 ir_bias=0.7, release_altitude=60e3),
        ],
        release_t=180.0,
    ),
    EngagementScenario(engagement_end=350.0),
    "RV + 2 signature-evolving balloon decoys released in midcourse",
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

# --- Geodetic target presets from reference/locations.yml ---------------
# Foreign strategic launch complexes (designation: target-launch-site) are
# registered as ballistic targets using their exact database coordinates.
# Designated defended targets (designation: defended-target) are registered
# both as defended-point aim locations and as ballistic targets so they can be
# used to model threat trajectories toward them.

def _register_from_locations():
    from reference.locations import (
        locations_by_designation,
        coordinates_to_ecef,
        _sanitize_key,
    )

    def _ballistic_from_record(rec, v_mag, engagement_end):
        coord = rec["coordinates"]
        r0 = geodetic_to_ecef(
            float(coord["latitude"]),
            float(coord["longitude"]),
            _FT_TO_M * float(coord.get("altitude", 0.0)),
        )
        v0 = np.array([0.0, v_mag, 0.0])
        tags = ", ".join(rec.get("tags", [])[:2])
        return BallisticScenario(r0=r0, v0=v0), tags

    groups = locations_by_designation()

    # Foreign strategic launch complexes -> ballistic target launch sites.
    # Speed scaled by a rough threat class label so heavy ICBM fields get a
    # higher-energy launch than smaller/road-mobile garrisons.
    for rec in groups.get("target-launch-site", []):
        name = rec["name"]
        key = "ballistic_target_" + _sanitize_key(name)
        v_mag = 900.0 if any(
            t in " ".join(rec.get("tags", [])).lower()
            for t in ("icbm", "sarmat", "satan", "mirv", "df-41", "df-5", "jl-3", "jl-2")
        ) else 700.0
        target, tags = _ballistic_from_record(rec, v_mag, 300.0)
        _register(
            key,
            target,
            EngagementScenario(engagement_end=300.0),
            f"Ballistic target from {name} ({tags})",
        )

    # Defended targets -> usable as defended points and as target trajectories.
    for rec in groups.get("defended-target", []):
        name = rec["name"]
        coord = rec["coordinates"]
        r0 = geodetic_to_ecef(
            float(coord["latitude"]),
            float(coord["longitude"]),
            _FT_TO_M * float(coord.get("altitude", 0.0)),
        )
        tags = ", ".join(rec.get("tags", [])[:2])
        key = "defended_" + _sanitize_key(name)
        _register(
            key,
            BallisticScenario(r0=r0, v0=np.zeros(3)),
            EngagementScenario(
                engagement_end=300.0,
                target_launch_site=r0,
            ),
            f"Defended target: {name} ({tags})",
        )

    # Curated threat -> defended pairs: each major foreign launch complex is
    # aimed at a representative high-value US defended target with a real
    # great-circle azimuth/elevation and range-derived launch speed.
    _THREAT_TO_DEFENDED = [
        ("Kozelsk", "Washington D.C."),
        ("Tatishchevo", "Washington D.C."),
        ("Dombarovsky / Yasny", "Whiteman AFB"),
        ("Uzhur", "Minot AFB"),
        ("Plesetsk Cosmodrome", "NORAD, Peterson SFB"),
        ("Vypolzovo / Yedrovo", "The Pentagon"),
        ("Teykovo", "Raven Rock Mountain Complex"),
        ("Yoshkar-Ola", "Offutt AFB"),
        ("Novosibirsk", "Malmstrom AFB"),
        ("Irkutsk", "F.E. Warren AFB"),
        ("Barnaul", "Naval Station Norfolk"),
        ("Yumen Silo Field", "Los Angeles"),
        ("Hami Silo Field", "San Francisco"),
        ("Ordos Silo Field", "Denver"),
        ("Luoning Complex", "Kirtland AFB Albuquerque"),
        ("Longpo Naval Base", "San Diego"),
        ("Tonghua Garrison", "Naval Base Kitsap"),
        ("Xiangyang Garrison", "Houston"),
    ]
    by_name = {rec["name"]: rec for rec in groups.get("target-launch-site", [])}
    defended_by_name = {rec["name"]: rec for rec in groups.get("defended-target", [])}
    # Each curated pair is registered for every scenario family so the database
    # covers ballistic, FOBS, HGV, suppressed, and swarm threat profiles aimed
    # at the same defended point.
    _SCENARIO_VARIANTS = [
        ("ballistic", {}, ""),
        ("fobs", {"apoapsis_km": 200.0}, "_fobs"),
        ("hgv", {"glide_alt_km": 70.0, "speed_mach": 12.0}, "_hgv"),
        ("suppressed", {}, "_suppressed"),
    ]
    for threat_name, defended_name in _THREAT_TO_DEFENDED:
        if threat_name not in by_name or defended_name not in defended_by_name:
            continue
        threat = by_name[threat_name]["coordinates"]
        defended = defended_by_name[defended_name]["coordinates"]
        for sc_type, sc_kwargs, suffix in _SCENARIO_VARIANTS:
            preset = geodetic_launch_to_target(
                float(threat["latitude"]),
                float(threat["longitude"]),
                _FT_TO_M * float(threat.get("altitude", 0.0)),
                float(defended["latitude"]),
                float(defended["longitude"]),
                _FT_TO_M * float(defended.get("altitude", 0.0)),
                launch_el_deg=45.0,
                scenario_type=sc_type,
                engagement_end=1200.0,
                **sc_kwargs,
            )
            key = (
                f"threat_{_sanitize_key(threat_name)}_to_"
                f"{_sanitize_key(defended_name)}{suffix}"
            )
            _register(key, preset.target, preset.engagement, preset.description)


_register_from_locations()


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


def build_threat_to_defended(
    threat_name: str,
    defended_name: str,
    launch_el_deg: float = 45.0,
    scenario_type: str = "ballistic",
    engagement_end: float = 1200.0,
    **kwargs,
) -> TargetPreset:
    """Build a threat-vs-defended :class:`TargetPreset` from the locations DB.

    Looks up ``threat_name`` (a ``target-launch-site``) and ``defended_name`` (a
    ``defended-target``) in ``reference/locations.yml`` and returns a target
    whose ``v0`` is aimed along the great-circle azimuth toward the defended
    point. ``scenario_type`` selects the threat profile (``ballistic``,
    ``fobs``, ``hgv``, ``suppressed``, ``swarm``); extra scenario parameters are
    passed via ``kwargs``. This is the modeling entry point for end-to-end
    threat trajectories; the curated ``threat_*_to_*`` presets use the same
    machinery.
    """
    from reference.locations import locations_by_designation

    groups = locations_by_designation()
    by_name = {rec["name"]: rec for rec in groups.get("target-launch-site", [])}
    defended_by_name = {rec["name"]: rec for rec in groups.get("defended-target", [])}
    if threat_name not in by_name:
        raise KeyError(f"Unknown threat launch site: {threat_name}")
    if defended_name not in defended_by_name:
        raise KeyError(f"Unknown defended target: {defended_name}")
    threat = by_name[threat_name]["coordinates"]
    defended = defended_by_name[defended_name]["coordinates"]
    return geodetic_launch_to_target(
        float(threat["latitude"]),
        float(threat["longitude"]),
        _FT_TO_M * float(threat.get("altitude", 0.0)),
        float(defended["latitude"]),
        float(defended["longitude"]),
        _FT_TO_M * float(defended.get("altitude", 0.0)),
        launch_el_deg=launch_el_deg,
        scenario_type=scenario_type,
        engagement_end=engagement_end,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Interceptor vehicle presets (2B.4) — Arrow-3 / Tamir / GMD with UQ
# ---------------------------------------------------------------------------
# OSINT-approximate parameters (illustrative research defaults, NOT controlled
# data). Each preset builds a multi-stage InterceptorConfig with realistic
# stage masses/Isp and a selectable terminal guidance backend. sample_uq()
# draws parameter perturbations (±20% mass/Isp/divert) for Monte-Carlo.

def _constant_thrust(value: float):
    return lambda t: value


def _build_arrow3() -> dict:
    return dict(
        name="Arrow-3 (exoatmospheric hit-to-kill)",
        mass=1300.0,
        area=0.20,
        ref_length=0.7,
        kill_radius=0.5,
        kill_mechanism="hit_to_kill",
        accel_limit=180.0,
        seeker_snr=20.0,
        inertia=np.diag([60.0, 120.0, 150.0]),
        mkv_mass=15.0,
        mkv_divert_impulse=85.0,
        stages=[
            StageSpec(thrust=_constant_thrust(420e3), burn_time=6.0,
                      wet_mass=900.0, dry_mass=150.0, Isp=300.0, name="booster"),
            StageSpec(thrust=_constant_thrust(120e3), burn_time=8.0,
                      wet_mass=350.0, dry_mass=80.0, Isp=320.0, name="sustainer"),
        ],
        guidance=dict(
            midcourse_n=5.0, midcourse_accel_limit=60.0, terminal_n=4.0,
            terminal_accel_limit=180.0, terminal_guidance_law="apn",
            seeker_mode="ir", seeker_fov_deg=60.0, ukf_enabled=True,
        ),
    )


def _build_tamir() -> dict:
    return dict(
        name="Iron Dome Tamir (endoaortic point-defense)",
        mass=90.0,
        area=0.03,
        ref_length=0.15,
        kill_radius=0.5,
        kill_mechanism="blast_frag",
        accel_limit=300.0,
        seeker_snr=18.0,
        inertia=np.diag([2.0, 4.0, 5.0]),
        mkv_mass=0.0,
        mkv_divert_impulse=0.0,
        stages=[
            StageSpec(thrust=_constant_thrust(60e3), burn_time=2.0,
                      wet_mass=60.0, dry_mass=15.0, Isp=250.0, name="booster"),
        ],
        guidance=dict(
            midcourse_n=5.0, midcourse_accel_limit=40.0, terminal_n=4.0,
            terminal_accel_limit=300.0, terminal_guidance_law="sdre_mpc",
            seeker_mode="radar", seeker_fov_deg=50.0, ukf_enabled=True,
            zem_horizon=3.0,
        ),
    )


def _build_gmd() -> dict:
    return dict(
        name="GMD GBII (exoatmospheric EKV hit-to-kill)",
        mass=64000.0,
        area=1.5,
        ref_length=1.3,
        kill_radius=0.5,
        kill_mechanism="hit_to_kill",
        accel_limit=120.0,
        seeker_snr=22.0,
        inertia=np.diag([3000.0, 6000.0, 8000.0]),
        mkv_mass=60.0,
        mkv_divert_impulse=200.0,
        stages=[
            StageSpec(thrust=_constant_thrust(8.5e6), burn_time=60.0,
                      wet_mass=30000.0, dry_mass=4000.0, Isp=270.0, name="stage1"),
            StageSpec(thrust=_constant_thrust(4.0e6), burn_time=70.0,
                      wet_mass=14000.0, dry_mass=2000.0, Isp=290.0, name="stage2"),
            StageSpec(thrust=_constant_thrust(2.0e6), burn_time=80.0,
                      wet_mass=8000.0, dry_mass=1500.0, Isp=300.0, name="stage3"),
        ],
        guidance=dict(
            midcourse_n=5.0, midcourse_accel_limit=40.0, terminal_n=4.0,
            terminal_accel_limit=120.0, terminal_guidance_law="zem",
            seeker_mode="ir", seeker_fov_deg=65.0, ukf_enabled=True,
        ),
    )


def _build_patriot() -> dict:
    # OSINT-approximate PAC-3 MSE: endoatmospheric lower-tier point/area defense,
    # radar seeker, hit-to-kill. Mass/Isp/stages are illustrative research
    # defaults, NOT controlled data.
    return dict(
        name="Patriot PAC-3 MSE (endoatmospheric hit-to-kill)",
        mass=350.0,
        area=0.05,
        ref_length=0.25,
        kill_radius=0.5,
        kill_mechanism="hit_to_kill",
        accel_limit=600.0,
        seeker_snr=18.0,
        inertia=np.diag([3.0, 6.0, 8.0]),
        mkv_mass=25.0,
        mkv_divert_impulse=120.0,
        stages=[
            StageSpec(thrust=_constant_thrust(180e3), burn_time=4.0,
                      wet_mass=300.0, dry_mass=50.0, Isp=250.0, name="booster"),
        ],
        guidance=dict(
            midcourse_n=5.0, midcourse_accel_limit=120.0, terminal_n=4.0,
            terminal_accel_limit=600.0, terminal_guidance_law="sdre_mpc",
            seeker_mode="radar", seeker_fov_deg=55.0, ukf_enabled=True,
            zem_horizon=3.0,
        ),
    )


def _build_thaad() -> dict:
    # OSINT-approximate THAAD: lower/mid-tier hit-to-kill, terminal IR seeker,
    # engages endo- and exo-atmospheric. Mass/Isp/stages are illustrative research
    # defaults, NOT controlled data.
    return dict(
        name="THAAD (endo/exo hit-to-kill)",
        mass=900.0,
        area=0.10,
        ref_length=0.35,
        kill_radius=0.5,
        kill_mechanism="hit_to_kill",
        accel_limit=550.0,
        seeker_snr=20.0,
        inertia=np.diag([20.0, 40.0, 50.0]),
        mkv_mass=40.0,
        mkv_divert_impulse=160.0,
        stages=[
            StageSpec(thrust=_constant_thrust(500e3), burn_time=6.0,
                      wet_mass=800.0, dry_mass=100.0, Isp=260.0, name="booster"),
        ],
        guidance=dict(
            midcourse_n=5.0, midcourse_accel_limit=120.0, terminal_n=4.0,
            terminal_accel_limit=550.0, terminal_guidance_law="apn",
            seeker_mode="ir", seeker_fov_deg=60.0, ukf_enabled=True,
        ),
    )


_INTERCEPTOR_BUILDERS: Dict[str, callable] = {
    "arrow3": _build_arrow3,
    "tamir": _build_tamir,
    "gmd": _build_gmd,
    "patriot": _build_patriot,
    "thaad": _build_thaad,
}


def _spec_to_pair(spec: dict) -> Tuple[InterceptorConfig, GuidanceConfig]:
    g = spec.pop("guidance")
    guidance_cfg = GuidanceConfig(**g)
    cfg = InterceptorConfig(**spec)
    return cfg, guidance_cfg


def build_interceptor_config(name: str) -> Tuple[InterceptorConfig, GuidanceConfig]:
    """Build an interceptor preset by name.

    Returns ``(InterceptorConfig, GuidanceConfig)``. Supported names:
    ``arrow3``, ``tamir``, ``gmd``, ``patriot``, ``thaad``. Each bundles a
    multi-stage thrust model and a calibrated GuidanceConfig selecting one of
    the 2B.2 terminal backends.
    """
    if name not in _INTERCEPTOR_BUILDERS:
        raise KeyError(
            f"Unknown interceptor preset: {name}. Available: {list(_INTERCEPTOR_BUILDERS.keys())}"
        )
    return _spec_to_pair(_INTERCEPTOR_BUILDERS[name]())


def get_interceptor_config_presets() -> Dict[str, Tuple[InterceptorConfig, GuidanceConfig]]:
    """Return a mapping of all interceptor presets (nominal params)."""
    return {name: build_interceptor_config(name) for name in _INTERCEPTOR_BUILDERS}


def interceptor_config_preset(name: str) -> Tuple[InterceptorConfig, GuidanceConfig]:
    """Return a single interceptor preset by name."""
    return build_interceptor_config(name)


def sample_interceptor_uq(name: str, rng: Optional[np.random.Generator] = None,
                          frac: float = 0.20) -> Tuple[InterceptorConfig, GuidanceConfig]:
    """Draw a UQ-perturbed copy of an interceptor preset (2B.4).

    Mass, stage Isp, and divert impulse are perturbed by up to ``frac`` (±20%
    default) log-normal draws. Returns ``(InterceptorConfig, GuidanceConfig)``;
    the caller's RNG is threaded for reproducibility.
    """
    rng = rng or np.random.default_rng()
    cfg, guidance_cfg = build_interceptor_config(name)
    perturb = lambda base: float(base * np.exp(rng.uniform(-frac, frac)))
    cfg.mass = perturb(cfg.mass)
    cfg.mkv_divert_impulse = perturb(cfg.mkv_divert_impulse)
    new_stages = []
    for s in cfg.stages:
        new_stages.append(StageSpec(
            thrust=s.thrust, burn_time=s.burn_time,
            wet_mass=perturb(s.wet_mass), dry_mass=perturb(s.dry_mass),
            Isp=perturb(s.Isp), gimbal_limits=s.gimbal_limits,
            gimbal_rate=s.gimbal_rate, name=s.name,
        ))
    cfg.stages = new_stages
    return cfg, guidance_cfg
