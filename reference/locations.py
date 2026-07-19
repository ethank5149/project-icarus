from __future__ import annotations

import os
import numpy as np
import yaml

# Resolve the locations database relative to this module.
_LOCATIONS_YAML = os.path.join(os.path.dirname(__file__), "locations.yml")


def load_locations(path: str = _LOCATIONS_YAML) -> list:
    """Parse ``reference/locations.yml`` and return the list of location records."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data.get("locations", [])


def locations_by_designation(path: str = _LOCATIONS_YAML) -> dict:
    """Return ``{designation: [location records]}`` grouped from the database."""
    groups: dict = {}
    for rec in load_locations(path):
        groups.setdefault(rec.get("designation", "unknown"), []).append(rec)
    return groups


def locations_by_name(path: str = _LOCATIONS_YAML) -> dict:
    """Return ``{name: record}`` keyed by location name."""
    return {rec["name"]: rec for rec in load_locations(path)}


def coordinates_to_ecef(record: dict) -> np.ndarray:
    """Convert a location record's geodetic coordinates to ECEF (meters)."""
    from src.scenarios.presets import geodetic_to_ecef

    coord = record["coordinates"]
    return geodetic_to_ecef(
        lat_deg=float(coord["latitude"]),
        lon_deg=float(coord["longitude"]),
        alt_m=float(coord.get("altitude", 0.0)),
    )


def _sanitize_key(name: str) -> str:
    """Turn a human-friendly location name into a lowercase preset key."""
    key = name.lower()
    key = key.replace("(", "").replace(")", "")
    key = key.replace(" / ", "_").replace("/", "_")
    key = key.replace(" ", "_").replace("-", "_").replace(".", "")
    key = key.replace("__", "_")
    while key.endswith("_"):
        key = key[:-1]
    return key
