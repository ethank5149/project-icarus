"""Load sensor networks from the canonical ``reference/locations.yml`` registry.

This keeps the sensor layer data-driven: defended-target / interceptor /
sensor sites declared in the YAML become ``Sensor`` objects, so coverage maps
and detection scans reuse the same geodetic database as the engagement
presets. No controlled data; all sites are OSINT-approximate.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from src.sensors.sensor import Sensor


def load_sensors_from_locations(
    path: str = "reference/locations.yml",
    designation: Optional[Sequence[str]] = None,
    tags_include: Optional[Sequence[str]] = None,
    default_range_m: float = 2000e3,
) -> List[Sensor]:
    """Build ``Sensor`` objects from ``locations.yml``.

    Parameters
    ----------
    path
        YAML registry path (relative to project root or absolute).
    designation
        If given, only sites whose ``designation`` is in this list are loaded
        (e.g. ``["interceptor-launch-site"]``). ``None`` loads every
        coordinate-bearing site.
    tags_include
        If given, only sites carrying at least one of these tags (case
        insensitive substring match) are loaded. Useful to select radar/GBI
        emplacements (e.g. ``["radar", "GBI", "interceptor"]``).
    default_range_m
        Fallback kinematic range applied when a site omits radar parameters.
    """
    import os

    import yaml

    if not os.path.isabs(path):
        # Resolve relative to project root (two levels up from src/sensors/).
        root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        path = os.path.join(root, path)

    with open(path) as f:
        data = yaml.safe_load(f)

    sensors: List[Sensor] = []
    for site in data.get("locations", []):
        coord = site.get("coordinates", {})
        if "latitude" not in coord or "longitude" not in coord:
            continue
        if designation is not None and site.get("designation") not in designation:
            continue
        if tags_include is not None:
            tags = " ".join(str(t) for t in (site.get("tags", []) or [])).lower()
            if not any(tok.lower() in tags for tok in tags_include):
                continue
        # Larger range for dedicated radar/GBI-tagged sites.
        rng = default_range_m
        tags = " ".join(str(t) for t in (site.get("tags", []) or [])).lower()
        if any(k in tags for k in ("uewr", "pave paws", "lrdr", "cobra", "gbi", "interceptor", "radar")):
            rng = 4800e3
        sensors.append(
            Sensor(
                name=site.get("name", "unknown"),
                lat_deg=float(coord["latitude"]),
                lon_deg=float(coord["longitude"]),
                # reference/locations.yml stores altitude in FEET; Sensor/ECEF want m.
                alt_m=0.3048 * float(coord.get("altitude", 0.0)),
                max_range_m=rng,
            )
        )
    return sensors
