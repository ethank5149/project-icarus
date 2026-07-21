from __future__ import annotations

import os
import re
import math
import numpy as np
import rasterio

_REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reference")
_GMTED2010_DIR = os.path.join(_REFERENCE_DIR, "GMTED2010")

_TILE_RE = re.compile(
    r"GMTED2010([NS])(\d{2})([EW])(\d{3})_(\d{3})"
)


def _parse_tile_bounds(tile_name: str):
    m = _TILE_RE.match(tile_name)
    if not m:
        raise ValueError(f"Invalid GMTED2010 tile name: {tile_name}")
    ns, lat_str, ew, lon_str, _ = m.groups()
    lat_min = int(lat_str) if ns == "N" else -int(lat_str)
    lon_min = int(lon_str) if ew == "E" else -int(lon_str)
    return lat_min, lat_min + 20, lon_min, lon_min + 30


_cache: dict[str, dict] = {}


def _load_tile(tile_name: str) -> dict:
    entry = _cache.get(tile_name)
    if entry is not None:
        return entry
    tile_path = os.path.join(_GMTED2010_DIR, tile_name)
    if not os.path.isdir(tile_path):
        return None
    candidates = sorted(f for f in os.listdir(tile_path) if "_mea" in f and f.endswith(".tif"))
    if not candidates:
        return None
    tif_path = os.path.join(tile_path, candidates[0])
    try:
        src = rasterio.open(tif_path)
        lat_min, lat_max, lon_min, lon_max = _parse_tile_bounds(tile_name)
        nrows, ncols = src.height, src.width
        dlat = (lat_max - lat_min) / nrows
        dlon = (lon_max - lon_min) / ncols
        entry = {
            "src": src,
            "closed": False,
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
            "nrows": nrows,
            "ncols": ncols,
            "dlat": dlat,
            "dlon": dlon,
        }
        _cache[tile_name] = entry
        return entry
    except Exception:
        return None


def _find_tile(lat_deg: float, lon_deg: float):
    lat = float(lat_deg)
    lon = float(lon_deg)

    if lat >= 10.0:
        ns = "N"
        lat_band = ((int(lat) - 10) // 20) * 20 + 10
    elif lat >= -10.0:
        ns = "S"
        lat_band = 10
    else:
        ns = "S"
        lat_band = (int((abs(lat) + 9) / 20) * 20) + 10

    if lon >= 0.0:
        ew = "E"
        lon_band = (int(lon) // 30) * 30
    else:
        ew = "W"
        lon_band = math.ceil(-lon / 30.0) * 30.0

    tile_name = (
        f"GMTED2010{ns}{int(lat_band):02d}"
        f"{ew}{int(lon_band):03d}_075"
    )
    return _load_tile(tile_name)


def _interpolate(tile, lat_deg: float, lon_deg: float):
    row_float = (tile["lat_max"] - lat_deg) / tile["dlat"]
    col_float = (lon_deg - tile["lon_min"]) / tile["dlon"]
    row0 = int(math.floor(row_float))
    col0 = int(math.floor(col_float))
    row1 = min(row0 + 1, tile["nrows"] - 1)
    col1 = min(col0 + 1, tile["ncols"] - 1)
    row0 = max(row0, 0)
    col0 = max(col0, 0)
    if row0 >= tile["nrows"]:
        row0 = tile["nrows"] - 1
    if col0 >= tile["ncols"]:
        col0 = tile["ncols"] - 1
    dr = row_float - row0
    dc = col_float - col0
    window = ((row0, row1 + 1), (col0, col1 + 1))
    arr = tile["src"].read(1, window=window)
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 2, axis=0)
    if arr.shape[1] == 1:
        arr = np.repeat(arr, 2, axis=1)
    v00 = arr[0, 0]
    v01 = arr[0, 1]
    v10 = arr[1, 0]
    v11 = arr[1, 1]
    return float(
        v00 * (1 - dr) * (1 - dc)
        + v01 * (1 - dr) * dc
        + v10 * dr * (1 - dc)
        + v11 * dr * dc
    )


def get_surface_elevation(lat_deg: float, lon_deg: float) -> float:
    tile = _find_tile(float(lat_deg), float(lon_deg))
    if tile is None:
        return 0.0
    return _interpolate(tile, float(lat_deg), float(lon_deg))


def ecef_to_surface_altitude(r):
    from project_icarus.scenarios.target_factory import _ecef_to_geodetic
    lat, lon, _ = _ecef_to_geodetic(r)
    return get_surface_elevation(lat, lon)
