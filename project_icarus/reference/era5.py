from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import xarray as xr
    _HAVE_XARRAY = True
except ImportError:  # pragma: no cover - optional dependency
    _HAVE_XARRAY = False


class ERA5Interpolator:
    """Lazy, memory-safe interpolator for ERA5 reanalysis GRIB files.

    Opens one or more monthly GRIBs with ``xarray`` + ``dask`` backend and
    exposes 4-D interpolation (lat / lon / alt / time) for wind, temperature,
    pressure, and density.  Never materializes a full month (~40 GB).
    """

    def __init__(
        self,
        grib_paths: Union[str, Sequence[str]],
        variables: Optional[List[str]] = None,
        chunks: Optional[Dict[str, int]] = None,
    ):
        if not _HAVE_XARRAY:
            raise ImportError("xarray + cfgrib are required for ERA5 interpolation")
        if isinstance(grib_paths, str):
            grib_paths = [grib_paths]
        self.grib_paths: List[str] = [str(p) for p in grib_paths]
        self.variables = variables or ["u", "v", "t", "q"]
        self.chunks = chunks or {"time": 1}
        self._ds: Optional[xr.Dataset] = None
        self._time_ns: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Lazy dataset open
    # ------------------------------------------------------------------
    def _open(self) -> xr.Dataset:
        if self._ds is None:
            self._ds = xr.open_mfdataset(
                self.grib_paths,
                engine="cfgrib",
                combine="by_coords",
                chunks=self.chunks,
                backend_kwargs={"indexpath": ""},
            )
            self._time_ns = self._ds.time.values.astype("datetime64[ns]").view("i8")
        return self._ds

    def close(self):
        if self._ds is not None:
            self._ds.close()
            self._ds = None
            self._time_ns = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _to_ns(self, t_query) -> int:
        if isinstance(t_query, (int, float, np.integer, np.floating)):
            return int(t_query)
        if hasattr(t_query, "value"):
            return int(np.datetime64(t_query, "ns").view("i8"))
        return int(np.datetime64(str(t_query), "ns").view("i8"))

    def _safe_time(self, t_query: float) -> float:
        """Clamp ``t_query`` to the loaded time window; emit a warning if clamped."""
        ds = self._open()
        t_ns = self._time_ns
        q_ns = self._to_ns(t_query)
        t_min = int(t_ns[0])
        t_max = int(t_ns[-1])
        if q_ns < t_min or q_ns > t_max:
            logger.warning(
                "ERA5 query %s outside loaded window [%s, %s]; clamping",
                t_query,
                np.datetime64(t_min, "ns"),
                np.datetime64(t_max, "ns"),
            )
            q_ns = int(np.clip(q_ns, t_min, t_max))
        return float(q_ns)

    def _time_dim(self, ds: xr.Dataset) -> str:
        return "time"

    def _vertical_coord(self, ds: xr.Dataset) -> str:
        for candidate in ("level", "isobaricInhPa", "pressure"):
            if candidate in ds.dims:
                return candidate
        raise KeyError("No vertical pressure level dimension found in ERA5 dataset")

    def _USSA76_pressure(self, alt_m: float) -> float:
        """Invert USSA76 to get pressure at the given geometric altitude (m)."""
        from ..dynamics.atmosphere import _ussa76_pressure
        return float(_ussa76_pressure(np.asarray([alt_m], dtype=float))[0])

    def _alt_to_p(self, alt_m: float) -> float:
        if alt_m < 0.0:
            alt_m = 0.0
        if alt_m > 48e3:
            alt_m = 48e3
        return self._USSA76_pressure(alt_m)

    def _interp_vertical(
        self, vals: np.ndarray, p_levels: np.ndarray, p_target: float
    ) -> float:
        """Log-linear interpolation of ``vals`` at ``p_target``."""
        if p_target <= p_levels[0]:
            return float(vals[0])
        if p_target >= p_levels[-1]:
            return float(vals[-1])
        log_p = np.log(p_levels)
        log_pt = np.log(p_target)
        for i in range(len(p_levels) - 1):
            if log_p[i] <= log_pt <= log_p[i + 1]:
                frac = (log_pt - log_p[i]) / (log_p[i + 1] - log_p[i])
                return float(vals[i] * (1.0 - frac) + vals[i + 1] * frac)
        return float(vals[-1])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def wind(self, lat: float, lon: float, alt: float, t_query) -> Tuple[float, float, float]:
        """Return (u, v, w) wind components at the given point.

        ``w`` is set to ``0.0`` because vertical wind is not in ERA5.
        """
        u = self._interp_point(lat, lon, alt, t_query, "u")
        v = self._interp_point(lat, lon, alt, t_query, "v")
        return float(u), float(v), 0.0

    def temperature(self, lat: float, lon: float, alt: float, t_query) -> float:
        return float(self._interp_point(lat, lon, alt, t_query, "t"))

    def pressure(self, lat, lon, alt, t_query):
        return float(self._alt_to_p(max(float(alt), 0.0)))

    def density(self, lat: float, lon: float, alt: float, t_query) -> float:
        T = self.temperature(lat, lon, alt, t_query)
        p = self.pressure(lat, lon, alt, t_query)
        q = self._interp_point(lat, lon, alt, t_query, "q")
        M_wet = 28.97e-3
        R_spec = 8.31446 / M_wet
        return float(p / (R_spec * max(T, 150.0)) * (1.0 + 0.61 * float(q)))

    def _interp_point(self, lat, lon, alt, t_query, var):
        """Common 4-D interpolation for ERA5 variables."""
        ds = self._open()
        q_ns = self._to_ns(t_query)
        t_ns = self._time_ns
        vcoord = self._vertical_coord(ds)
        p_levels = ds[vcoord].values.astype(float)

        t_min = int(t_ns[0])
        t_max = int(t_ns[-1])
        q_clamped = int(np.clip(q_ns, t_min, t_max))

        # Time index / fraction
        if q_clamped <= t_ns[0]:
            t_idx0, t_idx1, frac_t = 0, 0, 0.0
        elif q_clamped >= t_ns[-1]:
            t_idx0, t_idx1, frac_t = -1, -1, 0.0
        else:
            t_idx1 = int(np.searchsorted(t_ns, q_clamped))
            t_idx0 = t_idx1 - 1
            frac_t = (q_clamped - t_ns[t_idx0]) / max(t_ns[t_idx1] - t_ns[t_idx0], 1)

        # Lat/lon index / fraction
        lat_vals = ds.latitude.values
        lon_vals = ds.longitude.values
        if lat_vals[0] > lat_vals[-1]:
            lat_vals = lat_vals[::-1]
        if lon_vals[0] > lon_vals[-1]:
            lon_vals = lon_vals[::-1]

        def _lat_idx(vals, val):
            i = int(np.argmin(np.abs(vals - val)))
            return min(i, len(vals) - 2)

        def _lon_idx(vals, val):
            i = int(np.argmin(np.abs(vals - val)))
            return min(i, len(vals) - 2)

        li = _lat_idx(lat_vals, lat)
        lo = _lon_idx(lon_vals, lon)
        frac_lat = (lat - lat_vals[li]) / max(lat_vals[li + 1] - lat_vals[li], 1e-12)
        frac_lon = (lon - lon_vals[lo]) / max(lon_vals[lo + 1] - lon_vals[lo], 1e-12)
        frac_lat = np.clip(frac_lat, 0.0, 1.0)
        frac_lon = np.clip(frac_lon, 0.0, 1.0)

        def _bilerp(slc, iy, ix):
            return (
                slc[:, iy, ix] * (1 - frac_lat) * (1 - frac_lon)
                + slc[:, iy + 1, ix + 1] * frac_lat * frac_lon
                + slc[:, iy + 1, ix] * frac_lat * (1 - frac_lon)
                + slc[:, iy, ix + 1] * (1 - frac_lat) * frac_lon
            )

        def _get(slc_idx):
            slc = ds[var].isel({self._time_dim(ds): slc_idx})
            if lat_vals[0] > lat_vals[-1]:
                slc = slc.isel(latitude=slice(None, None, -1))
            return _bilerp(slc.values, li, lo)

        v0 = _get(t_idx0)
        if frac_t < 1e-12 or t_idx0 == t_idx1:
            v_interp = v0
        else:
            v1 = _get(t_idx1)
            v_interp = v0 * (1 - frac_t) + v1 * frac_t

        # Vertical interpolation using USSA76 pressure mapping
        p_ref = self._alt_to_p(max(float(alt), 0.0))
        p_ref = np.clip(p_ref, p_levels[-1] * 0.5, p_levels[0] * 1.5)
        return self._interp_vertical(v_interp, p_levels, float(p_ref))
