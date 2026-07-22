"""Optional ERA5 GRIB -> Zarr converter for faster repeated access.

Usage
-----
    python reference/prepare_era5.py reference/ERA5/era5_global_native_025_2015_01.grib \\
        --output reference/ERA5/era5_2015_01.zarr

After conversion, point ``ERA5Interpolator`` at the ``.zarr`` directory instead
of the ``.grib`` file:
    era5 = ERA5Interpolator("reference/ERA5/era5_2015_01.zarr")
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Sequence

logger = logging.getLogger(__name__)


def prepare_era5_zarr(
    grib_paths: Sequence[str],
    output_path: str,
    variables: Sequence[str] = ("u", "v", "t", "q"),
    chunks: dict | None = None,
) -> str:
    """Open one or more ERA5 GRIBs and write them to a Zarr store.

    Parameters
    ----------
    grib_paths :
        Input GRIB file paths.
    output_path :
        Destination Zarr directory.
    variables :
        Variables to include (subset of ``u, v, t, q``).
    chunks :
        Chunk sizes for the Zarr store.  Defaults to ``{"time": 1, "isobaricInhPa": 21}``.

    Returns
    -------
    str
        The output path on success.
    """
    try:
        import xarray as xr
    except ImportError as e:
        raise ImportError("xarray is required for Zarr conversion") from e

    chunks = chunks or {"time": 1, "isobaricInhPa": 21}
    ds = xr.open_mfdataset(
        grib_paths,
        engine="cfgrib",
        combine="by_coords",
        chunks=chunks,
        backend_kwargs={"indexpath": ""},
    )
    # Subset to requested variables.
    ds = ds[list(variables)]

    encoding = {var: {"compressor": None} for var in ds.data_vars}
    ds.to_zarr(output_path, mode="w", encoding=encoding)
    ds.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert ERA5 GRIB to Zarr")
    parser.add_argument("gribs", nargs="+", help="Input ERA5 GRIB file(s)")
    parser.add_argument("--output", "-o", required=True, help="Output Zarr directory")
    parser.add_argument("--variables", nargs="+", default=["u", "v", "t", "q"])
    parser.add_argument("--chunks", default="1,21", help="Comma-separated chunk sizes (time,level)")
    args = parser.parse_args()

    chunk_parts = [int(x) for x in args.chunks.split(",")]
    chunks = {"time": chunk_parts[0], "isobaricInhPa": chunk_parts[1]}

    logging.basicConfig(level=logging.INFO)
    out = prepare_era5_zarr(args.gribs, args.output, variables=args.variables, chunks=chunks)
    print(f"Wrote Zarr store: {out}")


if __name__ == "__main__":
    main()
