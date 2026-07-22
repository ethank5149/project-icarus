from __future__ import annotations

import argparse
import cdsapi
import os
from typing import List, Optional
from tqdm import tqdm
import requests


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, ".cdsapirc")


def _read_cds_config() -> tuple[Optional[str], Optional[str]]:
    url = None
    key = None
    with open(CONFIG_PATH, 'r') as f:
        for line in f.read().splitlines():
            if line.startswith('url:'):
                url = line.split(':', 1)[1].strip()
            elif line.startswith('key:'):
                key = line.split(':', 1)[1].strip()
    return url, key


DEFAULT_TIMES_6H = ['00:00', '06:00', '12:00', '18:00']
DEFAULT_TIMES_1H = [f"{h:02d}:00" for h in range(24)]


def download_month(
    cds_client: cdsapi.Client,
    year: str,
    month: str,
    days: List[str],
    times: List[str],
    output_dir: str = ".",
) -> str:
    output_filename = f"era5_global_native_025_{year}_{month}.grib"
    output_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_path):
        print(f"Skipping existing: {output_path}")
        return output_path

    print(f"\nProcessing Batch: {year}-{month}")
    try:
        result = cds_client.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'format': 'grib',
                'variable': ['temperature', 'u_component_of_wind', 'v_component_of_wind',
                             'specific_humidity'],
                'pressure_level': [
                    '1', '2', '3', '5', '7', '10', '20', '30', '50', '70',
                    '100', '150', '200', '250', '300', '400', '500',
                    '700', '850', '925', '1000',
                ],
                'year': [year],
                'month': [month],
                'day': days,
                'time': times,
            }
        )
        download_url = result.location
        response = requests.get(download_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        print(f"Server processing complete. Streaming {total_size / (1024**2):.2f} MB to disk...")

        with open(output_path, 'wb') as file, tqdm(
            desc=output_filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))

        print(f"Successfully finalized: {output_path}")
        return output_path
    except Exception as e:
        raise RuntimeError(f"Failed to download {year}-{month}: {e}") from e


def main(
    years: Optional[List[str]] = None,
    months_per_year: Optional[List[str]] = None,
    output_dir: str = ".",
    hourly: bool = False,
    days: Optional[List[str]] = None,
):
    if years is None:
        years = ['2015']
    if months_per_year is None:
        months_per_year = [f"{m:02d}" for m in range(1, 13)]
    if days is None:
        days = [f"{d:02d}" for d in range(1, 32)]

    times = DEFAULT_TIMES_1H if hourly else DEFAULT_TIMES_6H

    url, key = _read_cds_config()
    c = cdsapi.Client(url=url, key=key)

    for year in years:
        for month in months_per_year:
            try:
                download_month(c, year, month, days, times, output_dir)
            except RuntimeError as exc:
                print(str(exc))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download ERA5 reanalysis pressure-level data")
    parser.add_argument("--years", nargs="+", default=["2015"], help="Years to download")
    parser.add_argument("--months", nargs="+", default=None, help="Months (02d format)")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--hourly", action="store_true", help="Download hourly (24 snapshots/day) instead of 6-hourly")
    args = parser.parse_args()
    main(years=args.years, months_per_year=args.months, output_dir=args.output_dir, hourly=args.hourly)
