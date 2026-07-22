"""Generate DOLFINx-based aero tables from STL meshes and train GPR surrogates.

This script runs a structured (Mach, alpha, beta, altitude, delta) sweep for
each confirmed CAD vehicle and writes:
  - HDF5 coefficient table: reference/vehicles/<key>/aero.h5
  - GPR surrogate pickle: reference/vehicles/<key>/surrogate.pkl

Vehicles are selected from project_icarus.aero.stl_loader.CAD_MANIFEST.
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import numpy as np

from project_icarus.aero.cfd_generators import (
    SweepSpec,
    run_sweep,
    save_sweep_hdf5,
)
from project_icarus.aero.geometry import get_vehicle
from project_icarus.aero.stl_loader import CAD_MANIFEST
from project_icarus.surrogates.train_gpr import (
    default_model_path,
    default_h5_path,
    train_vehicle_gpr,
)


SWEEP_GRID = dict(
    mach_range=(0.5, 5.0, 6),
    alpha_range=(-10.0, 10.0, 5),
    beta_range=(-5.0, 5.0, 3),
    altitude_range=(0.0, 100e3, 4),
    delta_range=(0.0, 0.0, 1),
)


def generate_and_train(key: str) -> dict:
    info = CAD_MANIFEST[key]
    print(f"[{datetime.now():%H:%M:%S}] Generating aero table for {key} ({info.proxy_for})...")
    t0 = time.time()

    spec = SweepSpec(
        vehicle=key,
        backend="dolfinx",
        **SWEEP_GRID,
    )
    result = run_sweep(spec)
    elapsed = time.time() - t0
    n_points = result["coeffs"].shape[0]
    print(f"  Sweep done: {n_points} points in {elapsed:.1f}s ({elapsed/n_points:.2f}s/point)")

    h5_path = default_h5_path(info.proxy_for)
    os.makedirs(os.path.dirname(os.path.abspath(h5_path)), exist_ok=True)
    save_sweep_hdf5(result, h5_path)
    print(f"  Saved HDF5 -> {h5_path}")

    print(f"  Training GPR surrogate for '{info.proxy_for}'...")
    model = train_vehicle_gpr(info.proxy_for)
    model_path = default_model_path(info.proxy_for)
    print(f"  Surrogate saved -> {model_path}")

    pred = model.predict(np.array([[1.5, 2.5, 0.0, 10e3, 0.0]]))
    print(f"  Sanity prediction @ M=1.5 alpha=2.5: Cd={pred[0,0]:.4f} Cy={pred[0,1]:.4f}")

    return {
        "key": key,
        "vehicle": info.proxy_for,
        "n_points": n_points,
        "sweep_time_s": elapsed,
        "h5_path": h5_path,
        "model_path": model_path,
    }


def main():
    confirmed = [k for k, info in CAD_MANIFEST.items() if info.proxy_for]
    if not confirmed:
        print("No confirmed vehicles in CAD_MANIFEST.")
        return

    summary = []
    for key in confirmed:
        try:
            summary.append(generate_and_train(key))
        except Exception as exc:
            print(f"  FAILED {key}: {exc}")
            summary.append({"key": key, "error": str(exc)})

    print("\n=== Generation Summary ===")
    for row in summary:
        if "error" in row:
            print(f"  {row['key']}: FAILED ({row['error']})")
        else:
            print(
                f"  {row['key']} -> {row['vehicle']}: "
                f"{row['n_points']} points, {row['sweep_time_s']:.1f}s, "
                f"{row['h5_path']}, {row['model_path']}"
            )


if __name__ == "__main__":
    main()
