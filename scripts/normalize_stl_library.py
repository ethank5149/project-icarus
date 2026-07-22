"""One-time normalization of all CAD STL meshes in the manifest.

For each entry in ``project_icarus.aero.stl_loader.CAD_MANIFEST`` this script:
  1. Reads the raw STL.
  2. Applies the manifest scale/axis transformations.
  3. Recenters the centroid to the origin.
  4. Aligns the longest axis to +Z.
  5. Writes the canonical normalized STL to ``reference/vehicles/<key>/model.stl``.
  6. Re-reads the written file and verifies orientation/centering tolerances.

Run:
    PYTHONPATH=/mnt/user/public/project-icarus /config/miniconda3/envs/.conda-venv/bin/python scripts/normalize_stl_library.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from project_icarus.aero.stl_loader import (
    CAD_MANIFEST,
    ensure_normalized,
    load_cad_vehicle,
    normalized_path_for,
    normalize_stl,
    read_stl,
    read_stl_for_key,
    surface_area,
    write_stl,
)


def _verify_normalized(path: Path, tolerance: float = 1e-3) -> dict:
    """Re-read a normalized STL and assert centering + axis alignment."""
    vertices, faces = read_stl(str(path))
    centroid = vertices.mean(axis=0)
    extents = vertices.max(axis=0) - vertices.min(axis=0)

    primary = int(np.argmax(extents))
    ok_axis = primary == 2
    ok_center = np.linalg.norm(centroid) < tolerance
    ok_finite = bool(np.all(np.isfinite(vertices)))
    ok_area = surface_area(vertices, faces) > 0.0

    return {
        "path": str(path),
        "vertices": int(vertices.shape[0]),
        "faces": int(faces.shape[0]),
        "centroid_norm": float(np.linalg.norm(centroid)),
        "extents": [float(x) for x in extents],
        "primary_axis": primary,
        "ok_axis": ok_axis,
        "ok_center": ok_center,
        "ok_finite": ok_finite,
        "ok_area": ok_area,
    }


def main():
    keys = sorted(CAD_MANIFEST.keys())
    print(f"Normalizing {len(keys)} CAD vehicles...")
    results = []

    t0 = time.time()
    for key in keys:
        info = CAD_MANIFEST[key]
        print(f"\n[{key}] {info.proxy_for or key} <- {info.filename}")
        try:
            out_path = normalized_path_for(key)
            if out_path.exists():
                print(f"  EXISTS: {out_path}")
                v, f = read_stl(str(out_path))
                area = surface_area(v, f)
                print(f"  Re-read: vertices={v.shape[0]}, faces={f.shape[0]}, area={area:.2f} m^2")
                stats = _verify_normalized(out_path)
                ok = stats["ok_axis"] and stats["ok_center"] and stats["ok_finite"] and stats["ok_area"]
                print(f"  Verify: axis={stats['ok_axis']}, center={stats['ok_center']}, finite={stats['ok_finite']}, area={stats['ok_area']}")
                if not ok:
                    print("  REGENERATING due to failed verification")
                    out_path = ensure_normalized(key, force=True)
                    stats = _verify_normalized(out_path)
                    ok = stats["ok_axis"] and stats["ok_center"] and stats["ok_finite"] and stats["ok_area"]
            else:
                out_path = ensure_normalized(key, force=True)
                stats = _verify_normalized(out_path)
                ok = stats["ok_axis"] and stats["ok_center"] and stats["ok_finite"] and stats["ok_area"]
                print(f"  Wrote: {out_path}")
                print(f"  Verify: axis={stats['ok_axis']}, center={stats['ok_center']}, finite={stats['ok_finite']}, area={stats['ok_area']}")

            z_len = stats["extents"][2]
            print(f"  Z-length: {z_len:.3f} m")
            results.append({"key": key, "ok": ok, **stats})
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results.append({"key": key, "ok": False, "error": str(exc)})

    elapsed = time.time() - t0
    ok_count = sum(1 for r in results if r.get("ok"))
    print(f"\nDone. {ok_count}/{len(results)} verified in {elapsed:.1f}s")
    for r in results:
        status = "OK" if r.get("ok") else f"FAIL: {r.get('error', 'verification failed')}"
        print(f"  {r['key']}: {status}")


if __name__ == "__main__":
    main()
