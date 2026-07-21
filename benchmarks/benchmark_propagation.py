"""Benchmark trajectory propagation speed.

Compares Python-loop RK4 vs Numba-JIT RK4 for BallisticScenario.
Run with: python -m benchmarks.benchmark_propagation
"""

import time
import numpy as np

from project_icarus.scenarios.target_factory import (
    BallisticScenario,
    HGVScenario,
    SuppressedScenario,
    R_EARTH,
)


def _benchmark_ballistic(n_trials=20):
    r0 = np.array([R_EARTH + 1e5, 0.0, 0.0])
    v0 = np.array([0.0, 0.0, 7000.0])

    tgt = BallisticScenario(r0=r0, v0=v0, adaptive=False)
    t_bench = 300.0

    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        state = tgt.propagate(t_bench)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        assert np.all(np.isfinite(state))

    med = float(np.median(times))
    print(f"BallisticScenario.propagate({t_bench}s): median {med*1000:.2f} ms over {n_trials} trials")


def _benchmark_hgv(n_trials=20):
    r0 = np.array([R_EARTH + 80e3, 0.0, 0.0])
    v0 = np.array([0.0, 0.0, 6000.0])
    tgt = HGVScenario(r0=r0, v0=v0, max_alt_km=80.0, lateral_range_km=1500.0)
    t_bench = 300.0

    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        state = tgt.propagate(t_bench)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        assert np.all(np.isfinite(state))

    med = float(np.median(times))
    print(f"HGVScenario.propagate({t_bench}s): median {med*1000:.2f} ms over {n_trials} trials")


def _benchmark_suppressed(n_trials=20):
    r0 = np.array([R_EARTH + 1e5, 0.0, 0.0])
    v0 = np.array([0.0, 2000.0, 7000.0])
    tgt = SuppressedScenario(r0=r0, v0=v0, midcourse_maneuver_mag=100.0)
    t_bench = 300.0

    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        state = tgt.propagate(t_bench)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        assert np.all(np.isfinite(state))

    med = float(np.median(times))
    print(f"SuppressedScenario.propagate({t_bench}s): median {med*1000:.2f} ms over {n_trials} trials")


if __name__ == "__main__":
    print("Trajectory propagation benchmark")
    print("=" * 50)
    _benchmark_ballistic()
    _benchmark_hgv()
    _benchmark_suppressed()
    print("=" * 50)
