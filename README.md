# Project Icarus - Multi-Tier 6-DOF Trajectory Optimization & End-to-End Simulation

## Overview

A high-fidelity, surrogate-based trajectory optimization pipeline for endo- and exo-atmospheric interceptors. It supports multi-target engagements with shared upstream states, multi-output GPR surrogates with uncertainty quantification, and kill-vehicle (MKV) separation events.

Inspired by tiered missile defense architectures (Arrow 2/3, David's Sling, Iron Dome), it models boost, midcourse, and terminal phases with 6-DOF Newton-Euler dynamics and unit quaternion attitude. An end-to-end simulation engine enables Monte Carlo sweeps over interceptor configurations, guidance laws, and target trajectories including fractional orbital bombardment (FOBS), hypersonic glide vehicles (HGV), suppressed trajectories, and swarms.

## Technical Architecture

### 6-DOF Dynamics
- **Newton-Euler** equations of motion with state `r` (inertial ECEF), `v` (inertial), `q` (body→inertial quaternion), `omega` (body), `m`.
- **Quaternion attitude** with normalization and norm-preserving kinematics.
- **Frame convention**: `v` is inertial; aerodynamic/gravity forces are evaluated in the body frame and rotated body→inertial (`R_b2i @ F_body`) before dividing by mass. Drag opposes velocity.
- **Regime toggle**: endo-atmospheric (drag + aero coefficients) vs exo-atmospheric (Newtonian, zero drag) with a smooth cosine taper at 100 km.
- **Gravity**: inverse-square law with a J2 toggle (active above ~50 km; J3/J4 included but currently contribute only through the J2 factor term).
- **Atmosphere**: US Standard Atmosphere 1976 piecewise layers (0–100 km, barometric with per-layer lapse rate) and a realistic thermosphere above 100 km (temperature relaxes 200 K → 500 K, exponential density with ~50 km scale height, Sutherland viscosity).
- **Thrust/MKV**: `Isp`-based mass flow (`mdot = -T / (Isp·g0)`), gimbaled thrust vector with first-order gimbal-rate limits, stage-separation impulses with optional spin, and MKV spring-ejection separation (relative velocity along bus x-axis) with 85 N gimbaled divert thrusters.

### Surrogate Modeling
- **Multi-output GPR** with composite kernels (`RBF + WhiteKernel`), outputs **Cd, Cy, Cm** (body-axis: drag, side force, pitch moment).
- **Analytical post-processing**: `Cn` (yaw moment) and `Cl_roll` (roll moment) plus damping derivatives are computed analytically from Newtonian/linear-viscous models, not from the GPR.
- **Coefficient convention** (standard missile/aircraft body axis, x-forward / y-right / z-down):
  - `Cd` — drag (−x body), `Cy` — side force (+y body), `Cm` — pitch moment (about +y)
  - `Cn` — yaw moment (about +z), `Cl_roll` — roll moment (about +x)
- **Uncertainty propagation** through EOM to miss-distance statistics (Monte Carlo + elementary-effects sensitivity).
- Regime-dependent data coverage validated against analytical baselines.

### Trajectory Optimization
- **Dymos** with **Radau transcription** for discontinuous events (stage separation, MKV sep).
- Explicit phases: **Boost**, **Midcourse**, **Terminal** with `link_phases` continuity. Each phase ODE uses the shared `EOM6DOF` and `blended_aero` surrogate (passing `boundary_alt` / `surrogate_path` through options).
- Guidance laws: gravity turn / pitch-over (boost), **true proportional navigation (PN)** with data-link updates (midcourse), PN with seeker FOV check and likelihood-ratio discrimination (terminal).
- Kill assessment: hit-to-kill (< 0.5 m) vs blast-frag (< 10 m) thresholds.

### End-to-End Simulation
- **Target families**: ballistic, FOBS (2-body patched conic), HGV (skip-glide), suppressed (deep dip + evasion), swarm (clustered RVs).
- **Separate-object architecture**: `InterceptorConfig`, `GuidanceLaw`, `TargetScenario`, `EngagementScenario`.
- **Monte Carlo runner**: adaptive `scipy.integrate.RK45` closed-loop integration with guidance loop; initial `v0` computed from launch-site / threat-axis geometry. State perturbations applied to position, velocity, and mass.
- **Batch sweep**: `run_sweep()` over interceptor × target × scenario grids with optional `joblib` parallelization.
- **Result API**: `EngagementResult` with built-in 3D trajectory plotting, miss-distance histogram, and kill-probability analysis.
- **Geodetic presets**: WGS84 coordinate conversion with realistic launch sites (USA interceptor bases, Russia/China target regions).

### Jupyter Notebooks
- `notebooks/engagement_sweep.ipynb` — high-level workflow: define configs, run single engagement, run batch sweep, visualize.
- `notebooks/interactive_dashboard.ipynb` — ipywidgets dashboard with preset dropdowns for interceptor launch sites and target regions, geodetic lat/lon/alt inputs, sliders for interceptor mass, kill radius, guidance gain N, and MC trials. Runs engagement on button click and displays 3D trajectory, miss-distance distribution, and kill/no-kill counts.

## File Structure

```
src/
├── aero/
│   ├── generate_aero_data.py
│   ├── aero_analytical.py
├── dynamics/
│   ├── eom_6dof.py
│   ├── coordinate_systems.py
│   ├── gravity.py
│   ├── thrust.py
│   └── atmosphere.py
├── guidance/
│   ├── boost_guidance.py
│   ├── midcourse_guidance.py
│   ├── terminal_guidance.py
│   └── law.py
├── surrogates/
│   ├── train_gpr.py
│   ├── aero_surrogate.py
│   └── uncertainty.py
├── optimization/
│   ├── trajectory_optimization.py
│   └── phases/
│       ├── boost_phase.py
│       ├── midcourse_phase.py
│       └── terminal_phase.py
├── interceptors/
│   └── config.py
├── scenarios/
│   ├── target_factory.py
│   ├── scenario.py
│   └── presets.py          # Geodetic presets + WGS84 helpers
├── sim/
│   ├── runner.py
│   ├── sweep.py
│   └── api.py
├── targets/
│   ├── target_model.py
│   └── decoy_model.py
├── constraints.py
tests/
├── test_eom.py
├── test_guidance.py
├── test_surrogate.py
├── test_optimization.py
├── test_sim.py
├── test_dymos.py
notebooks/
├── engagement_sweep.ipynb
└── interactive_dashboard.ipynb
requirements.txt
```

## Prerequisites

- Python 3.11 in `.conda-venv`
- FEniCS/DOLFINx (optional; synthetic data used by default)
- OpenMDAO, Dymos, scikit-learn, h5py, ipywidgets, pandas

## Usage

```bash
# Activate environment
source /config/miniconda3/bin/activate .conda-venv

# 1. Generate aerodynamic coefficients
python src/aero/generate_aero_data.py

# 2. Train surrogate model
python src/surrogates/train_gpr.py

# 3. Run tests (exclude the Dymos optimization suite; see Limitations)
python -m pytest tests/ -v --ignore=tests/test_dymos.py
```

### Python API

```python
from src.interceptors.config import InterceptorConfig
from src.guidance.law import GuidanceLaw
from src.scenarios.presets import (
    interceptor_preset,
    target_preset,
    set_interceptor_geodetic,
    set_target_geodetic,
)
from src.scenarios.scenario import EngagementScenario
from src.sim.api import run_engagement, run_sweep

# Use built-in geodetic presets
interceptor = InterceptorConfig(name="Arrow 3", mass=1000.0, kill_radius=0.5)
guidance = GuidanceLaw()
launch_site = interceptor_preset("vandenberg")       # 34.7°N, 120.6°W
preset = target_preset("ballistic_target_moscow")     # 55.8°N, 37.6°E
scenario = EngagementScenario(
    interceptor_launch_site=launch_site,
    **preset.engagement.__dict__,
)

result = run_engagement(interceptor, guidance, preset.target, scenario, n_trials=50)
print(result.miss_distance, result.kill_assessment)
result.plot_3d()
result.plot_miss_distance_distribution()

# Custom geodetic site
custom_site = set_interceptor_geodetic(28.4, -80.6, 0.0)  # Cape Canaveral
custom_target = set_target_geodetic(39.9, 116.4, 0.0)     # Beijing

# Batch sweep
sweep = run_sweep(
    interceptors=[interceptor],
    targets=[preset.target, custom_target.target, ...],
    scenarios=[scenario, ...],
    n_trials=30,
)
df = sweep.to_dataframe()
```

### Available Presets

**Interceptor launch sites (USA / US Indo-Pacific):**
- `vandenberg` — 34.7°N, 120.6°W (CA)
- `cape_canaveral` — 28.4°N, 80.6°W (FL)
- `kwajalein` — 8.7°N, 167.7°E (Marshall Islands)
- `schriever` — 38.8°N, 104.5°W (CO)
- `fort_greely` — 63.9°N, 147.6°W (AK)
- `clear_sfs` — 64.3°N, 149.2°W (AK)
- `custom_geodetic` — user-defined lat/lon/alt via `set_interceptor_geodetic()`

**Target regions (Russia / China):**
- Russia: `ballistic_target_moscow`, `ballistic_target_novosibirsk`, `ballistic_target_vladivostok`, `ballistic_target_murmansk`, `ballistic_target_yakutsk`
- China: `ballistic_target_beijing`, `ballistic_target_shanghai`, `ballistic_target_xian`, `ballistic_target_chengdu`, `ballistic_target_urumqi`

## Limitations & Remaining Steps

### Known limitations
- **Dymos optimization suite is not runnable.** `tests/test_dymos.py` is excluded from CI. `src/optimization/trajectory_optimization.py` uses a Dymos API invocation (`num_nodes` option) that no longer matches the installed Dymos/OpenMDAO version, and the engagement `runner.py` is not yet wired in as a callable ODE for the optimizer. The per-phase ODE components (`boost_phase.py`, `midcourse_phase.py`, `terminal_phase.py`) import and run, but the top-level problem assembly fails.
- **Aero surrogate is synthetic.** GPR is trained on analytical `blended_aero` data, not real wind-tunnel/CFD databases (DATCOM, CBA, FEniCS/DOLFINx). `Cn` and `Cl_roll` are analytic by design and bypass the GPR.
- **Guidance coupling is staged by time**, not by true event detection. `runner.py` switches boost→midcourse→terminal at fixed `t` thresholds (60 s / 180 s) rather than detecting thrust cutoff / dry mass / altitude as the plan specifies.
- **MKV / stage-separation events are modeled but not injected into the runner integration loop.** `StageSeparation` and `MKVSystem` exist and are unit-capable, but the RK45 runner does not yet check `eom.separations`/`eom.mkv` at each step to apply impulses and mass drops.
- **J3/J4 are not independently active.** They appear only inside the J2 gravity factor term; they are not applied as separate spherical-harmonic perturbations.
- **Thrust model** is single-stage with a scalar `thrust_profile`; ullage motors and true multi-stage mass sequencing are out of scope (per plan §5).
- **Atmosphere thermosphere** uses a fixed solar-activity factor (500 K exobase); no real F10.7/ap indexing.
- **Decoy/target discrimination** uses a likelihood-ratio stub over a 4-feature vector, not calibrated sensor models.

### Remaining steps (post physics-fidelity plan)
1. **Wire Dymos problem assembly** to the current OpenMDAO API (`num_nodes` → phase transcription) and integrate `EngagementRunner` as the terminal-phase ODE so the optimizer can minimize miss distance.
2. **Event-driven phase transitions** in `runner.py`: detect boost→midcourse on `thrust < threshold` or `m < dry_mass`; midcourse→terminal on `altitude < 100 km` / `range < 50 km` / MKV separation.
3. **Inject separation / MKV events** into the RK45 step loop (apply mass drops and velocity/attitude impulses at `t` crossings).
4. **Calibrate aero** against a real database or FEniCS-generated training data; add analytic GPR Jacobian derivatives (complex-step already used for OpenMDAO partials).
5. **Independent J3/J4** spherical-harmonic terms, and solar-activity-driven thermosphere.
6. **WGS84 altitude** in the EOM already uses Bowring's method; extend gravity-gradient torque and inertial→geodetic routines for full WGS84 geopotential.
7. **Calibrated discrimination** features (RCS bias, IR flux, Doppler width, micro-motion) from sensor models.

## Safety & Defense Context

This software is intended for theoretical modeling and simulation of interceptor performance characteristics. Users must ensure all simulations comply with applicable regional research protocols and safety requirements for defense-related technologies.

The integrated GPR-based uncertainty quantification penalizes trajectories that enter sparse-data regimes, promoting robust flight planning.

---

This project supports the development of multi-tiered defense system logic.
