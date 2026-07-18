# Project Icarus - Multi-Tier 6-DOF Trajectory Optimization & End-to-End Simulation

## Overview

A high-fidelity, surrogate-based trajectory optimization pipeline for endo- and exo-atmospheric interceptors. It supports multi-target engagements with shared upstream states, multi-output GPR surrogates with uncertainty quantification, and kill-vehicle (MKV) separation events.

Inspired by tiered missile defense architectures (Arrow 2/3, David's Sling, Iron Dome), it models boost, midcourse, and terminal phases with 6-DOF Newton-Euler dynamics and unit quaternion attitude. An end-to-end simulation engine enables Monte Carlo sweeps over interceptor configurations, guidance laws, and target trajectories including fractional orbital bombardment (FOBS), hypersonic glide vehicles (HGV), suppressed trajectories, and swarms.

## Technical Architecture

### 6-DOF Dynamics
- **Newton-Euler** equations in body frame.
- **Unit quaternion** attitude representation with normalization.
- **Regime toggle**: endo-atmospheric (drag + aero coefficients) vs exo-atmospheric (Newtonian, zero drag) with smooth taper at 100 km.
- **Gravity**: inverse-square law with J2 extension point.
- **Thrust/MKV**: mass flow, gimbal limits, stage separation impulses, MKV separation with 85 adj/sec divert thrusters.

### Surrogate Modeling
- **Multi-output GPR** (Cd, Cl, Cm) with composite kernels (`RBF + WhiteKernel`).
- **Uncertainty propagation** through EOM to miss-distance statistics (Monte Carlo + Sobol sensitivity).
- Regime-dependent data coverage validated against analytical baselines.

### Trajectory Optimization
- **Dymos** with **Radau transcription** for discontinuous events (stage separation, MKV sep).
- Explicit phases: **Boost**, **Midcourse**, **Terminal** with `link_phases` continuity.
- Guidance laws: gravity turn (boost), PN with data-link (midcourse), APN/PN (terminal).
- Kill assessment: hit-to-kill (< 0.5 m) vs blast-frag (< 10 m) thresholds.

### End-to-End Simulation
- **Target families**: ballistic, FOBS (2-body patched conic), HGV (skip-glide), suppressed (deep dip + evasion), swarm (clustered RVs).
- **Separate-object architecture**: `InterceptorConfig`, `GuidanceLaw`, `TargetScenario`, `EngagementScenario`.
- **Monte Carlo runner**: closed-loop `solve_ivp` integration with guidance loop, perturbing initial conditions and parameters.
- **Batch sweep**: `run_sweep()` over interceptor × target × scenario grids with optional `joblib` parallelization.
- **Result API**: `EngagementResult` with built-in 3D trajectory plotting, miss-distance histogram, and kill-probability analysis.

### Jupyter Notebooks
- `notebooks/engagement_sweep.ipynb` — high-level workflow: define configs, run single engagement, run batch sweep, visualize.
- `notebooks/interactive_dashboard.ipynb` — ipywidgets dashboard with sliders for interceptor mass, kill radius, guidance gain N, target type, and MC trials. Runs engagement on button click and displays 3D trajectory, miss-distance distribution, and kill/no-kill counts.

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
│   └── scenario.py
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

# 3. Run tests
python -m pytest tests/ -v
```

### Python API

```python
from src.interceptors.config import InterceptorConfig
from src.guidance.law import GuidanceLaw
from src.scenarios.target_factory import FOBSScenario
from src.scenarios.scenario import EngagementScenario
from src.sim.api import run_engagement, run_sweep

interceptor = InterceptorConfig(name="Arrow 3", mass=1000.0, kill_radius=0.5)
guidance = GuidanceLaw()
target = FOBSScenario.from_orbital_params(apoapsis_km=200.0, inclination_deg=30.0)
scenario = EngagementScenario(engagement_end=300.0)

result = run_engagement(interceptor, guidance, target, scenario, n_trials=50)
print(result.miss_distance, result.kill_assessment)
result.plot_3d()
result.plot_miss_distance_distribution()

# Batch sweep
sweep = run_sweep(
    interceptors=[interceptor],
    targets=[target, ...],
    scenarios=[scenario, ...],
    n_trials=30,
    parallel=True,
)
df = sweep.to_dataframe()
```

## Safety & Defense Context

This software is intended for theoretical modeling and simulation of interceptor performance characteristics. Users must ensure all simulations comply with applicable regional research protocols and safety requirements for defense-related technologies.

The integrated GPR-based uncertainty quantification penalizes trajectories that enter sparse-data regimes, promoting robust flight planning.

---

This project supports the development of multi-tiered defense system logic.
