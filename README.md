# Project Icarus - Multi-Tier 6-DOF Trajectory Optimization

## Overview

A high-fidelity, surrogate-based trajectory optimization pipeline for endo- and exo-atmospheric interceptors. It supports multi-target engagements with shared upstream states, multi-output GPR surrogates with uncertainty quantification, and kill-vehicle (MKV) separation events.

Inspired by tiered missile defense architectures (Arrow 2/3, David's Sling, Iron Dome), it models boost, midcourse, and terminal phases with 6-DOF Newton-Euler dynamics and unit quaternion attitude.

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

### Multi-Target / Decoy
- Shared upstream boost/midcourse states; per-target terminal phase groups.
- Decoy release, seeker discrimination, and kill probability across threat space.

## Pipeline Workflow

1. **Data Generation**: `src/aero/generate_aero_data.py` generates synthetic Mach/α/β/altitude data with regime flags in HDF5.
2. **Surrogate Training**: `src/surrogates/train_gpr.py` fits multi-output GPR and exports `.pkl`.
3. **Optimization**: `src/optimization/trajectory_optimization.py` wires Dymos phases, surrogates, and guidance into an OpenMDAO problem.
4. **Validation**: Run `pytest` in `.conda-venv`.

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
│   └── terminal_guidance.py
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
├── targets/
│   ├── target_model.py
│   └── decoy_model.py
├── constraints.py
tests/
├── test_eom.py
├── test_guidance.py
├── test_surrogate.py
├── test_optimization.py
requirements.txt
```

## Prerequisites

- Python 3.11 in `.conda-venv`
- FEniCS/DOLFINx (optional; synthetic data used by default)
- OpenMDAO, Dymos, scikit-learn, h5py

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

## Safety & Defense Context

This software is intended for theoretical modeling and simulation of interceptor performance characteristics. Users must ensure all simulations comply with applicable regional research protocols and safety requirements for defense-related technologies.

The integrated GPR-based uncertainty quantification penalizes trajectories that enter sparse-data regimes, promoting robust flight planning.

---

This project supports the development of multi-tiered defense system logic.
