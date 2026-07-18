
# Project Icarus - Interceptor Trajectory & Aerodynamic Optimizer

## Overview

This repository provides a high-fidelity, surrogate-based optimization pipeline designed for aerospace interceptor trajectory analysis. By bridging the gap between computational fluid dynamics (CFD) and multidisciplinary design optimization (MDO), this project enables precise modeling of interceptor flight phases—from boost and midcourse guidance to terminal homing.

The software implements a tiered approach consistent with sophisticated missile defense architectures, facilitating the computational evaluation of performance requirements for platforms such as the Arrow, David’s Sling, or Iron Dome interceptors.

## Technical Architecture

The pipeline utilizes a modular, multi-fidelity approach to solve the complex interactions between flight dynamics and structural/aerodynamic constraints.

* **Aerodynamic Generation**: Solves Navier-Stokes equations using **FEniCS/DOLFINx** to generate high-fidelity force coefficients ($C_d$, $C_l$, $C_m$).


* **Surrogate Modeling**: Employs **Gaussian Process Regression (Kriging)** to create continuous, differentiable surfaces from discrete HDF5 data tables. This ensures rapid evaluation during trajectory optimization while retaining physical accuracy.


* **Trajectory Optimization**: Utilizes **OpenMDAO** to manage the mission-level trajectory, incorporating surrogate models to predict real-time aerodynamic impacts on flight paths.



## Pipeline Workflow

1. **Data Generation**:
* Run simulation sweeps in `src/aero_data/`.
* Outputs are saved as structured **HDF5** datasets, capturing state variables like Mach number, angle of attack, and altitude.


2. **Surrogate Training**:
* Execute `src/training/train_gpr.py` to fit the Gaussian Process model to the aerodynamic dataset.
* The model exports a `.pkl` artifact for seamless integration into the optimization loop.


3. **Optimization**:
* Define mission constraints (e.g., intercept altitude, time-to-impact) in `src/optimization/trajectory_optimization.py`.
* The OpenMDAO driver minimizes the objective function, utilizing the surrogate model to handle aerodynamic load prediction.



## Prerequisites

This project requires an environment configured for scientific computing and high-performance MDO:

* **FEniCS/DOLFINx** (for finite element analysis).
* **OpenMDAO** (v3.x+) for multidisciplinary optimization.
* **scikit-learn** (for Gaussian Process implementation).
* **h5py** (for HDF5 data management).

## Usage

To initialize the optimization routine using the default aerodynamic surrogate:

```bash
# 1. Generate aerodynamic coefficients
python src/aero/generate_aero_data.py

# 2. Train the surrogate model
python src/training/train_gpr.py

# 3. Run the trajectory optimization
python src/optimization/trajectory_optimization.py

```

## Safety & Defense Context

This software is intended for theoretical modeling and simulation of interceptor performance characteristics. Users must ensure that all simulations comply with applicable regional research protocols and safety requirements for defense-related technologies.

The integration of surrogate-based uncertainty quantification is built-in; specifically, the GPR model should be used to penalize optimization trajectories that enter flight regimes where high-fidelity training data is sparse, ensuring safer and more robust flight planning.

---

This project is designed to support the development of multi-tiered defense system logic.