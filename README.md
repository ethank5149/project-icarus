# Project Icarus - Multi-Tier 6-DOF Trajectory Optimization & End-to-End Simulation

## Overview

A high-fidelity, surrogate-based trajectory optimization pipeline for endo- and exo-atmospheric interceptors. It supports multi-target engagements with shared upstream states, multi-output GPR surrogates with uncertainty quantification, and kill-vehicle (MKV) separation events.

Inspired by tiered missile defense architectures (Arrow 2/3, David's Sling, Iron Dome), it models boost, midcourse, and terminal phases with 6-DOF Newton-Euler dynamics and unit quaternion attitude. An end-to-end simulation engine enables Monte Carlo sweeps over interceptor configurations, guidance laws, and target trajectories including fractional orbital bombardment (FOBS), hypersonic glide vehicles (HGV), suppressed trajectories, and swarms.

The terminal phase is supported by a selectable guidance backend (classic / augmented / zero-effort-miss / SDRE-MPC), a UKF-tracked seeker that closes the guidance loop, a calibrated RV-vs-decoy discrimination model, and realistic interceptor presets (Arrow-3, Tamir, GMD) with uncertainty-quantification sampling.

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
- Boost guidance: gravity turn / pitch-over. Midcourse guidance: true proportional navigation (PN) with data-link updates.
- Kill assessment: hit-to-kill (< 0.5 m) vs blast-frag (< 10 m) thresholds.

### Terminal Guidance & Seeker (2B)

The terminal phase is driven by `TerminalGuidance` (`src/guidance/terminal_guidance.py`), which selects a backend via `GuidanceConfig.terminal_guidance_law`:

- **`pn`** — classic proportional navigation, `a = N · Vc · LOṠ`.
- **`apn`** — augmented PN, adds `(N/2)·a_target` (gravity-compensated target acceleration bias) to counter target maneuver/gravity.
- **`zem`** — zero-effort-miss guidance; steers `ZEM = z + t_go·ż` (linear, constant-closing-speed) to null with a navigation-ratio gain, `t_go` from range / |Vc|.
- **`sdre_mpc`** — SDRE-based MPC-lite: finite-horizon LQR on the linearized relative double-integrator kinematics, closed-form gain `K = [√2·√(q/r)·t_go·I, √(qv/r)·t_go·I]`.

`SeekerModel` (`src/guidance/seeker.py`) models an active/semi-active radar or imaging-IR seeker: noisy LOS (az/el) measurements with glint, Poisson-gated clutter, RCS scintillation, and one-frame latency; a cosine-cone FOV/gimbal mask; and a self-contained Unscented Kalman Filter (P-scaled Merwe sigma points) that tracks relative position/velocity and returns a smoothed LOS rate. When a seeker is attached, `runner.py` advances it each terminal step and feeds the UKF LOS rate into the selected backend via `commanded_accel_seeker(...)`; otherwise it falls back to the analytic PN law. `DiscriminationModel` provides calibrated RV-vs-decoy class-conditional Gaussian likelihoods over `[RCS_bias, IR_flux, Doppler_width, micro_motion_flag]` (micro-motion is Bernoulli), with `calibrate()` re-estimation from labelled data and a `log_likelihood_ratio` / `is_rv` decision.

### End-to-End Simulation
- **Target families**: ballistic, FOBS (2-body patched conic), HGV (skip-glide), suppressed (deep dip + evasion), swarm (clustered RVs).
- **Separate-object architecture**: `InterceptorConfig`, `GuidanceLaw`, `TargetScenario`, `EngagementScenario`.
- **Monte Carlo runner**: adaptive `scipy.integrate.RK45` closed-loop integration with guidance loop; initial `v0` computed from launch-site / threat-axis geometry. State perturbations applied to position, velocity, and mass.
- **Batch sweep**: `run_sweep()` over interceptor × target × scenario grids with optional `joblib` parallelization.
- **Result API**: `EngagementResult` with built-in 3D trajectory plotting, miss-distance histogram, and kill-probability analysis.
- **Geodetic presets**: WGS84 coordinate conversion with realistic launch sites (USA interceptor bases, Russia/China target regions).

### Real-World Location Database
- `reference/locations.yml` holds ~110 real-world geodetic records (lat/lon/alt) tagged by `designation`: `defended-target` (homeland/C2/nuclear/ICBM sites), `interceptor-launch-site` (GMD, Aegis Ashore, THAAD bases), and `target-launch-site` (Russian/Chinese ICBM/SLBM garrisons and silo fields).
- `reference/locations.py` loads the YAML and provides `locations_by_designation`, `locations_by_name`, `coordinates_to_ecef`, and a `_sanitize_key` helper.
- `src/scenarios/presets.py` merges this database into the preset libraries at import: every `interceptor-launch-site` becomes an interceptor preset (precise coordinates override the rough built-ins), every `target-launch-site` becomes a `ballistic_target_*` preset, and every `defended-target` becomes a `defended_*` preset usable as an interceptor aim point (its geodetic `target_launch_site` is set) and as a threatened location for trajectory modeling.
- **Threat→defended trajectories**: `geodetic_launch_to_target()` computes a great-circle launch azimuth and surface range from a threat site to a defended point, then derives a range-consistent launch velocity rotated into the local East/North/Up frame. `build_threat_to_defended(threat, defended, scenario_type=...)` exposes every threat family aimed at a defended point: `ballistic` (range-derived speed), `fobs` (fractional-orbit boost + deorbit steered toward the aim point), `hgv` (hypersonic glide inserted at glide altitude), `suppressed` (midcourse jinking along the threat axis), and `swarm` (spread RVs). A curated set of `threat_*_to_*` presets (e.g. `threat_kozelsk_to_washington_dc`, `..._fobs`, `..._hgv`, `..._suppressed`) registers all four families for 17 representative pairs. `FOBSScenario.propagate` was fixed so reentry steers toward the true aim point (previously hardcoded to a fixed location), and `SuppressedScenario` midcourse maneuvers now act along the horizontal threat axis instead of a hardcoded `+y`.

- **Real closed-loop FOBS reentry**: the FOBS reentry is now a genuine PN-guided, EOM6DOF solve rather than the ad-hoc steering term. `FOBSScenario` caches a full boost → coast → deorbit → guided-reentry trajectory; the reentry is integrated with `simulate_guided_threat()`, which uses `scipy.integrate.solve_ivp` (adaptive RK45 with a ground-impact terminal event) driving the repo's `EOM6DOF` and `TerminalGuidance` (proportional navigation). The threat is the controlled body and the defended aim point is the PN target. `GuidedThreatConfig` exposes the RV mass/inertia/area/aero-boundary and guidance `accel_limit`/`N`. The PN law's interceptor-style seeker FOV gate is bypassed for threat homing (`commanded_accel(..., disable_fov=True)`).

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
│   ├── seeker.py
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
reference/
├── locations.yml            # Real-world geodetic database of defended targets,
│                            #   interceptor bases, and foreign launch complexes
└── locations.py            # YAML loader + ECEF/key helpers
tests/
├── test_eom.py
├── test_guidance.py
├── test_surrogate.py
├── test_optimization.py
├── test_sim.py
├── test_dymos.py
├── test_seeker.py
├── test_guidance_backends.py
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
from src.guidance.law import GuidanceLaw
from src.scenarios.presets import (
    interceptor_preset,
    target_preset,
    set_interceptor_geodetic,
    set_target_geodetic,
    build_interceptor_config,
    sample_interceptor_uq,
)
from src.scenarios.scenario import EngagementScenario
from src.sim.api import run_engagement, run_sweep

# Built-in interceptor presets return (InterceptorConfig, GuidanceConfig).
# Each bundles a multi-stage thrust model and a calibrated terminal backend.
cfg, guidance_cfg = build_interceptor_config("arrow3")   # APN, IR seeker
guidance = GuidanceLaw(config=guidance_cfg)

launch_site = interceptor_preset("vandenberg")           # 34.7°N, 120.6°W
preset = target_preset("ballistic_target_moscow")         # 55.8°N, 37.6°E
scenario = EngagementScenario(
    interceptor_launch_site=launch_site,
    **preset.engagement.__dict__,
)

result = run_engagement(cfg, guidance, preset.target, scenario, n_trials=50)
print(result.miss_distance, result.kill_assessment)
result.plot_3d()
result.plot_miss_distance_distribution()

# Select a different terminal backend at runtime.
custom_guidance = GuidanceLaw.from_dict({"terminal_guidance_law": "sdre_mpc"})

# UQ: draw a ±20% log-normal perturbation of mass/Isp/divert for a Monte Carlo.
import numpy as np
uq_cfg, uq_g = sample_interceptor_uq("gmd", rng=np.random.default_rng(0), frac=0.20)

# Custom geodetic site / target
custom_site = set_interceptor_geodetic(28.4, -80.6, 0.0)  # Cape Canaveral
custom_target = set_target_geodetic(39.9, 116.4, 0.0)     # Beijing

# Batch sweep
sweep = run_sweep(
    interceptors=[cfg],
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

**Interceptor vehicle presets (`build_interceptor_config(name)` → `(InterceptorConfig, GuidanceConfig)`):**
- `arrow3` — Arrow-3 exoatmospheric hit-to-kill; 2-stage (booster + sustainer), IR seeker, APN terminal guidance.
- `tamir` — Iron Dome Tamir endoatmospheric point-defense; single-stage, radar seeker, SDRE-MPC terminal guidance (blast-frag).
- `gmd` — GMD GBII exoatmospheric EKV hit-to-kill; 3-stage, IR seeker, ZEM terminal guidance.

Each preset uses OSINT-approximate parameters (illustrative research defaults, not controlled data). `sample_interceptor_uq(name, rng, frac=0.20)` returns a ±20% log-normal-perturbed copy (mass, stage Isp, divert impulse) for Monte Carlo / sensitivity analysis. The returned `GuidanceConfig` selects one of the four terminal backends via `terminal_guidance_law`.

## Limitations & Remaining Steps

### Known limitations
- **Dymos optimization suite is not runnable.** `tests/test_dymos.py` is excluded from CI. `src/optimization/trajectory_optimization.py` uses a Dymos API invocation (`num_nodes` option) that no longer matches the installed Dymos/OpenMDAO version, and the engagement `runner.py` is not yet wired in as a callable ODE for the optimizer. The per-phase ODE components (`boost_phase.py`, `midcourse_phase.py`, `terminal_phase.py`) import and run, but the top-level problem assembly fails.
- **Aero surrogate is synthetic.** GPR is trained on analytical `blended_aero` data, not real wind-tunnel/CFD databases (DATCOM, CBA, FEniCS/DOLFINx). `Cn` and `Cl_roll` are analytic by design and bypass the GPR.
- **Guidance coupling is staged by time**, not by true event detection. `runner.py` switches boost→midcourse→terminal at fixed `t` thresholds (60 s / 180 s) rather than detecting thrust cutoff / dry mass / altitude as the plan specifies.
- **MKV / stage-separation events are modeled but not injected into the runner integration loop.** `StageSeparation` and `MKVSystem` exist and are unit-capable, but the RK45 runner does not yet check `eom.separations`/`eom.mkv` at each step to apply impulses and mass drops.
- **J3/J4 are not independently active.** They appear only inside the J2 gravity factor term; they are not applied as separate spherical-harmonic perturbations.
- **Thrust model** — interceptor presets now use true multi-stage `StageSpec` sequencing (Arrow-3 2-stage, GMD 3-stage); the legacy `EOM6DOF.thrust_profile` scalar path remains for ad-hoc configs.
- **Atmosphere thermosphere** uses a fixed solar-activity factor (500 K exobase); no real F10.7/ap indexing.
- **Decoy/target discrimination** — `DiscriminationModel` (RV-vs-decoy Gaussian class-conditionals over RCS bias / IR flux / Doppler width / micro-motion) is now calibrated-capable via `calibrate()`; OSINT-approximate priors are illustrative, not flight-rated.

### Remaining steps (post physics-fidelity plan)
1. **Wire Dymos problem assembly** to the current OpenMDAO API (`num_nodes` → phase transcription) and integrate `EngagementRunner` as the terminal-phase ODE so the optimizer can minimize miss distance.
2. **Event-driven phase transitions** in `runner.py`: detect boost→midcourse on `thrust < threshold` or `m < dry_mass`; midcourse→terminal on `altitude < 100 km` / `range < 50 km` / MKV separation.
3. **Inject separation / MKV events** into the RK45 step loop (apply mass drops and velocity/attitude impulses at `t` crossings).
4. **Calibrate aero** against a real database or FEniCS-generated training data; add analytic GPR Jacobian derivatives (complex-step already used for OpenMDAO partials).
5. **Independent J3/J4** spherical-harmonic terms, and solar-activity-driven thermosphere.
6. **WGS84 altitude** in the EOM already uses Bowring's method; extend gravity-gradient torque and inertial→geodetic routines for full WGS84 geopotential.
7. **Threat library** — expand `DiscriminationModel` priors per real threat signatures (2C.2); add decoy/`decoy_model.py` bodies to target families and exercise the threat discrimination loop end-to-end.

## Safety & Defense Context

This software is intended for theoretical modeling and simulation of interceptor performance characteristics. Users must ensure all simulations comply with applicable regional research protocols and safety requirements for defense-related technologies.

The integrated GPR-based uncertainty quantification penalizes trajectories that enter sparse-data regimes, promoting robust flight planning.

---

This project supports the development of multi-tiered defense system logic.
