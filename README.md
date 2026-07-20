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
- **Gravity**: inverse-square law with independent **J2–J10 zonal-harmonic terms** (`gravity_inertial`, EGM2008 low-order coefficients, active above ~50 km). `use_high_order` enables J5–J10; `use_third_body` adds Sun/Moon point-mass perturbations; `use_tides` adds degree-2 solid-Earth tidal acceleration. All zonal terms disable together via `use_j2=False`, and the higher-order/third-body/tide terms default on only where flagged.
- **Atmosphere**: US Standard Atmosphere 1976 piecewise layers (0–100 km, barometric with per-layer lapse rate) below 100 km, blended with a **NRLMSISE-00** thermosphere/exosphere above 100 km (`nrlmsise00` package; analytic exponential thermosphere used as fallback if unavailable). NRLMSISE-00 is driven by the 81-day mean and previous-day F10.7 solar flux, the ap geomagnetic index, and date/time + geodetic latitude/longitude, giving physically realistic solar-activity- and storm-dependent density/temperature.
- **Thrust/MKV**: `Isp`-based mass flow (`mdot = -T / (Isp·g0)`), gimbaled thrust vector with first-order gimbal-rate limits, stage-separation impulses with optional spin, and MKV spring-ejection separation (relative velocity along bus x-axis) with 85 N gimbaled divert thrusters.

### Surrogate Modeling
- **Multi-output GPR** with composite kernels (`RBF + WhiteKernel`), outputs **Cd, Cy, Cm** (body-axis: drag, side force, pitch moment).
- **In-repo aero generation** (`aero/geometry.py`, `aero/cfd_generators.py`): parametric geometry (cone-cylinder-fin + control surfaces) and a DOLFINx 0.9.0 CFD sweep driver producing Cd/Cy/Cm tables; SU2/OpenFOAM are optional pluggable backends (not installed).
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

`SeekerModel` (`src/guidance/seeker.py`) models an active/semi-active radar or imaging-IR seeker: noisy LOS (az/el) measurements with glint, Poisson-gated clutter, RCS scintillation, and one-frame latency; a cosine-cone FOV/gimbal mask; and a self-contained Unscented Kalman Filter (P-scaled Merwe sigma points) that tracks relative position/velocity and returns a smoothed LOS rate. When a seeker is attached, `runner.py` advances it each terminal step and feeds the UKF LOS rate into the selected backend via `commanded_accel_seeker(...)`; otherwise it falls back to the analytic PN law. `DiscriminationModel` provides calibrated RV-vs-decoy class-conditional Gaussian likelihoods over `[RCS_bias, IR_flux, Doppler_width, micro_motion_flag]` (micro-motion is Bernoulli), with `calibrate()` re-estimation from labelled data and a `log_likelihood_ratio` / `is_rv` decision. Every `GuidanceLaw` ships a discriminator pre-calibrated from `ThreatSignatureLibrary.default()` (OSINT-approximate RV/decoy samples), exposed via `GuidanceLaw.discriminate_target(features)`. During a terminal engagement the runner feeds any decoy feature vectors through this discriminator so the interceptor can prefer the RV contact.

### End-to-End Simulation
- **Target families**: ballistic, FOBS (2-body patched conic), HGV (skip-glide), suppressed (deep dip + evasion), swarm (clustered RVs), and `DecoyThreatScenario` (RV + released decoys with `decoy_features()` compatible with `DiscriminationModel`). `ThreatSignatureLibrary` (2C.2) holds OSINT-approximate RV/decoy signature samples for discrimination training.
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
│   ├── geometry.py
│   ├── cfd_generators.py
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
- `nrlmsise00` (optional; NRLMSISE-00 thermosphere — falls back to analytic thermosphere if absent)
- OpenMDAO, Dymos, scikit-learn, h5py, ipywidgets, pandas

## Usage

```bash
# Activate environment
source /config/miniconda3/bin/activate .conda-venv

# (Optional) Generate aerodynamic coefficients / train the GPR surrogate
python src/aero/generate_aero_data.py
python src/surrogates/train_gpr.py

# Run the test suite (the Dymos optimization suite is excluded; see Limitations)
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
- **Dymos optimization suite runs in CI.** `tests/test_dymos.py` assembles a 3-phase boost/midcourse/terminal Radau pseudospectral problem (Dymos 1.15 / OpenMDAO 3.45 API) and drives SLSQP to completion within budget; the terminal phase minimizes intercept time subject to a final-position equality constraint onto the defended aim point (reported as a `miss_distance` timeseries). The SLSQP solve is not guaranteed to fully converge from arbitrary initial guesses, so CI asserts assembly + finite positive phase durations, not optimality. The engagement `runner.py` is not yet wired in as the terminal ODE (objective is time, not direct miss-distance minimization of a closed-loop seeker solution).
- **Aero surrogate default data is synthetic.** `train_gpr.py` trains on analytical `blended_aero` by default; in-repo CFD generation (`aero/geometry.py`, `aero/cfd_generators.py`, DOLFINx 0.9.0) is available but not yet auto-wired to the GPR. `Cn` and `Cl_roll` are analytic by design and bypass the GPR.
- **Phase transitions** are event-driven (not time-based): `ThrustCutoffEvent` (thrust ≈ 0 or `m ≤ dry_mass`) ends boost; `ReentryEvent` (`alt < boundary`) / `RangeEvent` (`range < threshold`) enter terminal. A `t_max` safety cap still applies.
- **MKV / stage-separation events** are injected into the RK45 step loop: `InterceptorConfig._separations` builds `StageSeparation` impulses from multi-stage timing, and `_integrate_trajectory` applies mass drops and Δv/spin at each crossing.
- **Zonal gravity** — `gravity_inertial` applies independent J2–J10 EGM2008 spherical-harmonic zonal terms (`use_high_order` for J5–J10; `use_third_body` adds Sun/Moon point-mass terms; `use_tides` adds degree-2 solid-Earth tides). All zonal terms disable together via `use_j2=False`; higher-order/third-body/tide terms activate only when their flags are set.
- **Thrust model** — interceptor presets now use true multi-stage `StageSpec` sequencing (Arrow-3 2-stage, GMD 3-stage); the legacy `EOM6DOF.thrust_profile` scalar path remains for ad-hoc configs.
- **Atmosphere thermosphere** uses **NRLMSISE-00** (`nrlmsise00` package) driven by F10.7/a (81-day mean + previous-day) and ap, plus date/time and geodetic lat/lon; falls back to the analytic exponential thermosphere if the package is unavailable.
- **Decoy/target discrimination** — `DiscriminationModel` is calibrated-capable via `calibrate()` and every `GuidanceLaw` ships a discriminator pre-trained on `ThreatSignatureLibrary.default()`; OSINT-approximate priors are illustrative, not flight-rated.
- **Sensor layer (Phase 5A)** — `src/sensors/` provides a probabilistic `Sensor` (Pd vs range/RCS with an earth-mask elevation gate), an EKF `Track` maintainer over range/az/el LOS measurements, and a `SensorNetwork` that scans, fuses detections (nearest-neighbour), and builds geodetic coverage masks. `load_sensors_from_locations()` turns `reference/locations.yml` into sensors (radar/UEWR/GBI tagged sites get ~4800 km range). With a single passive sensor, track velocity is only weakly observable, so `Track` matches a constant-velocity prior and stays stable rather than fully converging; multi-sensor fusion (covered by `test_network_fuses_multi_sensor_into_one_track`) recovers observability.
- **Site altitude units** — `reference/locations.yml` stores all `altitude` values in **feet** (see its header). Consumers (`src/scenarios/presets.py`, `src/sensors/network.py`) convert feet→metres via `0.3048` before calling `geodetic_to_ecef` (which expects metres). Keep new YAML entries in feet.

### Remaining steps (post physics-fidelity plan)
1. **Phase 5A/5B/5C (COMPLETE 2026-07-19):** `src/sensors/` provides the sensor network with clutter/false-alarm generation (`Sensor.emit_clutter`, Poisson rate) and M-of-N track confirmation (`SensorNetwork.confirmation_hits` + chi-square + kinematic-consistency gating that rejects random clutter). `src/c2/BattleManager` holds `ThreatTrack`/`Battery`/`BattleManagerConfig`, with greedy + Hungarian (scipy `linear_sum_assignment`) allocators, shoot-look-shoot vs salvo doctrine, C2 latency + data-link-refresh config, and `run_with_tracks` (track-driven, persistent defeats). `run_campaign` extends `run_sweep` to a saturation raid; `run_discrete_event`/`C2Scenario` drive the sensor→C2→weapon loop (pure-python stepper, optional `simpy` 4.1 clock via `env.timeout(dt)`). Reports system metrics (leakage fraction, shots fired, battery utilization) via `EngagementResult`-compatible `CampaignResult`. `tests/test_sensors.py` (12) and `tests/test_c2.py` (12) cover detection, M-of-N clutter rejection, allocation, doctrine, latency, and a track-driven raid (incl. the simpy clock path).
1. **Phase 6 — National "Golden Dome" scale (IN PROGRESS 2026-07-19):** `src/c2/layers.py` adds the tiered architecture (`Tier`/`Layer`/`DefenseArchitecture`) built from `reference/locations.yml` interceptor-launch-sites, with `build_architecture_from_locations` classifying sites into boost/upper/mid/lower regimes and expanding each into per-base `Battery` objects (default GMD/Arrow-3/Tamir mixes). Distributed C2 (`SpaceSensor` SBIRS-like early-warning + `DistributedC2Config`) models the space→ground→battery handoff with `c2_latency_s` cue delay and finite `bandwidth_tracks_per_s` that *drops* tracks under saturation raids; `run_layered_campaign` passes a raid through this chain then runs the existing `BattleManager` doctrine across all layers so boost/upper/mid/lower interceptors can be committed. `BattleManager.run` now honors a per-threat `_cue_latency_s` (round-skip gate). National metrics (`architecture_summary`/`national_metrics`) roll the architecture + raid outcome into readiness/leakage stats; `src/c2/visualization.py` renders the ECEF coverage map in PyVista (bases by tier color, defended + threat points). Parallel backend: `run_campaign`/`run_layered_campaign` fan independent `(threat, battery)` engagements across host cores via **joblib multiprocessing**; transport uses **industry-standard formats** — JSON specs (interceptor/guidance rebuilt by NAME, no lambda thrust profiles cross the boundary) + **HDF5** result persistence (`src/c2/persistence.py`: `save_campaign_hdf5`/`load_campaign_hdf5`). Note: dask-distributed's worker nanny is non-functional in this container, so joblib is the active many-core lever; the RTX 3090 GPU is not used by the scalar RK45 integrator (vectorized batch integration is Phase 7 perf work). `tests/test_layers.py` + `tests/test_persistence.py` cover tiers, handoff saturation, layered-campaign leakage, HDF5 round-trip, and the JSON-spec parallel contract.
2. **Phase 6.4 live dashboard (DONE 2026-07-19):** `src/c2/dashboard.py` is a **Panel** app (`panel serve src/c2/dashboard.py --show`) wrapping the full layered pipeline — raid size, magazine, space-sensor `bandwidth`/`p_detect`, raid arrival spread, and a real-6-DOF (parallel) toggle. It renders the PyVista ECEF coverage map (offscreen PNG) plus national metrics and the per-tier coverage table. Defaults use a synthetic `assess` (instant) so the controls stay responsive; ticking *real engagements* precomputes true per-(threat, battery) miss distances across host cores via the JSON-spec + HDF5 transport.
3. **Wire `EngagementRunner` into the Dymos terminal phase** as the closed-loop ODE so the optimizer directly minimizes miss distance of a seeker-guided solution (currently the terminal phase uses open-loop `accel` controls and minimizes intercept time under a final-position constraint).
2. **Calibrate aero** against a real database or FEniCS-generated training data; add analytic GPR Jacobian derivatives (complex-step already used for OpenMDAO partials).
3. **WGS84 altitude** in the EOM already uses Bowring's method; extend gravity-gradient torque and inertial→geodetic routines for full WGS84 geopotential.
4. **Threat library** — expand `ThreatSignatureLibrary` with more per-class samples (boosted/MBV RVs, advanced decoys) and fold real threat signatures into the `GuidanceLaw` discriminator; add maneuvering-RV aero to target families.

## Safety & Defense Context

This software is intended for theoretical modeling and simulation of interceptor performance characteristics. Users must ensure all simulations comply with applicable regional research protocols and safety requirements for defense-related technologies.

The integrated GPR-based uncertainty quantification penalizes trajectories that enter sparse-data regimes, promoting robust flight planning.

---

This project supports the development of multi-tiered defense system logic.
