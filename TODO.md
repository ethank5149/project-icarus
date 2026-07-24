# TODO

## IN PROGRESS: Fix RS-28 Sarmat Production Guidance Physics

### Immediate Tasks
- [ ] **T1**: Repair `ICBMGuidance._desired_flight_path_angle` to be the live pitch schedule (energy-based, phase-1–4) and wire it into `thrust_direction`; remove dead `gravity_turn_gain`, `max_flight_path_angle`, `_desired_burnout_velocity`, `update_star_sighting`
- [ ] **T2**: Make `SarmatBoostODE` instantiate real `ICBMGuidance` and expose its actual tunables (`pitch_over_start`, `initial_elevation`, `burnout_vmag`) as Dymos parameters; remove synthetic `el_0/el_1/t_cross/T3_scale`
- [ ] **T2**: Make `SarmatScenario` instantiate `ICBMGuidance` directly; remove `PrecomputedTrajectoryProfile`, `_init_icbm_guidance`, `_current_thrust_dir`, and `SarmatTrajectoryLibrary`
- [ ] **T3**: Add `mass_flow` clamping to `MultiStageThrustModel.mass_rate` (`T/(Isp·G0)` clamped to `(wet-dry)/burn_time`)
- [ ] **T3**: Centralize RS-28 stage specs in `thrust.py` factory; both `SarmatScenario` and `SarmatBoostODE` use the same source of truth
- [ ] **T4**: Remove `_fallback_thrust_direction` and `_compute_via_guidance` from `precomputed_trajectory.py`; raise on uncomputed profile
- [ ] **T5**: Run Dymos on corrected production physics; verify `miss_distance < 100 m` at final node
- [ ] **T6**: Update tests for corrected behavior (`test_icbm_guidance.py`, `test_sarmat_physics_validation.py`, `test_sarmat_trajectory.py`)

### Completed
- (none yet)

---

## Out of Scope (deferred, not in this work plan)

- Cython/CUDA acceleration for guidance kernels
- 6-DOF attitude dynamics inside Dymos (`q`, `omega` states remain decoupled from thrust direction)
- FOBS depressed trajectory or MIRV bus modeling
- Replace manual RK4 with OpenMDAO (note: `SarmatScenario._integrate_full` still uses manual RK4 but now routed through fixed `ICBMGuidance`)
- Predictive Impact Point (PIP) midcourse guidance
- GPR aero surrogate extensions to 6-DOF coefficients
- Monte Carlo dispersion harness for CEP characterization
- Gravity inertial transport corrections (`gravity_inertial` ECEF Coriolis/centrifugal verification)
- Altitude/Mach-dependent thrust tables in `MultiStageThrustModel`
- `tests/test_sarmat_mc.py`
