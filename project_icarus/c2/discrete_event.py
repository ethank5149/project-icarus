"""Phase 5B discrete-event C2 orchestration.

Drives the sensor layer (``SensorNetwork``) and the ``BattleManager`` through a
time-stepped scenario. Each scan the network produces M-of-N-confirmed tracks;
the ``BattleManager`` ingests them (with C2 latency + data-link refresh) and
fires interceptor shots whose kill outcome is supplied by a caller callback.

The loop is deliberately dependency-free: it is a pure-python fixed-step
simulator. ``simpy`` is supported *only* as an optional drop-in clock if a
``simpy`` ``Environment`` is passed in; otherwise the built-in stepper runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Sequence
import numpy as np


@dataclass
class C2Scenario:
    """Time-discretised engagement scenario fed to ``run_discrete_event``."""
    name: str = "c2_scenario"
    t_start: float = 0.0
    t_end: float = 600.0
    dt: float = 1.0  # scan cadence (s)


def run_discrete_event(
    scenario: C2Scenario,
    network: Any,  # SensorNetwork
    bm: Any,  # BattleManager
    truth_states: Callable[[float], Sequence[np.ndarray]],
    assess: Callable[[Any, int], float],
    rcs_m2: float = 1.0,
    env: Any = None,  # optional simpy.Environment
) -> Dict[str, Any]:
    """Step the sensor network and C2 loop over the scenario timeline.

    Parameters
    ----------
    scenario : C2Scenario
        Timeline (t_start/t_end/dt).
    network : SensorNetwork
        Pre-built with sensors; produces M-of-N confirmed tracks via ``scan``.
    bm : BattleManager
        Pre-built with batteries + doctrine config (latency/refresh set there).
    truth_states : callable
        ``truth_states(t) -> list[ECEF ndarray]`` ground-truth target positions.
    assess : callable
        ``assess(threat_track, battery_index) -> miss_m`` kill-outcome callback
        used by ``BattleManager.run_with_tracks``.
    rcs_m2 : float
        Target RCS fed to the sensor network.
    env : simpy.Environment, optional
        If provided, the loop uses ``env`` as the clock (advanced externally).
        When ``None``, a plain python stepper drives time.

    Returns
    -------
    dict with keys: ``t`` (scan times), ``n_confirmed`` (per-scan confirmed
    count), ``battle`` (final BattleResult), ``shots`` (per-scan shot log).
    """
    times: List[float] = []
    n_confirmed: List[int] = []
    shot_log: List[Dict[str, Any]] = []

    t = scenario.t_start

    def step(t_now: float):
        targets = list(truth_states(t_now))
        network.scan(targets, rcs_m2=rcs_m2, t=t_now)
        confirmed = network.confirmed_tracks()
        times.append(t_now)
        n_confirmed.append(len(confirmed))
        if not confirmed:
            return
        # C2 latency: only act if the scenario has advanced past the first
        # contact by at least c2_latency_s (models sensor->C2->weapon delay).
        if t_now - scenario.t_start < bm.cfg.c2_latency_s:
            return
        result = bm.run_with_tracks(confirmed, assess)
        for s in result.shots:
            shot_log.append({"t": t_now, **s})
        return

    if env is not None:
        # simpy-driven: the event loop's own clock advances time via
        # ``env.timeout(dt)`` each scan, so other simpy processes can run
        # concurrently (e.g. a data-link refresh generator). ``step`` reads
        # ``env.now`` so the scenario timeline is the simpy clock.
        def _proc():
            while env.now <= scenario.t_end:
                yield env.timeout(scenario.dt)
                step(env.now)
            # Final partial step if the clock landed exactly on t_end.
            if env.now <= scenario.t_end:
                step(env.now)
        env.process(_proc())
        env.run(until=scenario.t_end + scenario.dt)
    else:
        while t <= scenario.t_end:
            step(t)
            t += scenario.dt

    # Cumulative battle state: declared threats carry persistent ``defeated``
    # flags (propagated by ``run_with_tracks``), so we summarise from them
    # rather than re-running (which would find nothing left to engage).
    from .battle_manager import BattleResult
    final = BattleResult(threats=bm.threats, batteries=bm.batteries, shots=shot_log)
    return {
        "t": np.asarray(times),
        "n_confirmed": np.asarray(n_confirmed),
        "battle": final,
        "shots": shot_log,
    }
