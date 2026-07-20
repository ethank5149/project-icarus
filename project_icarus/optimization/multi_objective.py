"""Phase 4: pymoo multi-objective trajectory optimization stretch.

Wraps the existing Dymos/OpenMDAO closed-loop terminal phase as a pymoo
:class:`~pymoo.core.problem.Problem` so the optimizer can explore the
Pareto front of miss distance vs. control effort (and optionally vs.
phase duration). Uses ``ClosedLoopTerminalODE`` directly to keep the
guidance loop intact.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import openmdao.api as om
import dymos as dm

from project_icarus.optimization.phases.terminal_phase import ClosedLoopTerminalODE
from project_icarus.guidance.terminal_guidance import TerminalGuidance


class MultiObjectiveTerminalProblem:
    """Build a pymoo Problem from the Dymos closed-loop terminal phase.

    Decision variables
    ------------------
    * ``N``          – proportional navigation gain (2 .. 8)
    * ``accel_limit`` – interceptor acceleration limit m/s^2 (50 .. 250)
    * ``t_duration`` – terminal phase duration s (10 .. 60) [optional]

    Objectives (minimize)
    ---------------------
    * ``miss_distance`` – final miss distance (m)
    * ``accel_rms``     – RMS commanded acceleration over the phase (m/s^2)
    * ``t_duration``    – phase duration (s) [optional]

    The OpenMDAO problem is cached between evaluations so pymoo sees a
    pure python callable with no external process spawn overhead.
    """

    def __init__(
        self,
        n_seg: int = 6,
        law: str = "pn",
        include_time_objective: bool = False,
        initial_guess: Optional[np.ndarray] = None,
    ):
        self.n_seg = n_seg
        self.law = law
        self.include_time = include_time_objective
        self.initial_guess = initial_guess

        self._p: Optional[om.Problem] = None
        self._drivers: list = []

    def _build_problem(self) -> om.Problem:
        p = om.Problem()
        traj = dm.Trajectory()
        p.model.add_subsystem("traj", traj, promotes=["*"])

        tx = dm.Radau(num_segments=self.n_seg, order=3)
        term = dm.Phase(ode_class=ClosedLoopTerminalODE, transcription=tx)
        traj.add_phase("term", term)
        term.options["ode_init_kwargs"] = {"law": self.law}

        term.set_time_options(fix_initial=True, duration_bounds=(10.0, 60.0), units="s")
        term.add_state("r", rate_source="dr_dt", units="m", shape=(3,),
                       fix_initial=True, fix_final=False,
                       val=np.array([120e3, 0.0, 6471e3]))
        term.add_state("v", rate_source="dv_dt", units="m/s", shape=(3,),
                       fix_initial=True, fix_final=False,
                       val=np.array([-1500.0, 0.0, 0.0]))
        term.add_state("q", rate_source="dq_dt", units=None, shape=(4,),
                       fix_initial=True, fix_final=False,
                       val=np.array([1.0, 0.0, 0.0, 0.0]))
        term.add_state("omega", rate_source="domega_dt", units="rad/s", shape=(3,),
                       fix_initial=True, fix_final=False, val=np.zeros(3))
        term.add_state("m", rate_source="dm_dt", units="kg",
                       fix_initial=True, fix_final=False, val=200.0)

        term.add_control("accel_x", units="m/s**2", opt=False, val=0.0)
        term.add_control("accel_y", units="m/s**2", opt=False, val=0.0)
        term.add_control("accel_z", units="m/s**2", opt=False, val=0.0)

        term.add_parameter("target_r", units="m", shape=(3,),
                           val=np.array([100e3, 0.0, 6471e3]), opt=False)
        term.add_parameter("target_v", units="m/s", shape=(3,), val=np.zeros(3), opt=False)
        term.add_parameter("N", units=None, val=4.0, opt=True,
                           lower=2.0, upper=8.0)
        term.add_parameter("accel_limit", units="m/s**2", val=150.0, opt=True,
                           lower=50.0, upper=250.0)

        term.add_timeseries_output("miss_distance")
        term.add_timeseries_output("cmd_accel")
        term.add_objective("miss_distance", scaler=1.0, loc="final")

        p.driver = om.ScipyOptimizeDriver(optimizer="SLSQP")
        p.driver.declare_coloring()
        p.driver.opt_settings["maxiter"] = 20
        p.setup()

        p.set_val("traj.term.t_duration", 30.0)
        return p

    def _evaluate(self, x: np.ndarray) -> tuple[float, float, float]:
        n_val = float(x[0])
        accel_val = float(x[1])
        t_dur = float(x[2]) if self.include_time else 30.0

        if self._p is None:
            self._p = self._build_problem()

        p = self._p
        p.set_val("traj.term.N", n_val)
        p.set_val("traj.term.accel_limit", accel_val)
        p.set_val("traj.term.t_duration", t_dur)

        dm.run_problem(p, run_driver=True, simulate=False, make_plots=False)

        miss = float(np.asarray(p.get_val("traj.term.timeseries.miss_distance")).ravel()[-1])
        cmd = np.asarray(p.get_val("traj.term.timeseries.cmd_accel"))
        accel_rms = float(np.sqrt(np.mean(np.sum(cmd ** 2, axis=1)))) if cmd.size else 0.0

        return miss, accel_rms, t_dur

    def __call__(self, x: np.ndarray) -> np.ndarray:
        miss, accel_rms, t_dur = self._evaluate(x)
        if self.include_time:
            return np.array([miss, accel_rms, t_dur])
        return np.array([miss, accel_rms])


def build_pymoo_problem(
    n_var: int = 2,
    n_obj: int = 2,
    n_seg: int = 6,
    law: str = "pn",
) -> "pymoo.core.problem.Problem":
    """Construct a pymoo Problem around the Dymos closed-loop terminal phase.

    Parameters
    ----------
    n_var
        Number of decision variables: 2 (N, accel_limit) or 3 (add t_duration).
    n_obj
        Number of objectives: 2 (miss, accel_rms) or 3 (add t_duration).
    n_seg
        Transcription segments for the terminal phase.
    law
        Guidance law: "pn", "apn", "zem", or "sdre_mpc".

    Returns
    -------
    pymoo.core.problem.Problem
    """
    from pymoo.core.problem import Problem
    from pymoo.core.variable import Real

    include_time = (n_var == 3) and (n_obj == 3)

    xl = np.array([2.0, 50.0, 10.0]) if include_time else np.array([2.0, 50.0])
    xu = np.array([8.0, 250.0, 60.0]) if include_time else np.array([8.0, 250.0])

    evaluator = MultiObjectiveTerminalProblem(
        n_seg=n_seg, law=law, include_time_objective=include_time
    )

    class _MOProblem(Problem):
        def _evaluate(self, X, out, *args, **kwargs):
            F = np.array([evaluator(xi) for xi in X])
            out["F"] = F

    return _MOProblem(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu)
