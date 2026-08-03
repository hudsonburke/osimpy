"""MocoTrack tool wrapper.

Provides a Pydantic-based settings model that configures and runs
``osim.MocoTrack`` for state-tracking optimal control problems.

Unlike traditional OpenSim tools (ID, IK, CMC), MocoTrack does not use the
``tool.printToXML() → tool.run()`` pattern.  Instead it follows:
``track.initialize() → study.solve()``.  This module adapts that workflow
into the same ``Settings → Result`` contract used by the rest of osimpy.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import opensim as osim
import polars as pl
from pydantic import BaseModel, Field, FilePath, PrivateAttr

from ..io import STOMetadata
from ..io.sto import sto_to_df
from .model_processing import (
    CoordinateReserveOptimalForceOverride,
    build_moco_model_processor,
)

logger = logging.getLogger(__name__)


class MocoTrackResult(BaseModel):
    """Result from a MocoTrack solve."""

    success: bool
    objective: float | None = None
    num_iterations: int | None = None
    solver_duration_s: float | None = None

    solution_file: FilePath | None = None
    results_directory: Path | None = None

    start_time: datetime | None = None
    end_time: datetime | None = None

    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def load_solution(self) -> tuple[pl.DataFrame, STOMetadata]:
        if self.solution_file is None:
            raise FileNotFoundError("No solution file (solve may have failed)")
        return sto_to_df(self.solution_file)



class CoordinateConstraint(BaseModel):
    """Pin a coordinate to a fixed value ± tolerance."""

    name: str
    value: float
    tolerance: float = 1e-6



class StateWeight(BaseModel):
    """Override tracking weight for a specific state."""

    state_path: str = Field(
        description="Full state path, e.g. '/jointset/ground_spine/sacrum_y/value'"
    )
    weight: float



class ControlWeightPattern(BaseModel):
    """Apply a weight multiplier to controls matching a regex pattern."""

    pattern: str = Field(description="Regex matched against control names")
    weight: float



class MocoTrackSettings(BaseModel):
    """MocoTrack configuration.

    This is the public API for configuring and running a MocoTrack
    optimal control problem.  All OpenSim-specific wiring is handled
    internally by :meth:`run`.

    The ``model_operators`` list controls the ``ModelProcessor`` pipeline.
    By default it includes the operators needed for DeGrooteFregly2016
    muscles and reserve actuators — the standard Moco recipe.
    """

    name: str = "moco_track"

    model_path: FilePath
    coordinates_path: FilePath = Field(
        description="IK .mot/.sto with coordinate trajectories"
    )
    external_loads_path: FilePath | None = None
    results_directory: Path = Field(default_factory=lambda: Path.cwd())
    solution_filename: str = "moco_solution.sto"

    initial_time: float | None = None
    final_time: float | None = None

    replace_muscles_with_dgf: bool = True
    ignore_tendon_compliance: bool = True
    ignore_passive_fiber_forces: bool = True
    active_fiber_force_scale_width: float = 1.5
    dgf_fiber_damping: float | None = None
    reserve_optimal_force: float = 0.1
    rigid_tendon_muscle_names: list[str] = Field(default_factory=list)
    coordinate_reserve_optimal_force_overrides: list[
        CoordinateReserveOptimalForceOverride
    ] = Field(default_factory=list)
    muscle_path_set_file: FilePath | None = None

    states_global_tracking_weight: float = 10.0
    allow_unused_references: bool = True
    track_reference_position_derivatives: bool = False
    control_effort_weight: float = 0.001
    coordinates_in_degrees: bool = True

    mesh_interval: float = 0.02
    convergence_tolerance: float = 1e-3
    constraint_tolerance: float = 1e-4
    max_iterations: int = 1000
    verbosity: int = 2

    coordinate_constraints: list[CoordinateConstraint] = Field(default_factory=list)
    state_weights: list[StateWeight] = Field(default_factory=list)
    control_weight_patterns: list[ControlWeightPattern] = Field(default_factory=list)
    initial_guess_file: FilePath | None = None

    _solution: osim.MocoSolution | None = PrivateAttr(default=None)

    def _build_model_processor(self) -> osim.ModelProcessor:
        return build_moco_model_processor(
            model_path=Path(self.model_path),
            external_loads_path=(
                Path(self.external_loads_path)
                if self.external_loads_path is not None
                else None
            ),
            replace_muscles_with_dgf=self.replace_muscles_with_dgf,
            ignore_tendon_compliance=self.ignore_tendon_compliance,
            ignore_passive_fiber_forces=self.ignore_passive_fiber_forces,
            active_fiber_force_scale_width=self.active_fiber_force_scale_width,
            reserve_optimal_force=self.reserve_optimal_force,
            muscle_path_set_file=(
                Path(self.muscle_path_set_file)
                if self.muscle_path_set_file is not None
                else None
            ),
            rigid_tendon_muscle_names=self.rigid_tendon_muscle_names,
            dgf_fiber_damping=self.dgf_fiber_damping,
            coordinate_reserve_optimal_force_overrides=self.coordinate_reserve_optimal_force_overrides,
            results_directory=Path(self.results_directory),
            temp_model_stem=self.name,
        )

    def _build_states_reference(self) -> osim.TableProcessor:
        tp = osim.TableProcessor(str(self.coordinates_path))
        if self.coordinates_in_degrees:
            tp.append(osim.TabOpConvertDegreesToRadians())
        tp.append(osim.TabOpUseAbsoluteStateNames())
        return tp

    @staticmethod
    def _get_state_names(problem: osim.MocoProblem) -> list[str]:
        model = problem.getPhase(0).getModelProcessor().process()
        model.initSystem()
        sv = model.getStateVariableNames()
        return [sv.get(i) for i in range(sv.getSize())]

    def _apply_coordinate_constraints(self, problem: osim.MocoProblem) -> None:
        if not self.coordinate_constraints:
            return
        state_names = self._get_state_names(problem)
        for cc in self.coordinate_constraints:
            path = next(
                (s for s in state_names if f"/{cc.name}/value" in s), None
            )
            if path is None:
                logger.warning("No state path found for coordinate %s", cc.name)
                continue
            problem.setStateInfo(
                path,
                osim.MocoBounds(cc.value - cc.tolerance, cc.value + cc.tolerance),
            )

    def _apply_state_weights(self, problem: osim.MocoProblem) -> None:
        if not self.state_weights:
            return
        goal = osim.MocoStateTrackingGoal.safeDownCast(
            problem.updGoal("state_tracking")
        )
        if goal is None:
            logger.warning("Could not find 'state_tracking' goal for weight overrides")
            return
        for sw in self.state_weights:
            goal.setWeightForState(sw.state_path, sw.weight)

    def _apply_control_weight_patterns(self, problem: osim.MocoProblem) -> None:
        if not self.control_weight_patterns:
            return
        goal = osim.MocoControlGoal.safeDownCast(
            problem.updGoal("control_effort")
        )
        if goal is None:
            logger.warning("Could not find 'control_effort' goal for control patterns")
            return
        for cwp in self.control_weight_patterns:
            goal.setWeightForControlPattern(cwp.pattern, cwp.weight)

    def run(self) -> MocoTrackResult:
        results_dir = Path(self.results_directory)
        results_dir.mkdir(parents=True, exist_ok=True)

        warnings: list[str] = []
        errors: list[str] = []
        start = datetime.now()
        prev_dir = os.getcwd()

        try:
            track = osim.MocoTrack()
            track.setName(self.name)
            track.setModel(self._build_model_processor())
            track.setStatesReference(self._build_states_reference())

            track.set_states_global_tracking_weight(
                self.states_global_tracking_weight
            )
            track.set_allow_unused_references(self.allow_unused_references)
            track.set_track_reference_position_derivatives(
                self.track_reference_position_derivatives
            )
            track.set_control_effort_weight(self.control_effort_weight)

            if self.initial_time is not None:
                track.set_initial_time(self.initial_time)
            if self.final_time is not None:
                track.set_final_time(self.final_time)
            track.set_mesh_interval(self.mesh_interval)

            study = track.initialize()
            problem = study.updProblem()

            self._apply_coordinate_constraints(problem)
            self._apply_state_weights(problem)
            self._apply_control_weight_patterns(problem)

            solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
            solver.set_optim_convergence_tolerance(self.convergence_tolerance)
            solver.set_optim_constraint_tolerance(self.constraint_tolerance)
            solver.set_optim_max_iterations(self.max_iterations)
            solver.set_verbosity(self.verbosity)

            if self.initial_guess_file is not None:
                traj = osim.MocoTrajectory(str(self.initial_guess_file))
                solver.setGuess(traj)

            os.chdir(str(results_dir))
            t0 = time.time()
            solution = study.solve()
            elapsed = time.time() - t0

            if solution.isSealed():
                solution.unseal()

            sol_path = results_dir / self.solution_filename
            solution.write(str(sol_path))
            self._solution = solution

            return MocoTrackResult(
                success=True,
                objective=solution.getObjective(),
                num_iterations=solution.getNumIterations(),
                solver_duration_s=elapsed,
                solution_file=sol_path,
                results_directory=results_dir,
                start_time=start,
                end_time=datetime.now(),
                warnings=warnings,
            )

        except Exception as e:
            errors.append(str(e))
            logger.error("MocoTrack failed: %s", e)
            return MocoTrackResult(
                success=False,
                results_directory=results_dir,
                start_time=start,
                end_time=datetime.now(),
                errors=errors,
            )
        finally:
            os.chdir(prev_dir)

    def cli_cmd(self) -> None:
        print(self.run().model_dump_json(indent=2, exclude_none=True))
