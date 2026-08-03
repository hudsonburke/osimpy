"""MocoInverse tool wrapper.

Provides both the legacy ``solveMocoInverse()`` function and a Pydantic-based
``MocoInverseSettings`` that follows the same contract as ``MocoTrackSettings``.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path

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


class MocoInverseResult(BaseModel):
    """Result from a MocoInverse solve."""

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


class MocoInverseSettings(BaseModel):
    """MocoInverse configuration following the same contract as MocoTrackSettings."""

    name: str = "moco_inverse"

    model_path: FilePath
    coordinates_path: FilePath
    external_loads_path: FilePath | None = None
    results_directory: Path = Field(default_factory=lambda: Path.cwd())
    solution_filename: str = "moco_inverse_solution.sto"

    initial_time: float | None = None
    final_time: float | None = None

    replace_muscles_with_dgf: bool = True
    ignore_tendon_compliance: bool = True
    ignore_passive_fiber_forces: bool = True
    active_fiber_force_scale_width: float = 1.5
    dgf_fiber_damping: float | None = None
    reserve_optimal_force: float = 1.0
    rigid_tendon_muscle_names: list[str] = Field(default_factory=list)
    coordinate_reserve_optimal_force_overrides: list[
        CoordinateReserveOptimalForceOverride
    ] = Field(default_factory=list)
    muscle_path_set_file: FilePath | None = None
    allow_extra_columns: bool = True

    mesh_interval: float = 0.02
    generate_report: bool = True
    bilateral: bool = True

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

    def run(self) -> MocoInverseResult:
        results_dir = Path(self.results_directory)
        results_dir.mkdir(parents=True, exist_ok=True)

        warnings: list[str] = []
        errors: list[str] = []
        start = datetime.now()
        prev_dir = os.getcwd()

        try:
            inverse = osim.MocoInverse()
            model_proc = self._build_model_processor()
            inverse.setModel(model_proc)
            inverse.setKinematics(osim.TableProcessor(str(self.coordinates_path)))

            if self.initial_time is not None:
                inverse.set_initial_time(self.initial_time)
            if self.final_time is not None:
                inverse.set_final_time(self.final_time)
            inverse.set_mesh_interval(self.mesh_interval)
            inverse.set_kinematics_allow_extra_columns(self.allow_extra_columns)

            os.chdir(str(results_dir))
            t0 = time.time()
            solution_wrapper = inverse.solve()
            elapsed = time.time() - t0

            solution = solution_wrapper.getMocoSolution()
            sol_path = results_dir / self.solution_filename
            solution.write(str(sol_path))
            self._solution = solution

            if self.generate_report:
                try:
                    model = model_proc.process()
                    report = osim.report.Report(
                        model, str(sol_path), bilateral=self.bilateral
                    )
                    report.generate()
                except Exception as e:
                    warnings.append(f"Report generation failed: {e}")

            return MocoInverseResult(
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
            logger.error("MocoInverse failed: %s", e)
            return MocoInverseResult(
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


def solveMocoInverse(
    model_file: str,
    external_loads_file: str,
    coordinates_file: str,
    solution_path: str | None = None,
    initial_time: float = 0.0,
    final_time: float = -1.0,
    mesh_interval: float = 0.02,
    active_fiber_force_scale: float = 1.5,
    muscle_path_set_file: str | None = None,
    generate_report: bool = True,
    bilateral: bool = True,
):
    """Legacy functional interface — delegates to MocoInverseSettings."""
    sol_filename = "moco_inverse_solution.sto"
    if solution_path is not None:
        p = Path(solution_path)
        results_dir = p.parent
        sol_filename = p.name
    else:
        results_dir = Path.cwd()
        sol_filename = Path(coordinates_file).stem + "_MocoInverse_solution.sto"

    settings = MocoInverseSettings(
        model_path=Path(model_file),
        coordinates_path=Path(coordinates_file),
        external_loads_path=Path(external_loads_file) if external_loads_file else None,
        results_directory=results_dir,
        solution_filename=sol_filename,
        initial_time=initial_time,
        final_time=final_time if final_time > 0 else None,
        active_fiber_force_scale_width=active_fiber_force_scale,
        muscle_path_set_file=Path(muscle_path_set_file) if muscle_path_set_file else None,
        mesh_interval=mesh_interval,
        generate_report=generate_report,
        bilateral=bilateral,
    )
    return settings.run()
