"""RRA (Residual Reduction Algorithm) tool wrapper.

Provides a Pydantic-based settings model that can:
  1. Build an ``osim.RRATool`` with all fields properly wired.
  2. Save the setup XML, ``os.chdir`` into the XML directory, reload,
     and run — so that relative paths in the XML resolve correctly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import polars as pl

import opensim as osim
from pydantic import Field, FilePath
from ..io import STOMetadata
from .tool import ToolResult, ToolSettings

logger = logging.getLogger(__name__)


class RRAResult(ToolResult):
    """Result from Residual Reduction Algorithm analysis."""

    kinematics_file: FilePath | None = Field(
        None, description="Path to output kinematics file (*_Kinematics_q.sto)"
    )
    actuation_force_file: FilePath | None = Field(
        None, description="Path to output actuation forces (*_Actuation_force.sto)"
    )
    position_error_file: FilePath | None = Field(
        None, description="Path to output position errors (*_pErr.sto)"
    )
    adjusted_model_file: FilePath | None = Field(
        None, description="Path to COM-adjusted output model (.osim)"
    )

    def load_kinematics(self) -> tuple[pl.DataFrame, STOMetadata]:
        """Load the output kinematics file as a DataFrame."""
        return self._load_sto(self.kinematics_file)

    def load_actuation_forces(self) -> tuple[pl.DataFrame, STOMetadata]:
        """Load the output actuation forces file as a DataFrame."""
        return self._load_sto(self.actuation_force_file)

    def load_position_errors(self) -> tuple[pl.DataFrame, STOMetadata]:
        """Load the output position error file as a DataFrame."""
        return self._load_sto(self.position_error_file)


class RRASettings(ToolSettings[RRAResult]):
    """RRA (Residual Reduction Algorithm) settings.

    All ``FilePath`` fields require the file to exist at construction time
    and are stored as absolute paths internally.  The ``run()`` method
    writes relative paths into the setup XML and ``os.chdir``s into the
    XML directory before executing, so the XML remains portable.

    RRA adjusts the model's mass distribution to reduce residual forces
    and torques that arise from inconsistencies between measured kinematics
    and external forces.

    References
    ----------
    OpenSim RRA User Guide:
    https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53089699
    """

    # ── Input files ──────────────────────────────────────────────────────
    external_loads_path: FilePath | None = Field(
        None, description="Path to external loads XML file"
    )
    force_set_paths: list[FilePath] = Field(
        default_factory=list,
        description="Paths to force set files (.xml) with reserve actuators",
    )
    desired_points_path: FilePath | None = Field(
        None,
        description="Motion/storage file with desired point trajectories",
    )
    desired_kinematics_path: FilePath | None = Field(
        None,
        description="Motion/storage file with desired kinematic trajectories (IK output)",
    )
    task_set_path: FilePath | None = Field(
        None,
        description="File containing RRA tracking tasks (coordinates, weights)",
    )
    constraints_path: FilePath | None = Field(
        None,
        description="File containing control constraints",
    )

    # ── Parameters ───────────────────────────────────────────────────────
    solve_for_equilibrium_for_auxiliary_states: bool = Field(
        False,
        description="Compute equilibrium for states other than coords/speeds",
    )
    lowpass_cutoff_frequency: float = Field(
        -1.0,
        description="Low-pass cutoff (Hz) for desired kinematics; -1 = no filter",
    )

    # ── Integrator settings ──────────────────────────────────────────────
    maximum_number_of_integrator_steps: int = Field(
        20000, description="Maximum number of integrator steps"
    )
    maximum_integrator_step_size: float = Field(
        0.001, description="Maximum integration step size (s)"
    )
    minimum_integrator_step_size: float = Field(
        1e-8, description="Minimum integration step size (s)"
    )
    integrator_error_tolerance: float = Field(
        1e-5, description="Integrator error tolerance"
    )

    # ── Optimizer settings ───────────────────────────────────────────────
    optimizer_algorithm: Literal["ipopt", "cfsqp"] = Field(
        "ipopt", description="Optimizer algorithm"
    )
    numerical_derivative_step_size: float = Field(
        1e-4, description="Step size for numerical derivatives"
    )
    optimization_convergence_tolerance: float = Field(
        1e-5, description="Optimizer convergence tolerance"
    )

    # ── COM adjustment ───────────────────────────────────────────────────
    adjust_com_to_reduce_residuals: bool = Field(
        True,
        description="Adjust body COM positions to minimise residual forces",
    )
    initial_time_for_com_adjustment: float = Field(
        -1.0,
        description="Start time for COM adjustment (-1 = use analysis initial_time)",
    )
    final_time_for_com_adjustment: float = Field(
        -1.0,
        description="End time for COM adjustment (-1 = use analysis final_time)",
    )
    adjusted_com_body: str | None = Field(
        None,
        description="Body whose COM is adjusted (e.g. 'torso', 'spine')",
    )
    output_model_file: str = Field(
        "adjusted_model.osim",
        description="Filename for the COM-adjusted output model",
    )

    # ── Misc ─────────────────────────────────────────────────────────────
    replace_force_set: bool = Field(
        True,
        description="Replace (True) or append (False) model's force set with force_set_files",
    )
    use_verbose_printing: bool = Field(
        False, description="Verbose RRA printing"
    )
    output_precision: int = Field(8, description="Output precision")

    # ── Output file resolution ───────────────────────────────────────────
    def resolve_output_files(self) -> dict[str, Path | None]:
        results_dir = Path(self.results_directory)

        def _first_match(*patterns: str) -> Path | None:
            for pattern in patterns:
                matches = sorted(results_dir.glob(pattern))
                if matches:
                    return matches[0]
            return None

        # RRA also writes an adjusted model — check both results_dir and parent
        adjusted = _first_match(self.output_model_file)
        if adjusted is None:
            # RRA sometimes writes the adjusted model relative to the setup XML
            candidate = results_dir / self.output_model_file
            if candidate.exists():
                adjusted = candidate

        return {
            "kinematics_file": _first_match("*_Kinematics_q.sto"),
            "actuation_force_file": _first_match(
                "*_Actuation_force.sto", "*Actuation_force*.sto"
            ),
            "position_error_file": _first_match("*_pErr.sto"),
            "adjusted_model_file": adjusted,
        }

    # ── Tool creation ────────────────────────────────────────────────────
    def create_tool(self) -> osim.RRATool:
        """Build a fully configured ``osim.RRATool``."""

        if self.setup_path is not None:
            tool = osim.RRATool(str(self.setup_path.resolve()))
        else:
            tool = osim.RRATool()

        rel_model_path = self.get_relative_path(self.model_path)
        rel_results_dir = self.get_relative_path(self.results_directory)

        tool.setModelFilename(rel_model_path)
        tool.setResultsDir(rel_results_dir)

        # ── Time range ───────────────────────────────────────────────
        initial_time = self.initial_time
        final_time = self.final_time
        if initial_time is None or final_time is None:
            try:
                if self.desired_kinematics_path is not None:
                    sto = osim.Storage(str(self.desired_kinematics_path.resolve()))
                    if initial_time is None:
                        initial_time = sto.getFirstTime()
                    if final_time is None:
                        final_time = sto.getLastTime()
                elif self.desired_points_path is not None:
                    trc = osim.MarkerData(str(self.desired_points_path.resolve()))
                    if initial_time is None:
                        initial_time = trc.getStartFrameTime()
                    if final_time is None:
                        final_time = trc.getLastFrameTime()
                else:
                    raise ValueError(
                        "Missing desired kinematics or points file to set missing time range"
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load data to set the time range: {e}"
                ) from e

        tool.setInitialTime(initial_time)
        tool.setFinalTime(final_time)
        tool.setOutputPrecision(self.output_precision)

        # ── Force set ────────────────────────────────────────────────
        tool.setReplaceForceSet(self.replace_force_set)
        if self.force_set_paths:
            arr = osim.ArrayStr()
            for fp in self.force_set_paths:
                arr.append(self.get_relative_path(fp))
            tool.setForceSetFiles(arr)

        # ── External loads ───────────────────────────────────────────
        if self.external_loads_path:
            tool.setExternalLoadsFileName(
                self.get_relative_path(self.external_loads_path)
            )

        # ── Kinematics & tracking ────────────────────────────────────
        if self.desired_points_path:
            tool.setDesiredPointsFileName(
                self.get_relative_path(self.desired_points_path)
            )
        if self.desired_kinematics_path:
            tool.setDesiredKinematicsFileName(
                self.get_relative_path(self.desired_kinematics_path)
            )
        if self.task_set_path:
            tool.setTaskSetFileName(
                self.get_relative_path(self.task_set_path)
            )
        if self.constraints_path:
            tool.setConstraintsFileName(
                self.get_relative_path(self.constraints_path)
            )

        # ── Parameters ───────────────────────────────────────────────
        tool.setSolveForEquilibrium(self.solve_for_equilibrium_for_auxiliary_states)
        tool.setLowpassCutoffFrequency(self.lowpass_cutoff_frequency)
        tool.setAdjustCOMToReduceResiduals(self.adjust_com_to_reduce_residuals)
        if self.adjusted_com_body is not None:
            tool.setAdjustedCOMBody(self.adjusted_com_body)
        tool.setOutputModelFileName(self.output_model_file)

        # ── Integrator ───────────────────────────────────────────────
        tool.setMaximumNumberOfSteps(self.maximum_number_of_integrator_steps)
        tool.setMaxDT(self.maximum_integrator_step_size)
        tool.setMinDT(self.minimum_integrator_step_size)
        tool.setErrorTolerance(self.integrator_error_tolerance)

        # ── Optimizer (no dedicated setters — use PropertyHelper) ────
        p = tool.updPropertyByName("optimizer_algorithm")
        osim.PropertyHelper.setValueString(self.optimizer_algorithm, p)

        p = tool.updPropertyByName("numerical_derivative_step_size")
        osim.PropertyHelper.setValueDouble(self.numerical_derivative_step_size, p)

        p = tool.updPropertyByName("optimization_convergence_tolerance")
        osim.PropertyHelper.setValueDouble(self.optimization_convergence_tolerance, p)

        p = tool.updPropertyByName("initial_time_for_com_adjustment")
        osim.PropertyHelper.setValueDouble(self.initial_time_for_com_adjustment, p)

        p = tool.updPropertyByName("final_time_for_com_adjustment")
        osim.PropertyHelper.setValueDouble(self.final_time_for_com_adjustment, p)

        p = tool.updPropertyByName("use_verbose_printing")
        osim.PropertyHelper.setValueBool(self.use_verbose_printing, p)

        return tool
