"""CMC (Computed Muscle Control) tool wrapper.

Provides a Pydantic-based settings model that can:
  1. Build an ``osim.CMCTool`` with all fields properly wired.
  2. Save the setup XML, ``os.chdir`` into the XML directory, reload,
     and run — so that relative paths in the XML resolve correctly.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Literal

import opensim as osim
from pydantic import Field, FilePath

from .tool import ToolResult, ToolSettings

logger = logging.getLogger(__name__)


class CMCResult(ToolResult):
    """Result from Computed Muscle Control analysis."""

    controls_file: FilePath = Field(description="Path to output controls file")
    forces_file: FilePath = Field(description="Path to output forces file")
    states_file: FilePath = Field(description="Path to output states file")


class CMCSettings(ToolSettings[CMCResult]):
    """CMC (Computed Muscle Control) settings.

    All ``FilePath`` fields require the file to exist at construction time
    and are stored as absolute paths internally.  The ``run()`` method
    writes relative paths into the setup XML and ``os.chdir``s into the
    XML directory before executing, so the XML remains portable.

    References
    ----------
    OpenSim CMC User Guide:
    https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53089721
    """

    # Input files
    external_loads_path: FilePath | None = Field(
        None, description="Path to external loads XML file"
    )
    force_set_paths: list[FilePath] = Field(
        default_factory=list,
        description="Paths to force set files (.xml) to append actuators",
    )
    desired_points_path: FilePath | None = Field(
        None,
        description="Motion/storage file with desired point trajectories",
    )
    desired_kinematics_path: FilePath | None = Field(
        None,
        description="Motion/storage file with desired kinematic trajectories",
    )
    task_set_path: FilePath | None = Field(
        None,
        description="File containing CMC tracking tasks (coordinates, weights)",
    )
    constraints_path: FilePath | None = Field(
        None,
        description="File containing control constraints",
    )
    rra_controls_path: FilePath | None = Field(
        None,
        description="File with RRA controls to constrain residuals during CMC",
    )

    # Parameters
    solve_for_equilibrium_for_auxiliary_states: bool = Field(
        True,
        description="Compute equilibrium for states other than coords/speeds",
    )
    cmc_time_window: float = Field(
        0.01,
        description="Time window (s) over which desired actuator forces are achieved",
    )
    use_curvature_filter: bool = Field(
        False,
        description="Use curvature filter to reduce oscillations",
    )
    use_fast_optimization_target: bool = Field(
        True,
        description="Use fast CMC target (requires accelerations to be met)",
    )
    lowpass_cutoff_frequency: float = Field(
        -1.0,
        description="Low-pass cutoff (Hz) for desired kinematics; -1 = no filter",
    )

    # Integrator settings
    maximum_number_of_integrator_steps: int = Field(
        20000, description="Maximum number of integrator steps"
    )
    maximum_integrator_step_size: float = Field(
        1.0, description="Maximum integration step size (s)"
    )
    minimum_integrator_step_size: float = Field(
        1e-8, description="Minimum integration step size (s)"
    )
    integrator_error_tolerance: float = Field(
        1e-5, description="Integrator error tolerance"
    )

    # Optimizer settings
    optimizer_algorithm: Literal["ipopt", "cfsqp"] = Field(
        "ipopt", description="Optimizer algorithm"
    )
    optimization_convergence_tolerance: float = Field(
        1e-4, description="Optimizer convergence tolerance"
    )
    optimizer_max_iterations: int = Field(
        1000, description="Maximum optimizer iterations"
    )
    optimizer_print_level: int = Field(0, description="Optimizer print level (0-3)")

    # Misc
    use_verbose_printing: bool = Field(False, description="Verbose CMC printing")
    replace_force_set: bool = Field(
        False,
        description="Replace (True) or append (False) model's force set with force_set_files",
    )
    actuators_to_exclude: list[str] = Field(
        default_factory=list,
        description="Actuators/groups to exclude from CMC control",
    )
    output_precision: int = Field(8, description="Output precision")

    def get_result_type(self) -> type[CMCResult]:
        return CMCResult

    def get_result_kwargs(self) -> dict[str, Path]:
        results_dir = Path(self.results_directory)
        controls_files = sorted(glob.glob(str(results_dir / "*_controls.sto")))
        forces_files = sorted(glob.glob(str(results_dir / "*Actuation_force*.sto")))
        if not forces_files:
            forces_files = sorted(glob.glob(str(results_dir / "*force*.sto")))
        states_files = sorted(glob.glob(str(results_dir / "*_states.sto")))

        if not controls_files:
            controls_files = sorted(glob.glob(str(results_dir / "*_controls.xml")))

        return {
            "controls_file": Path(controls_files[0]),
            "forces_file": Path(forces_files[0]),
            "states_file": Path(states_files[0]),
        }

    def create_tool(self) -> osim.CMCTool:
        """Build a fully configured ``osim.CMCTool``."""

        if self.setup_path is not None:
            tool = osim.CMCTool(self.setup_path.resolve())
        else:
            tool = osim.CMCTool()

        rel_model_path = self.get_relative_path(self.model_path)
        rel_results_dir = self.get_relative_path(self.results_directory)

        tool.setModelFilename(rel_model_path)

        tool.setResultsDir(rel_results_dir)

        initial_time = self.initial_time
        final_time = self.final_time
        if initial_time is None or final_time is None:
            try:
                if self.desired_kinematics_path is not None:
                    sto = osim.Storage(self.desired_kinematics_path.resolve())
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
                    # TODO: This could be validated on construction instead of at runtime
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

        tool.setReplaceForceSet(self.replace_force_set)
        if self.force_set_paths:
            arr = osim.ArrayStr()
            for fp in self.force_set_paths:
                arr.append(self.get_relative_path(fp))
            tool.setForceSetFiles(arr)

        if self.external_loads_path:
            rel_external_loads_path = self.get_relative_path(self.external_loads_path)
            tool.setExternalLoadsFileName(rel_external_loads_path)

        if self.desired_points_path:
            rel_desired_points_path = self.get_relative_path(self.desired_points_path)
            tool.setDesiredPointsFileName(rel_desired_points_path)
        if self.desired_kinematics_path:
            rel_desired_kinematics_path = self.get_relative_path(
                self.desired_kinematics_path
            )

            tool.setDesiredKinematicsFileName(rel_desired_kinematics_path)
        if self.task_set_path:
            rel_task_set_path = self.get_relative_path(self.task_set_path)
            tool.setTaskSetFileName(rel_task_set_path)
        if self.constraints_path:
            rel_constraints_path = self.get_relative_path(self.constraints_path)
            tool.setConstraintsFileName(rel_constraints_path)
        if self.rra_controls_path:
            rel_rra_controls_path = self.get_relative_path(self.rra_controls_path)
            tool.setRRAControlsFileName(rel_rra_controls_path)

        # CMC parameters
        tool.setSolveForEquilibrium(self.solve_for_equilibrium_for_auxiliary_states)
        tool.setTimeWindow(self.cmc_time_window)
        tool.setUseFastTarget(self.use_fast_optimization_target)
        tool.setLowpassCutoffFrequency(self.lowpass_cutoff_frequency)
        tool.setUseVerbosePrinting(self.use_verbose_printing)

        # Integrator
        tool.setMaximumNumberOfSteps(self.maximum_number_of_integrator_steps)
        tool.setMaxDT(self.maximum_integrator_step_size)
        tool.setMinDT(self.minimum_integrator_step_size)
        tool.setErrorTolerance(self.integrator_error_tolerance)

        # Optimizer (no dedicated setters — use PropertyHelper)
        p = tool.updPropertyByName("optimizer_algorithm")
        osim.PropertyHelper.setValueString(self.optimizer_algorithm, p)

        p = tool.updPropertyByName("optimization_convergence_tolerance")
        osim.PropertyHelper.setValueDouble(self.optimization_convergence_tolerance, p)

        p = tool.updPropertyByName("optimizer_max_iterations")
        osim.PropertyHelper.setValueInt(self.optimizer_max_iterations, p)

        p = tool.updPropertyByName("optimizer_print_level")
        osim.PropertyHelper.setValueInt(self.optimizer_print_level, p)

        p = tool.updPropertyByName("use_curvature_filter")
        osim.PropertyHelper.setValueBool(self.use_curvature_filter, p)

        # Excluded actuators
        if self.actuators_to_exclude:
            arr = osim.ArrayStr()
            for name in self.actuators_to_exclude:
                arr.append(name)
            tool.setExcludedActuators(arr)

        return tool
