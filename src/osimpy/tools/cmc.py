import opensim as osim
from pydantic import Field, FilePath, DirectoryPath, NewPath
from typing import Literal
from pathlib import Path
from .tool import ToolSettings, ToolResult


class CMCResult(ToolResult):
    """Result from Computed Muscle Control analysis."""

    controls_file: FilePath = Field(description="Path to output controls file")
    forces_file: FilePath = Field(description="Path to output forces file")
    states_file: FilePath = Field(description="Path to output states file")

class CMCSettings(ToolSettings):
    """CMC (Computed Muscle Control) settings.

    Descriptions are available in Field(...) metadata for runtime/schema usage.

    References
    ----------
    1. OpenSim CMC User Guide:
    https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53089721/CMC+Settings+Files+and+XML+Tag+Definitions
    """

    initial_time: float = Field(description="Initial time for the simulation")
    final_time: float = Field(description="Final time for the simulation")

    external_loads_file: FilePath | None = Field(
        None, description="Path to external loads XML file"
    )
    force_set_files: list[FilePath] = Field(
        default_factory=list,
        description="Paths to force set files (.xml) to add actuators",
    )

    solve_for_equilibrium_for_auxiliary_states: bool = Field(
        True,
        description=(
            "Flag indicating whether or not to compute equilibrium values for "
            "states other than the coordinates or speeds. For example, equilibrium "
            "muscle fiber lengths or muscle forces."
        ),
    )
    maximum_number_of_integrator_steps: int = Field(
        10000, description="Maximum number of integrator steps."
    )
    maximum_integrator_step_size: float = Field(
        1.0, description="Maximum integration step size."
    )
    minimum_integrator_step_size: float = Field(
        0.0, description="Minimum integration step size."
    )
    integrator_error_tolerance: float = Field(
        1e-6,
        description=(
            "Integrator error tolerance. When the error is greater, the integrator "
            "step size is decreased."
        ),
    )
    desired_points_file: FilePath | None = Field(
        None,
        description=(
            "Motion (.mot) or storage (.sto) file containing the desired point "
            "trajectories."
        ),
    )
    desired_kinematics_file: FilePath | None = Field(
        None,
        description=(
            "Motion (.mot) or storage (.sto) file containing the desired kinematic "
            "trajectories."
        ),
    )
    task_set_file: FilePath | None = Field(
        None,
        description=(
            "File containing the tracking tasks. Which coordinates are tracked and "
            "with what weights are specified here."
        ),
    )
    constraints_file: FilePath | None = Field(
        None,
        description="File containing the constraints on the controls.",
    )
    rra_controls_file: FilePath | None = Field(
        None,
        description=(
            "File containing the controls output by RRA. These can be used to place "
            "constraints on the residuals during CMC."
        ),
    )
    lowpass_cutoff_frequency: float = Field(
        -1.0,
        description=(
            "Low-pass cut-off frequency for filtering the desired kinematics. A "
            "negative value results in no filtering. The default value is -1.0, so "
            "no filtering."
        ),
    )
    cmc_time_window: float = Field(
        0.01,
        description=(
            "Time window over which the desired actuator forces are achieved. "
            "Muscles forces cannot change instantaneously, so a finite time window "
            "must be allowed. The recommended time window for RRA is about 0.001 "
            "sec, and for CMC is about 0.010 sec."
        ),
    )
    use_curvature_filter: bool = Field(
        False,
        description=(
            "Flag (true or false) indicating whether or not to use the curvature "
            "filter. Setting this flag to true can reduce oscillations in the "
            "computed muscle excitations."
        ),
    )
    use_fast_optimization_target: bool = Field(
        True,
        description=(
            "Flag (true or false) indicating whether to use the fast CMC "
            "optimization target. The fast target requires the desired "
            "accelerations to be met. The optimizer fails if the accelerations "
            "constraints cannot be met, so the fast target can be less robust. The "
            "regular target does not require the acceleration constraints to be "
            "met; it meets them as well as it can, but it is slower and less "
            "accurate."
        ),
    )
    optimizer_algorithm: Literal["ipopt", "cfsqp"] = Field(
        "ipopt",
        description=(
            'Preferred optimizer algorithm (currently support "ipopt" or "cfsqp",'
            " the latter requiring the osimFSQP library.)"
        ),
    )
    optimizer_derivative_dx: float = Field(
        1e-6,
        description=(
            "Perturbation size used by the optimizer to compute numerical "
            "derivatives. A value between 1.0e-4 and 1.0e-8 is usually appropriate."
        ),
    )
    optimizer_convergence_criterion: float = Field(
        1e-4,
        description=(
            "Convergence criterion for the optimizer. The smaller this value, the "
            "deeper the convergence. Decreasing this number can improve a solution, "
            "but will also likely increase computation time."
        ),
    )
    optimizer_max_iterations: int = Field(
        500, description="Maximum number of iterations for the optimizer."
    )
    optimizer_print_level: int = Field(
        0,
        description=(
            "Print level for the optimizer, 0 - 3. 0=no printing, 3=detailed "
            "printing, 2=in between"
        ),
    )
    use_verbose_printing: bool = Field(
        False,
        description=(
            "True-false flag indicating whether or not to turn on verbose printing "
            "for cmc."
        ),
    )
    actuators_to_exclude: list[str] = Field(
        default_factory=list,
        description=(
            "List of individual Actuators by individual or user-defined group name to be "
            "excluded from CMC's control."
        ),
    )

    def create_tool(self) -> osim.CMCTool:
        """Create and configure a CMCTool instance.

        Returns
        -------
        CMCTool
            Configured CMC tool ready for execution
        """
        tool = osim.CMCTool()

        # Set results directory and time range
        tool.setResultsDir(self.results_directory)
        tool.setInitialTime(self.initial_time)
        tool.setFinalTime(self.final_time)

        # Set external loads if provided
        if self.external_loads_file:
            tool.setExternalLoadsFileName(self.external_loads_file)

        # Set force set files if provided
        if self.force_set_files:
            force_set_array = osim.ArrayStr()
            for file_path in self.force_set_files:
                force_set_array.append(file_path)
            tool.setForceSetFiles(force_set_array)

        # Set CMC-specific settings
        tool.setDesiredPointsFileName(self.desired_points_file)
        tool.setDesiredKinematicsFileName(self.desired_kinematics_file)
        tool.setTaskSetFileName(self.task_set_file)
        tool.setConstraintsFileName(self.constraints_file)
        tool.setRRAControlsFileName(self.rra_controls_file)
        tool.setLowpassCutoffFrequency(self.lowpass_cutoff_frequency)
        tool.setUseFastTarget(self.use_fast_optimization_target)

        return tool

    def run(self) -> None:
        import os

        model_dir = Path(self.model_file).parent
        os.chdir(model_dir)
