import opensim as osim
from datetime import datetime
import glob
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
        tool.setResultsDir(str(self.results_directory))
        tool.setInitialTime(self.initial_time)
        tool.setFinalTime(self.final_time)

        # Set external loads if provided
        if self.external_loads_file:
            tool.setExternalLoadsFileName(str(self.external_loads_file))

        # Set force set files if provided
        if self.force_set_files:
            force_set_array = osim.ArrayStr()
            for file_path in self.force_set_files:
                force_set_array.append(str(file_path))
            tool.setForceSetFiles(force_set_array)

        # Set CMC-specific settings
        tool.setDesiredPointsFileName(
            str(self.desired_points_file) if self.desired_points_file else ""
        )
        tool.setDesiredKinematicsFileName(
            str(self.desired_kinematics_file) if self.desired_kinematics_file else ""
        )
        tool.setTaskSetFileName(str(self.task_set_file) if self.task_set_file else "")
        tool.setConstraintsFileName(
            str(self.constraints_file) if self.constraints_file else ""
        )
        tool.setRRAControlsFileName(
            str(self.rra_controls_file) if self.rra_controls_file else ""
        )
        tool.setLowpassCutoffFrequency(self.lowpass_cutoff_frequency)
        tool.setUseFastTarget(self.use_fast_optimization_target)

        return tool

    def save_setup(self, filepath: str | None = None) -> str:
        """Save CMC tool setup to XML file with explicit model file.

        Args:
            filepath: Optional path to save the setup XML.

        Returns:
            Path to the saved setup XML.
        """
        tool = self.create_tool()

        if filepath is None:
            results_dir = Path(self.results_directory)
            results_dir.mkdir(parents=True, exist_ok=True)
            filepath = str(results_dir / "cmc_setup.xml")

        tool.printToXML(filepath)

        with open(filepath, "r", encoding="utf-8") as setup_file:
            content = setup_file.read()

        content = content.replace(
            "<model_file />", f"<model_file>{self.model_file}</model_file>"
        )

        with open(filepath, "w", encoding="utf-8") as setup_file:
            setup_file.write(content)

        return filepath

    def run(self) -> CMCResult:
        """Execute CMC analysis using XML-based workflow.

        Returns:
            Structured CMC result with discovered output file paths.
        """
        results_dir = Path(self.results_directory)
        results_dir.mkdir(parents=True, exist_ok=True)

        warnings = []
        errors = []
        success = False
        start_time = datetime.now()

        try:
            setup_file = self.save_setup()

            model = osim.Model(str(self.model_file))
            model.initSystem()

            tool = osim.CMCTool(setup_file, False)
            tool.setModel(model)
            run_result = tool.run()
            success = run_result if isinstance(run_result, bool) else True

        except Exception as exception:
            errors.append(str(exception))
            setup_file = str(results_dir / "failed_cmc_setup.xml")
            raise RuntimeError(f"Tool execution failed: {exception}") from exception

        finally:
            end_time = datetime.now()

        controls_candidates = sorted(glob.glob(str(results_dir / "*controls*.sto")))
        forces_candidates = sorted(glob.glob(str(results_dir / "*Actuation_force*.sto")))
        if not forces_candidates:
            forces_candidates = sorted(glob.glob(str(results_dir / "*force*.sto")))
        states_candidates = sorted(glob.glob(str(results_dir / "*states*.sto")))

        if not controls_candidates:
            controls_candidates = sorted(glob.glob(str(results_dir / "*_controls.xml")))
        if not controls_candidates:
            raise RuntimeError(
                f"CMC finished but no controls file found in {results_dir}."
            )
        if not forces_candidates:
            raise RuntimeError(f"CMC finished but no forces file found in {results_dir}.")
        if not states_candidates:
            raise RuntimeError(f"CMC finished but no states file found in {results_dir}.")

        return CMCResult(
            success=success,
            setup_file=Path(setup_file),
            results_directory=self.results_directory,
            start_time=start_time,
            end_time=end_time,
            run_time=(end_time - start_time).total_seconds(),
            warnings=warnings,
            errors=errors,
            controls_file=Path(controls_candidates[0]),
            forces_file=Path(forces_candidates[0]),
            states_file=Path(states_candidates[0]),
        )
