import json
import hashlib
from loguru import logger
import opensim as osim
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal
import glob
from .results import CMCResult


class CMCSettings(BaseModel):
    """CMC (Computed Muscle Control) settings.

    Descriptions are available in Field(...) metadata for runtime/schema usage.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core parameters
    model_file: str = Field(description="Path to OpenSim model file (.osim)")
    results_directory: str = Field(
        ".", description="Directory for results and setup files"
    )

    # Time range
    initial_time: float = Field(description="Initial time for analysis")
    final_time: float = Field(description="Final time for analysis")

    # Optional settings
    external_loads_file: str | None = Field(
        None, description="Path to external loads XML file"
    )
    force_set_files: list[str] = Field(
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
    desired_points_file: str = Field(
        "",
        description=(
            "Motion (.mot) or storage (.sto) file containing the desired point "
            "trajectories."
        ),
    )
    desired_kinematics_file: str = Field(
        "",
        description=(
            "Motion (.mot) or storage (.sto) file containing the desired kinematic "
            "trajectories."
        ),
    )
    task_set_file: str = Field(
        "",
        description=(
            "File containing the tracking tasks. Which coordinates are tracked and "
            "with what weights are specified here."
        ),
    )
    constraints_file: str = Field(
        "",
        description="File containing the constraints on the controls.",
    )
    rra_controls_file: str = Field(
        "",
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

    def save_setup(self, filepath: str | None = None) -> str:
        """Save CMC tool setup to XML file with model file path.

        Parameters
        ----------
        filepath : str | None
            Path to save the setup file. If None, uses results_directory
            with a default name.

        Returns
        -------
        str
            Path to the saved setup file
        """
        tool = self.create_tool()

        if filepath is None:
            # Create default setup filename in results directory
            from pathlib import Path

            results_dir = Path(self.results_directory)
            results_dir.mkdir(parents=True, exist_ok=True)
            tool_name = self.__class__.__name__.replace("Settings", "").lower()
            filepath = str(results_dir / f"{tool_name}_setup.xml")

        # Write to XML
        tool.printToXML(filepath)

        # Read the XML file and replace the empty model_file element
        with open(filepath, "r") as f:
            content = f.read()

        # Replace the empty model_file tag with the actual path
        # The printToXML() writes <model_file /> which we need to replace
        content = content.replace(
            "<model_file />", "<model_file>{}</model_file>".format(self.model_file)
        )

        with open(filepath, "w") as f:
            f.write(content)

        return filepath

    def run(self) -> CMCResult:
        """Execute CMC analysis and return results.

        CMC requires explicitly loading and setting the model.

        Returns
        -------
        CMCResult
            Structured CMC results with controls and kinematics file paths
        """
        # Ensure results directory exists
        results_dir = Path(self.results_directory)
        results_dir.mkdir(parents=True, exist_ok=True)

        warnings = []
        errors = []
        success = False
        start_time = datetime.now()

        try:
            # First create and save the setup XML
            setup_file = self.save_setup()

            # Load the model explicitly
            model = osim.Model(self.model_file)

            # Initialize the system
            state = model.initSystem()

            # Load CMC tool from the XML file
            tool = osim.CMCTool(
                setup_file, False
            )  # False = don't update from old versions

            # Set the model on the tool
            tool.setModel(model)

            # Execute tool
            result = tool.run()
            success = result if isinstance(result, bool) else True

        except Exception as e:
            errors.append(str(e))
            setup_file = str(results_dir / "failed_setup.xml")
            raise RuntimeError(f"Tool execution failed: {e}") from e

        finally:
            end_time = datetime.now()

        # CMC creates output files with specific naming patterns
        # Try to find the actual output files
        controls_pattern = str(results_dir / "*_controls.xml")
        kinematics_pattern = str(results_dir / "*_Kinematics*.sto")

        controls_files = glob.glob(controls_pattern)
        kinematics_files = glob.glob(kinematics_pattern)

        # Use the first match if found, otherwise use default names
        output_controls = (
            controls_files[0]
            if controls_files
            else str(results_dir / "cmc_controls.sto")
        )
        output_kinematics = (
            kinematics_files[0]
            if kinematics_files
            else str(results_dir / "cmc_kinematics.sto")
        )

        return CMCResult(
            success=success,
            setup_file=setup_file,
            results_directory=self.results_directory,
            start_time=start_time,
            end_time=end_time,
            run_time=(end_time - start_time).total_seconds(),
            warnings=warnings,
            errors=errors,
            output_controls_file=output_controls,
            output_kinematics_file=output_kinematics,
            desired_kinematics_file=self.desired_kinematics_file,
        )

    def to_dict(self) -> dict:
        """Export settings as dictionary for reproducibility.

        Returns:
            Dictionary containing all settings with JSON-compatible types.

        Example:
            >>> settings = CMCSettings(...)
            >>> data = settings.to_dict()
            >>> # Can be saved as JSON for thesis documentation
        """
        return self.model_dump(mode="json")

    def save_json(self, filepath: str) -> None:
        """Save settings as JSON file for easy tracking.

        Args:
            filepath: Path to save JSON file

        Example:
            >>> settings = CMCSettings(...)
            >>> settings.save_json("cmc_settings.json")
            >>> # Creates reproducible parameter record
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved settings to {filepath}")

    @classmethod
    def from_json(cls, filepath: str):
        """Load settings from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Settings instance with loaded parameters

        Example:
            >>> settings = CMCSettings.from_json("cmc_settings.json")
            >>> result = settings.run()  # Exact same parameters
        """
        with open(filepath) as f:
            data = json.load(f)
        logger.info(f"Loaded settings from {filepath}")
        return cls(**data)

    def get_hash(self) -> str:
        """Get deterministic hash of settings for caching/comparison.

        Returns:
            SHA256 hash of settings

        Example:
            >>> hash1 = settings.get_hash()
            >>> # Later...
            >>> hash2 = settings.get_hash()
            >>> assert hash1 == hash2  # Same settings = same hash
        """
        settings_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(settings_str.encode()).hexdigest()
