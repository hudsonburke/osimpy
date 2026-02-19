import opensim as osim
from datetime import datetime
from pathlib import Path
from pydantic import Field, FilePath, NewPath
from .tool import ToolSettings, ToolResult

class IDResult(ToolResult):
    """Result from Inverse Dynamics analysis."""

    forces_file: FilePath = Field(description="Path to output forces file (.sto)")

class IDSettings(ToolSettings):
    """Inverse Dynamics tool settings.

    Configure and run inverse dynamics analysis to compute generalized forces
    from joint kinematics and external forces.
    """

    coordinates_file: FilePath = Field(description="Path to coordinates file (.mot) from IK")
    output_forces_file: NewPath = Field(description="Path for output forces file (.sto)")

    # Time range
    initial_time: float = Field(
        -1.0, description="Initial time for analysis (-1 = auto from file)"
    )
    final_time: float = Field(
        -1.0, description="Final time for analysis (-1 = auto from file)"
    )

    # Optional settings
    external_loads_file: FilePath | None = Field(
        None, description="Path to external loads XML file"
    )
    lowpass_cutoff_frequency: float = Field(
        -1.0,
        description="Cutoff frequency for filtering coordinates (-1 = no filtering)",
    )
    excluded_forces: list[str] = Field(
        default_factory=list, description="List of force names to exclude from analysis"
    )

    def create_tool(self) -> osim.InverseDynamicsTool:
        """Create and configure an InverseDynamicsTool instance.

        Returns
        -------
        InverseDynamicsTool
            Configured ID tool ready for execution
        """
        tool = osim.InverseDynamicsTool()

        tool.setModelFileName(str(self.model_file))

        # Set results directory
        tool.setResultsDir(str(self.results_directory))

        # Set external loads if provided
        if self.external_loads_file:
            tool.setExternalLoadsFileName(str(self.external_loads_file))

        # Set coordinates and output files
        tool.setCoordinatesFileName(str(self.coordinates_file))
        tool.setOutputGenForceFileName(str(self.output_forces_file))

        # Set filtering
        if self.lowpass_cutoff_frequency > 0:
            tool.setLowpassCutoffFrequency(self.lowpass_cutoff_frequency)

        # Set excluded forces
        if self.excluded_forces:
            exclude = osim.ArrayStr()
            for force in self.excluded_forces:
                exclude.append(force)
            tool.setExcludedForces(exclude)

        # Set time range (auto-detect from coordinates file if needed)
        if self.initial_time == -1.0 or self.final_time == -1.0:
            sto = osim.Storage(str(self.coordinates_file))
            if self.initial_time == -1.0:
                tool.setStartTime(sto.getFirstTime())
            else:
                tool.setStartTime(self.initial_time)
            if self.final_time == -1.0:
                tool.setEndTime(sto.getLastTime())
            else:
                tool.setEndTime(self.final_time)
        else:
            tool.setStartTime(self.initial_time)
            tool.setEndTime(self.final_time)

        return tool

    def run(self) -> IDResult:
        """Execute ID analysis using XML-based workflow.

        Save settings to XML, load tool from XML (which loads the model), then run.

        Returns
        -------
        IDResult
            Structured ID results with forces file path and metadata
        """
        # Ensure results directory exists
        results_dir = Path(self.results_directory)
        results_dir.mkdir(parents=True, exist_ok=True)

        warnings = []
        errors = []
        success = False
        start_time = datetime.now()

        try:
            # Create and configure tool (for XML generation)
            tool = self.create_tool()

            # Save setup XML with model_file injected
            setup_file = self.save_setup()

            # Recreate tool from XML file (this loads the model)
            tool = osim.InverseDynamicsTool(setup_file, True)  # True = load model

            # Execute tool
            tool.run()
            success = True

        except Exception as e:
            errors.append(str(e))
            setup_file = str(results_dir / "failed_setup.xml")
            raise RuntimeError(f"Tool execution failed: {e}") from e

        finally:
            end_time = datetime.now()

        return IDResult(
            success=success,
            setup_file=Path(setup_file),
            results_directory=self.results_directory,
            start_time=start_time,
            end_time=end_time,
            run_time=(end_time - start_time).total_seconds(),
            warnings=warnings,
            errors=errors,
            forces_file=self.output_forces_file,
        )