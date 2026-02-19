import opensim as osim
from datetime import datetime
from pathlib import Path
from pydantic import Field, ConfigDict, FilePath, NewPath, BaseModel
from .tool import ToolSettings, ToolResult

class IKTask(BaseModel):
    pass


class IKResult(ToolResult):
    """Result from Inverse Kinematics analysis.

    Attributes
    ----------
    output_motion_file : str
        Path to the output motion (.mot) file containing computed coordinates

    """

    motion_file: FilePath = Field(description="Path to output motion file (.mot)")


class IKSettings(ToolSettings):
    """Inverse Kinematics tool settings.

    Configure and run inverse kinematics analysis to compute joint angles
    from marker trajectories.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    marker_file: FilePath = Field(description="Path to marker data file (.trc)")
    output_motion_file: NewPath = Field(description="Path for output motion file (.mot)")

    task_set: osim.IKTaskSet | None = Field(
        None, description="IK task set for tracking"
    )
    constraint_weight: float = Field(
        1.0, description="Weight for kinematic constraints"
    )
    accuracy: float = Field(1e-5, description="Convergence accuracy")
    report_marker_locations: bool = Field(
        False, description="Report marker locations in output"
    )

    def create_tool(self) -> osim.InverseKinematicsTool:
        """Create and configure an InverseKinematicsTool instance.

        Returns
        -------
        InverseKinematicsTool
            Configured InverseKinematicsTool instance
        """
        tool = osim.InverseKinematicsTool()

        tool.set_model_file(str(self.model_file))
        tool.setResultsDir(str(self.results_directory))
        tool.setMarkerDataFileName(str(self.marker_file))
        tool.setOutputMotionFileName(str(self.output_motion_file))

        if self.task_set is not None:
            tool.set_IKTaskSet(self.task_set)

        return tool
    
    def run(self) -> IKResult:
        """Execute IK analysis using XML-based workflow.

        Save settings to XML, load tool from XML (which loads the model), then run.

        Returns
        -------
        IKResult
            Structured IK results with motion file path and metadata
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
            tool = osim.InverseKinematicsTool(setup_file, True)  # True = load model

            # Execute tool
            tool.run()
            success = True

        except Exception as e:
            errors.append(str(e))
            setup_file = str(results_dir / "failed_setup.xml")
            raise RuntimeError(f"Tool execution failed: {e}") from e

        finally:
            end_time = datetime.now()

        return IKResult(
            success=success,
            setup_file=Path(setup_file),
            results_directory=self.results_directory,
            start_time=start_time,
            end_time=end_time,
            run_time=(end_time - start_time).total_seconds(),
            warnings=warnings,
            errors=errors,
            motion_file=self.output_motion_file,
        )
