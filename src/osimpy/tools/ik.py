import opensim as osim
from pydantic import Field, FilePath
from .tool import ToolSettings, ToolResult
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IKResult(ToolResult):
    """Result from Inverse Kinematics analysis.

    Attributes
    ----------
    motion_file : FilePath
        Path to the output motion (.mot) file containing computed coordinates
    """

    motion_file: str = Field(description="Name of output motion file (.mot)")


class IKSettings(ToolSettings[IKResult]):
    """Inverse Kinematics tool settings.

    Configure and run inverse kinematics analysis to compute joint angles
    from marker trajectories.

    If ``setup_file`` is provided, the InverseKinematicsTool is initialised
    from that XML template (preserving IKTaskSet marker weights, constraint
    weight, accuracy, etc.) and then individual fields are applied on top.
    Otherwise a blank InverseKinematicsTool is created.
    """

    marker_path: FilePath = Field(description="Path to marker data file (.trc)")
    output_motion_file: str = Field(description="Name of output motion file (.mot)")

    task_set_path: FilePath | None = Field(
        None, description="IK task set for tracking (overrides template)"
    )
    constraint_weight: float | None = Field(
        None,
        description="Weight for kinematic constraints (None = use template default)",
    )
    accuracy: float | None = Field(
        None, description="Convergence accuracy (None = use template default)"
    )
    report_marker_locations: bool | None = Field(
        None,
        description="Report marker locations in output (None = use template default)",
    )

    def get_result_type(self) -> type[IKResult]:
        return IKResult

    def get_result_kwargs(self) -> dict[str, str]:
        return {"motion_file": self.output_motion_file}

    def create_tool(self) -> osim.InverseKinematicsTool:
        """Create and configure an InverseKinematicsTool instance.

        If ``setup_file`` is set, the tool is loaded from the template XML
        first (preserving IKTaskSet, constraint weight, accuracy, etc.).
        Individual settings are then applied on top.

        Returns
        -------
        InverseKinematicsTool
            Configured InverseKinematicsTool instance
        """
        # --- Load from template or create blank ---
        if self.setup_path is not None:
            tool = osim.InverseKinematicsTool(str(self.setup_path.resolve()))
        else:
            tool = osim.InverseKinematicsTool()

        rel_model_path = str(self.get_relative_path(self.model_path))
        rel_marker_path = str(self.get_relative_path(self.marker_path))
        rel_results_dir = str(self.get_relative_path(self.results_directory))

        tool.set_model_file(rel_model_path)
        tool.setResultsDir(rel_results_dir)
        tool.setMarkerDataFileName(rel_marker_path)
        tool.setOutputMotionFileName(self.output_motion_file)

        # Override task set only if explicitly provided
        if self.task_set_path is not None:
            rel_task_set_path = str(self.get_relative_path(self.task_set_path))
            tool.set_IKTaskSet(rel_task_set_path)

        # Override constraint weight only if explicitly provided
        if self.constraint_weight is not None:
            tool.set_constraint_weight(self.constraint_weight)

        # Override accuracy only if explicitly provided
        if self.accuracy is not None:
            tool.set_accuracy(self.accuracy)

        # Override report_marker_locations only if explicitly provided
        if self.report_marker_locations is not None:
            tool.set_report_marker_locations(self.report_marker_locations)

        return tool
