import opensim as osim
from pathlib import Path
from pydantic import Field, FilePath
from .tool import ToolSettings, ToolResult
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IDResult(ToolResult):
    """Result from Inverse Dynamics analysis."""

    forces_file: FilePath = Field(description="Path to output forces file (.sto)")


class IDSettings(ToolSettings[IDResult]):
    """Inverse Dynamics tool settings.

    Configure and run inverse dynamics analysis to compute generalized forces
    from joint kinematics and external forces.

    If ``setup_file`` is provided, the InverseDynamicsTool is initialised
    from that XML template first and then individual fields are applied on
    top.  Otherwise a blank InverseDynamicsTool is created.
    """

    coordinates_path: FilePath = Field(
        description="Path to coordinates file (.mot) from IK"
    )
    output_forces_file: str = Field(description="Name of output forces file (.sto)")

    # Time range
    initial_time: float = Field(
        -1.0, description="Initial time for analysis (-1 = auto from file)"
    )
    final_time: float = Field(
        -1.0, description="Final time for analysis (-1 = auto from file)"
    )

    # Optional settings
    external_loads_path: FilePath | None = Field(
        None, description="Path to external loads XML file"
    )
    lowpass_cutoff_frequency: float = Field(
        -1.0,
        description="Cutoff frequency for filtering coordinates (-1 = no filtering)",
    )
    excluded_forces: list[str] = Field(
        default_factory=list, description="List of force names to exclude from analysis"
    )

    def get_result_type(self) -> type[IDResult]:
        return IDResult

    def get_result_kwargs(self) -> dict[str, Path]:
        return {"forces_file": Path(self.output_forces_file)}

    def create_tool(self) -> osim.InverseDynamicsTool:
        """Create and configure an InverseDynamicsTool instance.

        If ``setup_file`` is set, the tool is loaded from the template XML
        first.  Individual settings are then applied on top.

        Returns
        -------
        InverseDynamicsTool
            Configured ID tool ready for execution
        """
        # --- Load from template or create blank ---
        if self.setup_path is not None:
            tool = osim.InverseDynamicsTool(str(self.setup_path.resolve()))
        else:
            tool = osim.InverseDynamicsTool()

        rel_model_path = str(self.get_relative_path(self.model_path))

        tool.setModelFileName(rel_model_path)

        # Set results directory
        tool.setResultsDir(str(self.results_directory))

        # Set external loads if provided
        if self.external_loads_path:
            rel_external_loads_path = str(
                self.get_relative_path(self.external_loads_path)
            )
            tool.setExternalLoadsFileName(rel_external_loads_path)

        rel_coordinates_path = str(self.get_relative_path(self.coordinates_path))
        tool.setCoordinatesFileName(rel_coordinates_path)
        tool.setOutputGenForceFileName(self.output_forces_file)

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
            sto = osim.Storage(str(self.coordinates_path.resolve()))
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
