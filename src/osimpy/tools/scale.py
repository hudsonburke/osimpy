import opensim as osim
from pydantic import Field, FilePath
from .tool import ToolSettings, ToolResult
import logging

logger = logging.getLogger(__name__)


class ScaleResult(ToolResult):
    """Result from Scale Tool analysis.

    Attributes
    ----------
    scaled_model_file : str
        Path to the scaled model file
    """

    scaled_model_file: FilePath = Field(description="Path to scaled model file")


class ScaleSettings(ToolSettings[ScaleResult]):
    """Scale Tool settings.

    Configure and run the scale tool to scale a generic model to a subject's
    anthropometry based on marker data.

    If ``setup_file`` is provided, the ScaleTool is initialised from that XML
    template (preserving measurement sets, scale ordering, MarkerPlacer
    IKTaskSet, etc.) and then individual fields are applied on top.  Otherwise
    a blank ScaleTool is created.
    """

    marker_set_path: FilePath = Field(description="Path to the marker set file")
    marker_path: FilePath = Field(
        description="Path to marker data file for scaling (.trc)"
    )
    output_model_file: str = Field(description="Name for the output scaled model file")
    output_scale_file: str | None = Field(
        None, description="Name for the output scale factors XML"
    )

    scale_factors: dict[str, tuple[float, float, float]] = Field(
        default_factory=dict,
        description="Scale factors for body segments {segment_name: (x, y, z)}",
    )
    preserve_mass_distribution: bool = Field(
        True, description="Preserve mass distribution when scaling"
    )
    subject_mass: float | None = Field(
        None,
        description="Subject's total mass (kg). If None, uses generic model mass",
    )
    use_marker_placer: bool = Field(
        True, description="Whether to run the MarkerPlacer tool after scaling"
    )

    def get_result_type(self) -> type[ScaleResult]:
        return ScaleResult

    def get_result_kwargs(self) -> dict[str, str]:
        return {"scaled_model_file": self.output_model_file}

    def create_tool(self) -> osim.ScaleTool:
        """Create and configure a ScaleTool instance.

        If ``setup_file`` is set, the tool is loaded from the template XML
        first (preserving measurement sets, scale ordering, MarkerPlacer
        IKTaskSet, etc.).  Individual settings are then applied on top.

        Returns
        -------
        osim.ScaleTool
            Configured ScaleTool instance
        """
        # Load from template or create blank
        # TODO: Make sure you can load from template without overriding
        if self.setup_path is not None:
            tool = osim.ScaleTool(str(self.setup_path.resolve()))
        else:
            tool = osim.ScaleTool()

        generic_model_maker: osim.GenericModelMaker = tool.getGenericModelMaker()

        rel_model_path = self.get_relative_path(self.model_path)
        rel_marker_path = self.get_relative_path(self.marker_path)

        generic_model_maker.setModelFileName(rel_model_path)
        if self.marker_set_path:
            rel_marker_set_path = self.get_relative_path(self.marker_set_path)
            generic_model_maker.setMarkerSetFileName(rel_marker_set_path)

        if self.subject_mass is not None:
            tool.setSubjectMass(self.subject_mass)

        model_scaler: osim.ModelScaler = tool.getModelScaler()
        model_scaler.setApply(True)
        model_scaler.setMarkerFileName(rel_marker_path)
        model_scaler.setPreserveMassDist(self.preserve_mass_distribution)
        model_scaler.setOutputModelFileName(self.output_model_file)

        if self.output_scale_file is not None:
            model_scaler.setOutputScaleFileName(self.output_scale_file)

        time_array = osim.ArrayDouble()
        initial_time = self.initial_time
        final_time = self.final_time
        if initial_time is None or final_time is None:
            try:
                trc = osim.MarkerData(str(self.marker_path.resolve()))
                if initial_time is None:
                    initial_time = trc.getStartFrameTime()
                if final_time is None:
                    final_time = trc.getLastFrameTime()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load marker data from '{self.marker_path}': {e}"
                ) from e

        time_array.set(0, initial_time)
        time_array.set(1, final_time)
        model_scaler.setTimeRange(time_array)

        if self.scale_factors:
            scale_set: osim.ScaleSet = model_scaler.getScaleSet()
            for segment_name, factors in self.scale_factors.items():
                vec = osim.Vec3(*factors)
                found = False
                for j in range(scale_set.getSize()):
                    scale_obj = scale_set.get(j)
                    if scale_obj.getSegmentName() == segment_name:
                        scale_obj.setScaleFactors(vec)
                        found = True
                        logger.info(
                            f"Set manual scale factors for '{segment_name}': "
                            f"({factors[0]:.6f}, {factors[1]:.6f}, {factors[2]:.6f})"
                        )
                        break
                if not found:
                    logger.error(
                        f"No Scale object found for segment '{segment_name}' "
                        f"in ScaleSet — manual scale factors not applied."
                    )
        if self.use_marker_placer:
            marker_placer: osim.MarkerPlacer = tool.getMarkerPlacer()
            marker_placer.setApply(True)
            marker_placer.setMarkerFileName(rel_marker_path)
            marker_placer.setOutputModelFileName(self.output_model_file)
            marker_placer.setTimeRange(time_array)

        return tool
