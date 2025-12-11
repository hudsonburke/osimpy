import json
import hashlib
from loguru import logger
import opensim as osim
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from .results import ScaleResult


class ScaleSettings(BaseModel):
    """Scale Tool settings.

    Configure and run the scale tool to scale a generic model to a subject's
    anthropometry based on marker data.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    unscaled_model_path: str = Field(
        description="Path to the unscaled/generic model file"
    )
    marker_set_path: str = Field(description="Path to the marker set file")
    marker_file: str = Field(description="Path to marker data file for scaling (.trc)")
    output_model_file: str = Field(description="Path for the output scaled model")
    results_directory: str = Field(
        ".", description="Directory used for writing results."
    )
    scale_factors: dict[str, tuple[float, float, float]] = Field(
        default_factory=dict,
        description="Scale factors for body segments {segment_name: (x, y, z)}",
    )
    preserve_mass_distribution: bool = Field(
        True, description="Preserve mass distribution when scaling"
    )
    subject_mass: float | None = Field(
        None, description="Subject's total mass (kg). If None, uses generic model mass"
    )
    time_range: tuple[float, float] | None = Field(
        None, description="Time range (start, end) for marker data to use in scaling"
    )

    def create_tool(self) -> osim.ScaleTool:
        """Create and configure a ScaleTool instance.

        Returns
        -------
        ScaleTool
            Configured ScaleTool instance
        """
        tool = osim.ScaleTool()

        # Configure GenericModelMaker
        generic_model_maker: osim.GenericModelMaker = tool.getGenericModelMaker()
        generic_model_maker.setModelFileName(self.unscaled_model_path)
        if self.marker_set_path:
            generic_model_maker.setMarkerSetFileName(self.marker_set_path)

        # Configure ModelScaler
        model_scaler: osim.ModelScaler = tool.getModelScaler()
        model_scaler.setApply(True)
        model_scaler.setMarkerFileName(self.marker_file)
        model_scaler.setPreserveMassDist(self.preserve_mass_distribution)

        if self.time_range is not None:
            time_array = osim.ArrayDouble()
            time_array.set(0, self.time_range[0])
            time_array.set(1, self.time_range[1])
            model_scaler.setTimeRange(time_array)

        if self.subject_mass is not None:
            tool.setSubjectMass(self.subject_mass)

        # Apply scale factors
        if self.scale_factors:
            scale_set: osim.ScaleSet = model_scaler.getScaleSet()
            for segment_name, factors in self.scale_factors.items():
                try:
                    vec = osim.Vec3(*factors)
                    scale_set.get(segment_name).setScaleFactors(vec)
                except Exception as e:
                    logger.warning(
                        f"Warning: Could not set scale factor for '{segment_name}': {e}"
                    )

        marker_placer: osim.MarkerPlacer = tool.getMarkerPlacer()
        marker_placer.setApply(True)
        marker_placer.setMarkerFileName(self.marker_file)
        marker_placer.setOutputModelFileName(self.output_model_file)
        marker_placer.setTimeRange(model_scaler.getTimeRange())

        # Set output model filename
        tool.getModelScaler().setOutputModelFileName(self.output_model_file)

        return tool

    def save_setup(self, filepath: str | None = None) -> str:
        """Save tool setup to XML file.

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
            results_dir = Path(self.results_directory)
            results_dir.mkdir(parents=True, exist_ok=True)
            filepath = str(results_dir / "scale_setup.xml")

        tool.printToXML(filepath)
        return filepath

    def run(self) -> ScaleResult:
        """Execute Scale Tool analysis and return results.

        Returns
        -------
        ScaleResult
            Structured Scale results with scaled model file path
        """
        # Ensure results directory exists
        results_dir = Path(self.results_directory)
        results_dir.mkdir(parents=True, exist_ok=True)

        warnings = []
        errors = []
        success = False
        start_time = datetime.now()

        try:
            # Create and configure tool
            tool = self.create_tool()

            # Save setup XML
            setup_file = self.save_setup()

            tool = osim.ScaleTool(setup_file)  # True = load model
            # Execute tool
            tool.run()
            success = True

        except Exception as e:
            errors.append(str(e))
            setup_file = str(results_dir / "failed_setup.xml")
            raise RuntimeError(f"Tool execution failed: {e}") from e
        finally:
            end_time = datetime.now()

        return ScaleResult(
            success=success,
            setup_file=setup_file,
            results_directory=self.results_directory,
            start_time=start_time,
            end_time=end_time,
            run_time=(end_time - start_time).total_seconds(),
            warnings=warnings,
            errors=errors,
            output_model_file=self.output_model_file,
            output_marker_set=None,  # TODO: Get from tool if generated
            input_marker_file=self.marker_file,
            settings_dict=self.to_dict(),
        )

    def to_dict(self) -> dict:
        """Export settings as dictionary for reproducibility.

        Returns:
            Dictionary containing all settings with JSON-compatible types.

        Example:
            >>> settings = ScaleSettings(...)
            >>> data = settings.to_dict()
            >>> # Can be saved as JSON for thesis documentation
        """
        return self.model_dump(mode="json")

    def save_json(self, filepath: str) -> None:
        """Save settings as JSON file for easy tracking.

        Args:
            filepath: Path to save JSON file

        Example:
            >>> settings = ScaleSettings(...)
            >>> settings.save_json("scale_settings.json")
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
            >>> settings = ScaleSettings.from_json("scale_settings.json")
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
