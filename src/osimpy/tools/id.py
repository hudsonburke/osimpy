import json
import hashlib
from loguru import logger
import opensim as osim
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from .results import IDResult


class IDSettings(BaseModel):
    """Inverse Dynamics tool settings.

    Configure and run inverse dynamics analysis to compute generalized forces
    from joint kinematics and external forces.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_file: str = Field(description="Path to OpenSim model file (.osim)")
    coordinates_file: str = Field(description="Path to coordinates file (.mot) from IK")
    output_forces_file: str = Field(description="Path for output forces file (.sto)")
    results_directory: str = Field(
        ".", description="Directory for results and setup files"
    )

    # Time range
    initial_time: float = Field(
        -1.0, description="Initial time for analysis (-1 = auto from file)"
    )
    final_time: float = Field(
        -1.0, description="Final time for analysis (-1 = auto from file)"
    )

    # Optional settings
    external_loads_file: str | None = Field(
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

        tool.setModelFileName(self.model_file)

        # Set results directory
        tool.setResultsDir(self.results_directory)

        # Set external loads if provided
        if self.external_loads_file:
            tool.setExternalLoadsFileName(self.external_loads_file)

        # Set coordinates and output files
        tool.setCoordinatesFileName(self.coordinates_file)
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
            sto = osim.Storage(self.coordinates_file)
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

    def save_setup(self, filepath: str | None = None) -> str:
        """Save ID tool setup to XML file with model file path.

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

        return filepath

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
            setup_file=setup_file,
            results_directory=self.results_directory,
            start_time=start_time,
            end_time=end_time,
            run_time=(end_time - start_time).total_seconds(),
            warnings=warnings,
            errors=errors,
            output_forces_file=self.output_forces_file,
            coordinates_file=self.coordinates_file,
            external_loads_file=self.external_loads_file
            if self.external_loads_file
            else None,
            settings_dict=self.to_dict(),
        )

    def to_dict(self) -> dict:
        """Export settings as dictionary for reproducibility.

        Returns:
            Dictionary containing all settings with JSON-compatible types.

        Example:
            >>> settings = IDSettings(...)
            >>> data = settings.to_dict()
            >>> # Can be saved as JSON for thesis documentation
        """
        return self.model_dump(mode="json")

    def save_json(self, filepath: str) -> None:
        """Save settings as JSON file for easy tracking.

        Args:
            filepath: Path to save JSON file

        Example:
            >>> settings = IDSettings(...)
            >>> settings.save_json("id_settings.json")
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
            >>> settings = IDSettings.from_json("id_settings.json")
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
