import json
import hashlib
from loguru import logger
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field, FilePath, DirectoryPath


# TODO: Should the Settings be embedded in the Result?

class ToolResult(BaseModel):
    """Base class for tool execution results.

    Stores metadata about the tool execution and paths to output files.
    Results are lightweight - actual data loading is deferred.
    """

    success: bool = Field(description="Whether the tool completed successfully")
    results_directory: DirectoryPath = Field(description="Directory containing output files")

    start_time: datetime = Field(description="When the tool execution started")
    end_time: datetime = Field(description="When the tool execution finished")
    run_time: float = Field(description="Execution time in seconds")

    warnings: list[str] = Field(default_factory=list, description="Warning messages")
    errors: list[str] = Field(default_factory=list, description="Error messages")

    setup_file: FilePath = Field(description="Path to the tool setup XML file")

class ToolSettings(BaseModel):
    """Base class for OpenSim tool settings.

    This class can be extended to include common settings and methods for all tools.
    """

    model_file: FilePath = Field(description="Path to OpenSim model file (.osim)")
    results_directory: DirectoryPath = Field(description="Directory for results and setup files")

    def create_tool(self):
        """Create and configure the OpenSim tool instance.

        This method should be overridden by subclasses to return the specific tool.
        """
        raise NotImplementedError("Subclasses must implement create_tool()")

    # def run(self):
    #     """Execute analysis using XML-based workflow.

    #     Save settings to XML, load tool from XML (which loads the model), then run.

    #     Returns
    #     -------
    #     ToolResult
    #         Structured tool results with forces file path and metadata
    #     """
    #     # Ensure results directory exists
    #     results_dir = Path(self.results_directory)
    #     results_dir.mkdir(parents=True, exist_ok=True)

    #     warnings = []
    #     errors = []
    #     success = False
    #     start_time = datetime.now()

    #     try:
    #         # Create and configure tool (for XML generation)
    #         tool = self.create_tool()

    #         # Save setup XML with model_file injected
    #         setup_file = self.save_setup()

    #         tool = self # TODO

    #         # Execute tool
    #         tool.run()
    #         success = True

    #     except Exception as e:
    #         errors.append(str(e))
    #         setup_file = str(results_dir / "failed_setup.xml")
    #         raise RuntimeError(f"Tool execution failed: {e}") from e

    #     finally:
    #         end_time = datetime.now()

    #     return 

    def save_setup(self, filepath: str | None = None) -> str:
        """Save tool setup to XML file with model file path.

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
            results_dir = Path(self.results_directory)
            results_dir.mkdir(parents=True, exist_ok=True)
            tool_name = self.__class__.__name__.replace("Settings", "").lower()
            filepath = str(results_dir / f"{tool_name}_setup.xml")

        # Write to XML
        tool.printToXML(filepath)

        return filepath

    def to_dict(self) -> dict:
        """Export settings as dictionary for reproducibility.

        Returns:
            Dictionary containing all settings with JSON-compatible types.

        Example:
            >>> settings = Settings(...)
            >>> data = settings.to_dict()
        """
        return self.model_dump(mode="json")

    def save_json(self, filepath: str) -> None:
        """Save settings as JSON file for easy tracking.

        Args:
            filepath: Path to save JSON file

        Example:
            >>> settings = Settings(...)
            >>> settings.save_json("settings.json")
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
            >>> settings = Settings.from_json("settings.json")
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
