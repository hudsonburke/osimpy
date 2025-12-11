"""Result classes for OpenSim tool executions."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from loguru import logger
from pydantic import BaseModel, Field, ConfigDict


class ToolResult(BaseModel):
    """Base class for tool execution results.

    Stores metadata about the tool execution and paths to output files.
    Results are lightweight - actual data loading is deferred.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool = Field(description="Whether the tool completed successfully")
    setup_file: str = Field(description="Path to the tool setup XML file")
    results_directory: str = Field(description="Directory containing output files")
    start_time: datetime = Field(description="When the tool execution started")
    end_time: datetime = Field(description="When the tool execution finished")
    run_time: float = Field(description="Execution time in seconds")
    warnings: list[str] = Field(default_factory=list, description="Warning messages")
    errors: list[str] = Field(default_factory=list, description="Error messages")
    settings_dict: dict[str, Any] | None = Field(
        None,
        description="Settings used to generate this result"
    )
    
    @property
    def output_dir(self) -> Path:
        """Get results directory as Path object."""
        return Path(self.results_directory)
    
    def get_output_file(self, filename: str) -> Path:
        """Get full path to an output file.

        Parameters
        ----------
        filename : str
            Name of the output file

        Returns
        -------
        Path
            Full path to the output file
        """
        return self.output_dir / filename

    def save_provenance(self, filepath: str) -> None:
        """Save complete provenance information for reproducibility.

        Creates a JSON file with all execution details including settings,
        timing, and file paths. Essential for thesis reproducibility.

        Args:
            filepath: Path to save provenance JSON file

        Example:
            >>> result = ik_settings.run()
            >>> result.save_provenance("ik_provenance.json")
            >>> # Creates complete record for thesis documentation
        """
        provenance = {
            "tool": self.__class__.__name__.replace("Result", ""),
            "execution": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "run_time_seconds": self.run_time,
                "success": self.success,
            },
            "settings": self.settings_dict,
            "outputs": {
                "results_directory": self.results_directory,
                "setup_file": self.setup_file,
            },
            "warnings": self.warnings,
            "errors": self.errors,
            "movedb_version": self._get_version(),
        }

        # Add tool-specific outputs
        provenance["outputs"].update(self._get_tool_specific_outputs())

        with open(filepath, 'w') as f:
            json.dump(provenance, f, indent=2)

        logger.success(f"Saved provenance to {filepath}")

    def _get_version(self) -> str:
        """Get MoveDB version."""
        try:
            import movedb
            return movedb.__version__
        except AttributeError:
            return "unknown"

    def _get_tool_specific_outputs(self) -> dict:
        """Override in subclasses to add tool-specific output paths."""
        return {}

    @classmethod
    def from_provenance(cls, filepath: str):
        """Load result from provenance file.

        Args:
            filepath: Path to provenance JSON file

        Returns:
            Result instance (note: execution times will be from original run)

        Example:
            >>> result = IKResult.from_provenance("ik_provenance.json")
            >>> print(f"Tool ran for {result.run_time} seconds")
        """
        with open(filepath) as f:
            data = json.load(f)

        # Reconstruct result from provenance
        return cls(
            success=data["execution"]["success"],
            setup_file=data["outputs"]["setup_file"],
            results_directory=data["outputs"]["results_directory"],
            start_time=datetime.fromisoformat(data["execution"]["start_time"]),
            end_time=datetime.fromisoformat(data["execution"]["end_time"]),
            run_time=data["execution"]["run_time_seconds"],
            warnings=data.get("warnings", []),
            errors=data.get("errors", []),
            settings_dict=data.get("settings"),
            **cls._parse_tool_specific_outputs(data["outputs"])
        )

    @classmethod
    def _parse_tool_specific_outputs(cls, outputs: dict) -> dict:
        """Override in subclasses to parse tool-specific outputs."""
        return {}


class IKResult(ToolResult):
    """Result from Inverse Kinematics analysis.
    
    Attributes
    ----------
    output_motion_file : str
        Path to the output motion (.mot) file containing computed coordinates
    marker_file : str
        Path to the input marker (.trc) file used
    """
    output_motion_file: str = Field(description="Path to output motion file (.mot)")
    marker_file: str = Field(description="Path to input marker file (.trc)")
    
    @property
    def motion_path(self) -> Path:
        """Get path to motion file."""
        return Path(self.output_motion_file)

    def _get_tool_specific_outputs(self) -> dict:
        return {
            "output_motion_file": self.output_motion_file,
            "marker_file": self.marker_file,
        }

    @classmethod
    def _parse_tool_specific_outputs(cls, outputs: dict) -> dict:
        return {
            "output_motion_file": outputs.get("output_motion_file", ""),
            "marker_file": outputs.get("marker_file", ""),
        }


class IDResult(ToolResult):
    """Result from Inverse Dynamics analysis.
    
    Attributes
    ----------
    output_forces_file : str
        Path to the output forces (.sto) file containing generalized forces
    coordinates_file : str
        Path to the input coordinates (.mot) file used
    """
    output_forces_file: str = Field(description="Path to output forces file (.sto)")
    coordinates_file: str = Field(description="Path to input coordinates file (.mot)")
    external_loads_file: str | None = Field(None, description="Path to external loads file if used")
    
    @property
    def forces_path(self) -> Path:
        """Get path to forces file."""
        return Path(self.output_forces_file)

    def _get_tool_specific_outputs(self) -> dict:
        return {
            "output_forces_file": self.output_forces_file,
            "coordinates_file": self.coordinates_file,
            "external_loads_file": self.external_loads_file,
        }

    @classmethod
    def _parse_tool_specific_outputs(cls, outputs: dict) -> dict:
        return {
            "output_forces_file": outputs.get("output_forces_file", ""),
            "coordinates_file": outputs.get("coordinates_file", ""),
            "external_loads_file": outputs.get("external_loads_file"),
        }


class CMCResult(ToolResult):
    """Result from Computed Muscle Control analysis.
    
    Attributes
    ----------
    output_controls_file : str
        Path to the output controls file
    output_kinematics_file : str
        Path to the output kinematics file
    desired_kinematics_file : str
        Path to the input desired kinematics file
    """
    output_controls_file: str = Field(description="Path to output controls file")
    output_kinematics_file: str = Field(description="Path to output kinematics file")
    desired_kinematics_file: str = Field(description="Path to input desired kinematics")
    
    @property
    def controls_path(self) -> Path:
        """Get path to controls file."""
        return Path(self.output_controls_file)
    
    @property
    def kinematics_path(self) -> Path:
        """Get path to kinematics file."""
        return Path(self.output_kinematics_file)


class ScaleResult(ToolResult):
    """Result from Scale Tool analysis.
    
    Attributes
    ----------
    output_model_file : str
        Path to the scaled model file
    output_marker_set : str
        Path to the output marker set file
    input_marker_file : str
        Path to the input marker file used for scaling
    """
    output_model_file: str = Field(description="Path to scaled model file")
    output_marker_set: str | None = Field(None, description="Path to output marker set")
    input_marker_file: str = Field(description="Path to input marker file")
    
    @property
    def model_path(self) -> Path:
        """Get path to scaled model."""
        return Path(self.output_model_file)

    def _get_tool_specific_outputs(self) -> dict:
        return {
            "output_model_file": self.output_model_file,
            "output_marker_set": self.output_marker_set,
            "input_marker_file": self.input_marker_file,
        }

    @classmethod
    def _parse_tool_specific_outputs(cls, outputs: dict) -> dict:
        return {
            "output_model_file": outputs.get("output_model_file", ""),
            "output_marker_set": outputs.get("output_marker_set"),
            "input_marker_file": outputs.get("input_marker_file", ""),
        }
