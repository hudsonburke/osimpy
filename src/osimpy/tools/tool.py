from datetime import datetime
from pathlib import Path
import os
from typing import Any, Generic, TypeVar, cast
from pydantic import (
    BaseModel,
    Field,
    FilePath,
    DirectoryPath,
)

import logging

logger = logging.getLogger(__name__)


def validate_extension(file: str, extension: str) -> str:
    """Validate that the file has the expected extension."""
    if not file.lower().endswith(extension):
        raise ValueError(f"File {file} does not have expected {extension} extension")
    return file


def validate_not_path(file: str) -> str:
    """Validate that the input is a filename, not a path."""
    if os.path.sep in file or (os.path.altsep and os.path.altsep in file):
        raise ValueError(f"Expected a filename, but got a path: {file}")
    return file


class ToolResult(BaseModel):
    """Base class for tool execution results.

    Stores metadata about the tool execution and paths to output files.
    Results are lightweight - actual data loading is deferred.
    """

    success: bool = Field(description="Whether the tool completed successfully")
    results_directory: DirectoryPath = Field(
        description="Directory containing output files. Also the working directory where the tool is executed."
    )

    start_time: datetime = Field(description="When the tool execution started")
    end_time: datetime = Field(description="When the tool execution finished")

    warnings: list[str] = Field(default_factory=list, description="Warning messages")
    errors: list[str] = Field(default_factory=list, description="Error messages")

    setup_file: FilePath = Field(description="Path to the tool setup XML file")


ResultT = TypeVar("ResultT", bound=ToolResult)


class ToolSettings(BaseModel, Generic[ResultT]):
    """Base class for OpenSim tool settings.

    This class can be extended to include common settings and methods for all tools.
    """

    name: str = Field(description="Name for this tool configuration (for tracking)")
    setup_path: FilePath | None = Field(
        None,
        description="Path to an existing setup XML template. "
        "When set, the tool is loaded from this file first, "
        "then individual settings override the template values.",
    )

    initial_time: float | None = Field(
        None, description="Initial time for the analysis (None = load from input data)"
    )
    final_time: float | None = Field(
        None, description="Final time for the simulation (None = load from input data)"
    )

    model_path: FilePath = Field(description="Path to OpenSim model file (.osim)")
    results_directory: DirectoryPath = Field(
        description="Directory for results and setup files"
    )
    output_setup_file: str | None = Field(
        description="Name for the output setup XML file (default: <name>_<tool_name>_setup.xml)"
    )

    @property
    def tool_name(self) -> str:
        return self.__class__.__name__.replace("Settings", "")

    def get_relative_path(self, p: Path) -> str:
        return os.path.relpath(p.resolve(), self.results_directory)

    def create_tool(self):
        """Create and configure the OpenSim tool instance.

        This method should be overridden by subclasses to return the specific tool.
        """
        raise NotImplementedError("Subclasses must implement create_tool()")

    def get_result_kwargs(self) -> dict[str, Any]:
        """Return subclass-specific result fields."""
        return {}

    def get_result_type(self) -> type[ResultT]:
        """Return the result model class for this tool."""
        return cast(type[ResultT], ToolResult)

    def build_result(
        self,
        *,
        success: bool,
        setup_file: str | Path,
        start_time: datetime,
        end_time: datetime,
        warnings: list[str],
        errors: list[str],
    ) -> ResultT:
        """Build the configured result model for this tool."""
        return cast(
            ResultT,
            self.get_result_type()(
                success=success,
                results_directory=Path(self.results_directory),
                start_time=start_time,
                end_time=end_time,
                warnings=warnings,
                errors=errors,
                setup_file=Path(setup_file),
                **self.get_result_kwargs(),
            ),
        )

    def run(self) -> ResultT:
        """Execute analysis and return results.

        The tool is run from the ``results_directory`` so that relative paths
        in the setup XML resolve correctly.

        Returns
        -------
        ToolResult
            Structured results
        """
        results_dir = Path(self.results_directory)
        results_dir.mkdir(parents=True, exist_ok=True)

        warnings: list[str] = []
        errors: list[str] = []
        success = False
        start_time = datetime.now()

        prev_dir = os.getcwd()

        output_setup_file = self.output_setup_file
        if output_setup_file is None:
            output_setup_file = str(
                results_dir / f"{self.name}_{self.tool_name.lower()}_setup.xml"
            )
        try:
            # Create and configure tool, save setup XML
            os.chdir(str(results_dir))

            tool = self.create_tool()  # Returns OpenSim tool instance
            tool.printToXML(output_setup_file)  # Save the setup XML for reproducibility
            tool.run()
            success = True
        except Exception as e:
            errors.append(str(e))
            logger.error(
                f"Error running {self.tool_name} tool with name {self.name}: {e}"
            )
        finally:
            os.chdir(prev_dir)
            end_time = datetime.now()

        return self.build_result(
            success=success,
            setup_file=output_setup_file,
            start_time=start_time,
            end_time=end_time,
            warnings=warnings,
            errors=errors,
        )
