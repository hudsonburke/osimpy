"""Static Optimization tool wrapper.

Static Optimization runs as an ``Analysis`` inside ``osim.AnalyzeTool``.
This module wraps both layers into a single Pydantic settings model
with the same ``Settings → Result`` contract used by the rest of osimpy.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

import opensim as osim
from pydantic import Field, FilePath
from ..io import STOMetadata
from .tool import ToolResult, ToolSettings

logger = logging.getLogger(__name__)


class SOResult(ToolResult):
    """Result from Static Optimization analysis."""

    activation_file: FilePath | None = Field(
        None,
        description="Path to output activations (*_StaticOptimization_activation.sto)",
    )
    force_file: FilePath | None = Field(
        None,
        description="Path to output forces (*_StaticOptimization_force.sto)",
    )

    def load_activations(self) -> tuple[pl.DataFrame, STOMetadata]:
        """Load the output activations file as a DataFrame."""
        return self._load_sto(self.activation_file)

    def load_forces(self) -> tuple[pl.DataFrame, STOMetadata]:
        """Load the output forces file as a DataFrame."""
        return self._load_sto(self.force_file)


class SOSettings(ToolSettings[SOResult]):
    """Static Optimization settings.

    Wraps ``osim.AnalyzeTool`` + ``osim.StaticOptimization`` into a single
    Pydantic model.  The ``create_tool()`` method builds the AnalyzeTool,
    creates a StaticOptimization analysis, and adds it to the tool's
    AnalysisSet.

    References
    ----------
    OpenSim Static Optimization:
    https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53089631
    """

    # ── AnalyzeTool-level inputs ─────────────────────────────────────────
    external_loads_path: FilePath | None = Field(
        None, description="Path to external loads XML file"
    )
    force_set_paths: list[FilePath] = Field(
        default_factory=list,
        description="Paths to force set files (.xml) with reserve actuators",
    )
    coordinates_path: FilePath | None = Field(
        None,
        description="Motion/storage file with coordinate trajectories (IK output)",
    )
    states_path: FilePath | None = Field(
        None, description="States file (.sto) — alternative to coordinates_path"
    )
    speeds_path: FilePath | None = Field(
        None, description="Speeds file (.sto)"
    )

    replace_force_set: bool = Field(
        False,
        description="Replace (True) or append (False) model's force set",
    )
    solve_for_equilibrium_for_auxiliary_states: bool = Field(
        False,
        description="Compute equilibrium for states other than coords/speeds",
    )
    lowpass_cutoff_frequency_for_coordinates: float = Field(
        -1.0,
        description="Low-pass cutoff (Hz) for coordinate data; -1 = no filter",
    )
    output_precision: int = Field(8, description="Output precision")

    # ── Integrator settings (AnalyzeTool level) ──────────────────────────
    maximum_number_of_integrator_steps: int = Field(
        20000, description="Maximum number of integrator steps"
    )
    maximum_integrator_step_size: float = Field(
        1.0, description="Maximum integration step size (s)"
    )
    minimum_integrator_step_size: float = Field(
        1e-8, description="Minimum integration step size (s)"
    )
    integrator_error_tolerance: float = Field(
        1e-5, description="Integrator error tolerance"
    )

    # ── StaticOptimization analysis parameters ───────────────────────────
    use_model_force_set: bool = Field(
        True,
        description="Use the model's existing force set in the optimization",
    )
    activation_exponent: float = Field(
        2.0,
        description="Exponent applied to muscle activations in the objective (typically 2)",
    )
    use_muscle_physiology: bool = Field(
        True,
        description="Enforce muscle force-length-velocity constraints",
    )
    optimizer_convergence_criterion: float = Field(
        1e-4, description="Optimizer convergence tolerance"
    )
    optimizer_max_iterations: int = Field(
        100, description="Maximum optimizer iterations per time step"
    )

    def resolve_output_files(self) -> dict[str, Path | None]:
        results_dir = Path(self.results_directory)

        def _first_match(*patterns: str) -> Path | None:
            for pattern in patterns:
                matches = sorted(results_dir.glob(pattern))
                if matches:
                    return matches[0]
            return None

        return {
            "activation_file": _first_match(
                "*_StaticOptimization_activation.sto",
                "*StaticOptimization_activation*.sto",
            ),
            "force_file": _first_match(
                "*_StaticOptimization_force.sto",
                "*StaticOptimization_force*.sto",
            ),
        }

    def create_tool(self) -> osim.AnalyzeTool:
        """Build a configured ``osim.AnalyzeTool`` with a StaticOptimization analysis."""

        if self.setup_path is not None:
            tool = osim.AnalyzeTool(str(self.setup_path.resolve()))
        else:
            tool = osim.AnalyzeTool()

        rel_model_path = self.get_relative_path(self.model_path)
        rel_results_dir = self.get_relative_path(self.results_directory)

        tool.setModelFilename(rel_model_path)
        tool.setResultsDir(rel_results_dir)

        initial_time = self.initial_time
        final_time = self.final_time
        if initial_time is None or final_time is None:
            coord_file = self.coordinates_path or self.states_path
            if coord_file is not None:
                try:
                    sto = osim.Storage(str(coord_file.resolve()))
                    if initial_time is None:
                        initial_time = sto.getFirstTime()
                    if final_time is None:
                        final_time = sto.getLastTime()
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load coordinate data to set time range: {e}"
                    ) from e
            else:
                raise ValueError(
                    "Missing coordinates or states file to set missing time range"
                )

        tool.setInitialTime(initial_time)
        tool.setFinalTime(final_time)
        tool.setOutputPrecision(self.output_precision)

        tool.setReplaceForceSet(self.replace_force_set)
        if self.force_set_paths:
            arr = osim.ArrayStr()
            for fp in self.force_set_paths:
                arr.append(self.get_relative_path(fp))
            tool.setForceSetFiles(arr)

        if self.external_loads_path:
            tool.setExternalLoadsFileName(
                self.get_relative_path(self.external_loads_path)
            )

        if self.coordinates_path:
            tool.setCoordinatesFileName(
                self.get_relative_path(self.coordinates_path)
            )
        if self.states_path:
            tool.setStatesFileName(
                self.get_relative_path(self.states_path)
            )
        if self.speeds_path:
            tool.setSpeedsFileName(
                self.get_relative_path(self.speeds_path)
            )

        tool.setLowpassCutoffFrequency(self.lowpass_cutoff_frequency_for_coordinates)
        tool.setSolveForEquilibrium(self.solve_for_equilibrium_for_auxiliary_states)

        tool.setMaximumNumberOfSteps(self.maximum_number_of_integrator_steps)
        tool.setMaxDT(self.maximum_integrator_step_size)
        tool.setMinDT(self.minimum_integrator_step_size)
        tool.setErrorTolerance(self.integrator_error_tolerance)

        so = osim.StaticOptimization()
        so.setOn(True)
        so.setStartTime(initial_time)
        so.setEndTime(final_time)
        so.setStepInterval(1)
        so.setInDegrees(True)
        so.setUseModelForceSet(self.use_model_force_set)
        so.setActivationExponent(self.activation_exponent)
        so.setUseMusclePhysiology(self.use_muscle_physiology)
        so.setConvergenceCriterion(self.optimizer_convergence_criterion)
        so.setMaxIterations(self.optimizer_max_iterations)

        tool.getAnalysisSet().cloneAndAppend(so)

        return tool
