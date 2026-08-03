"""OpenSim export functionality."""

import polars as pl
import numpy as np
import opensim as osim
from ..utils import get_unit_conversion
import logging
from pathlib import Path
from pydantic import BaseModel, model_validator

logger = logging.getLogger(__name__)


class TRCMetadata(BaseModel):
    name: str = ""
    # Header
    PathFileType: int = 4

    DataRate: float
    CameraRate: float
    NumFrames: int
    NumMarkers: int
    Units: str

    OrigDataRate: float | None = None
    OrigDataStartFrame: int | None = None
    OrigNumFrames: int | None = None

    MarkerNames: list[str]

    @model_validator(mode="after")
    def set_orig_fields(self) -> "TRCMetadata":
        if self.OrigDataRate is None:
            self.OrigDataRate = self.DataRate
        if self.OrigNumFrames is None:
            self.OrigNumFrames = self.NumFrames
        if self.OrigDataStartFrame is None:
            self.OrigDataStartFrame = 1
        return self

    @model_validator(mode="after")
    def ensure_num_markers(self) -> "TRCMetadata":
        if self.NumMarkers != len(self.MarkerNames):
            raise ValueError(
                f"NumMarkers {self.NumMarkers} does not match length of MarkerNames {len(self.MarkerNames)}"
            )
        return self

    @property
    def header(self) -> str:
        return "\n".join([])


def trc_to_df(filepath: Path) -> tuple[pl.DataFrame, TRCMetadata]:
    with open(filepath, "r") as f:
        metadata = TRCMetadata(name=filepath.name, NumFrames=df.shape[0])
        df = pl.read_csv(f, separator="\t", skip_rows=10)
    return (df, metadata)


def df_to_trc(
    filepath: Path,
    df: pl.DataFrame,
    metadata: TRCMetadata | None = None,
    precision: int = 5,
) -> None:
    if metadata is None:
        metadata = TRCMetadata(
            name=filepath.stem,
            NumFrames=df.shape[0],
            NumMarkers=df.shape[1],
        )
    with open(filepath, "w") as f:
        f.write(metadata.header)

        f.write("\n")  # Empty line between header and data
        df.write_csv(f, separator="\t", float_precision=precision)


def export_tensor_as_trc(
    filepath: str,
    markers_tensor: np.ndarray,  # Expected shape: (Frames, Markers, 3)
    marker_names: list[str],
    time: np.ndarray,
    rate: float,
    units: str,
    output_units: str | None = None,
    rotation: np.ndarray = np.eye(3),
) -> None:
    """Export marker data to TRC file format used by OpenSim."""

    num_frames, _, dims = markers_tensor.shape
    if dims != 3:
        raise ValueError("All marker coordinates must be 3D")
    if num_frames != len(time):
        raise ValueError("Frames in tensor must match time array length")

    conversion_factor = 1.0
    if output_units is not None and units != output_units:
        logger.warning(
            f"Output units {output_units} do not match points units {units}. Converting coordinates."
        )
        conversion_factor = get_unit_conversion(units, output_units)

    processed_tensor = (markers_tensor @ rotation.T) * conversion_factor

    # Set up OpenSim Table
    table = osim.TimeSeriesTableVec3()
    table.setColumnLabels(marker_names)

    table.addTableMetaDataString(
        "Units", units if output_units is None else output_units
    )
    table.addTableMetaDataString("DataRate", str(rate))

    # Iterating is required by the osim C++ API
    for i in range(num_frames):
        row = [osim.Vec3(*coords) for coords in processed_tensor[i]]
        table.appendRow(time[i], osim.RowVectorVec3(row))

    adapter = osim.TRCFileAdapter()
    adapter.write(table, filepath)
