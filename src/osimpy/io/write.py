"""OpenSim export functionality."""

from typing import Any
from pydantic import BaseModel
import polars as pl
import numpy as np
import opensim as osim
from ..utils import get_unit_conversion
from .metadata import TRCMetadata, STOMetadata, MOTMetadata
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# TRC is very similar to TSV -> What can I leverage for this?
# TODO: Version that bypasses the OpenSim API and writes the TRC file directly from the tensor, for faster export of large datasets. This will require careful handling of the TRC file format, including headers and metadata, but can be much faster than iterating through frames in Python and calling OpenSim's C++ API for each row.
def write_trc(filepath: Path, data: np.ndarray, metadata: TRCMetadata) -> None:
    with open(filepath) as f:
        f.write(f"PathFileType\t{metadata.PathFileType}\t{metadata.FileName}\n")


def write_sto(
    filepath: Path, data: pl.DataFrame, metadata: STOMetadata | MOTMetadata
) -> None:
    with open(filepath) as f:
        f.write(f"nRows\t{metadata.nRows}\n")
        f.write(f"nColumns\t{metadata.nColumns}\n")
        f.write(f"inDegrees\t{metadata.inDegrees}\n")
        # if "comments" in metadata.dict():
        #     f.write("\n".join(f"{comment}" for comment in metadata["comments"]) + "\n")
        # Write column labels
        f.write("\t".join(data.columns) + "\n")
        # Write data rows
        for row in data.iter_rows(named=False):
            f.write("\t".join(str(val) for val in row) + "\n")


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
    assert dims == 3, "All marker coordinates must be 3D"
    assert num_frames == len(time), "Frames in tensor must match time array length"

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

    rate = float(rate.item()) if isinstance(rate, np.ndarray) else float(rate)
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


def export_trc(
    filepath: str,
    markers: dict[str, np.ndarray],
    time: np.ndarray,
    rate: float,
    units: str,
    output_units: str | None = None,
    rotation: np.ndarray = np.eye(3),
) -> None:
    """
    Export marker data to TRC file format used by OpenSim
    """
    # Markers is expected to be a dict of marker name to Nx3 numpy array of coordinates
    num_frames = len(time)
    if any(len(coords) != num_frames for coords in markers.values()):
        raise ValueError(
            "All markers must have the same number of frames as the time array"
        )
    assert all(coords.shape[1] == 3 for coords in markers.values()), (
        "All marker coordinates must be 3D"
    )

    table = osim.TimeSeriesTableVec3()
    marker_names = list(markers.keys())
    table.setColumnLabels(marker_names)
    conversion_factor = 1.0
    if output_units is not None and units != output_units:
        logger.warning(
            f"Output units {output_units} do not match points units {units}. Converting coordinates."
        )
        conversion_factor = get_unit_conversion(units, output_units)

    # Ensure rate is a scalar (extract from numpy array if needed)
    if isinstance(rate, np.ndarray):
        rate = float(rate.item())
    else:
        rate = float(rate)

    table.addTableMetaDataString(
        "Units", units if output_units is None else output_units
    )
    table.addTableMetaDataString("DataRate", str(rate))
    for frame in range(num_frames):
        row = []
        for marker_name, coords in markers.items():
            in_coords = coords[frame]
            if in_coords is not None:
                coords_rotated = np.array(
                    rotation @ np.array(in_coords).T
                ).T  # Apply rotation if needed
                coords_converted = (
                    coords_rotated * conversion_factor
                )  # Convert coordinates if needed
            else:
                coords_converted = np.array([np.nan, np.nan, np.nan])
            row.append(
                osim.Vec3(coords_converted[0], coords_converted[1], coords_converted[2])
            )
        time_val = time[frame]
        table.appendRow(time_val, osim.RowVectorVec3(row))
    adapter = osim.TRCFileAdapter()
    adapter.write(table, filepath)


def export_mot(
    filepath: str,
    data: pl.DataFrame,
    metadata: dict[str, Any] = {},
    nans_as_zero: bool = True,
):
    """
    Export data to OpenSim MOT file format.
    """
    mot_table = osim.TimeSeriesTable()

    if "time" not in data.columns:
        raise ValueError("Data must contain a 'time' column for MOT export")

    if nans_as_zero:
        # Replace NaNs with zeros in the data
        data = data.with_columns(
            [pl.col(col).fill_nan(0.0) for col in data.columns if col != "time"]
        )

    for row in data.iter_rows(named=True):
        time_val = row["time"]
        row_data = [row[col] for col in data.columns if col != "time"]
        mot_table.appendRow(time_val, osim.RowVector(row_data))

    column_labels = [col for col in data.columns if col != "time"]
    mot_table.setColumnLabels(column_labels)

    n_rows = len(data)
    metadata_rows = metadata.pop("nRows", None)
    if metadata_rows is not None and str(metadata_rows) != str(n_rows):
        logger.warning(
            f"Metadata 'nRows' does not match data length: {metadata.get('nRows', 'None')} != {n_rows}"
        )
    mot_table.addTableMetaDataString("nRows", str(n_rows))

    n_columns = len(data.columns)
    metadata_columns = metadata.pop("nColumns", None)
    if metadata_columns is not None and str(metadata_columns) != str(n_columns):
        logger.warning(
            f"Metadata 'nColumns' does not match data columns: {metadata.get('nColumns', 'None')} != {n_columns}"
        )
    mot_table.addTableMetaDataString("nColumns", str(n_columns))

    for key, value in metadata.items():
        mot_table.addTableMetaDataString(key, str(value))
    mot_file = osim.STOFileAdapter()
    mot_file.write(mot_table, filepath)


class OpenSimExternalForce(BaseModel):
    name: str
    applied_to_body: str
    force_expressed_in_body: str = "ground"
    point_expressed_in_body: str = "ground"
    force_identifier: str = r"force_v"
    point_identifier: str = r"force_p"
    torque_identifier: str = r"moment_"
    data_source_name: str | None = None

    def to_opensim(self) -> osim.ExternalForce:
        """
        Convert to OpenSim ExternalForce object.
        """
        ext_force = osim.ExternalForce()
        ext_force.setName(self.name)
        ext_force.setAppliedToBodyName(self.applied_to_body)
        ext_force.setForceExpressedInBodyName(self.force_expressed_in_body)
        ext_force.setPointExpressedInBodyName(self.point_expressed_in_body)
        ext_force.setForceIdentifier(self.force_identifier)
        ext_force.setPointIdentifier(self.point_identifier)
        ext_force.setTorqueIdentifier(self.torque_identifier)

        if self.data_source_name is not None:
            ext_force.set_data_source_name(self.data_source_name)

        return ext_force


def export_external_loads(
    filepath: str,
    external_forces: list[OpenSimExternalForce],
    datafile_name: str | None = None,
) -> None:
    """
    Export external loads to OpenSim ExternalLoads .xml file.
    """
    ext_loads = osim.ExternalLoads()
    for force in external_forces:
        ext_loads.cloneAndAppend(force.to_opensim())
    if datafile_name is not None:
        ext_loads.setDataFileName(datafile_name)
    ext_loads.printToXML(filepath)
