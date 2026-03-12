"""OpenSim export functionality."""

from typing import Any
import polars as pl
import numpy as np
import opensim as osim
from ..utils import get_unit_conversion
from .metadata import TRCMetadata, STOMetadata, MOTMetadata
import logging
from pathlib import Path
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# TRC is very similar to TSV -> What can I leverage for this?
# TODO: Version that bypasses the OpenSim API and writes the TRC file directly from the tensor, for faster export
def write_trc(filepath: Path, data: np.ndarray, metadata: TRCMetadata) -> None:
    with open(filepath) as f:
        f.write(f"PathFileType\t{metadata.PathFileType}\t{metadata.FileName}\n")


def write_sto(
    filepath: Path, data: np.ndarray, metadata: STOMetadata | MOTMetadata
) -> None:
    with open(filepath) as f:
        # for field_name, field_value in metadata.model_dump().items():
        #     f.write(f"{field_name}\t{field_value}\n")
        f.write(f"nRows\t{metadata.nRows}\n")
        f.write(f"nColumns\t{metadata.nColumns}\n")
        f.write(f"inDegrees\t{metadata.inDegrees}\n")
        # TODO
        # if "comments" in metadata.dict():
        #     f.write("\n".join(f"{comment}" for comment in metadata["comments"]) + "\n")
        # Write column labels


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


@dataclass
class OpenSimExternalForce:
    name: str
    applied_to_body: str
    force_expressed_in_body: str = "ground"
    point_expressed_in_body: str = "ground"
    force_identifier: str = r"force_v"
    point_identifier: str = r"force_p"
    torque_identifier: str = r"moment_"
    data_source_name: str | None = None


def force_to_opensim(force: OpenSimExternalForce) -> osim.ExternalForce:
    """
    Convert to OpenSim ExternalForce object.
    """
    ext_force = osim.ExternalForce()
    ext_force.setName(force.name)
    ext_force.setAppliedToBodyName(force.applied_to_body)
    ext_force.setForceExpressedInBodyName(force.force_expressed_in_body)
    ext_force.setPointExpressedInBodyName(force.point_expressed_in_body)
    ext_force.setForceIdentifier(force.force_identifier)
    ext_force.setPointIdentifier(force.point_identifier)
    ext_force.setTorqueIdentifier(force.torque_identifier)

    if force.data_source_name is not None:
        ext_force.set_data_source_name(force.data_source_name)

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
        ext_loads.cloneAndAppend(force_to_opensim(force))
    if datafile_name is not None:
        ext_loads.setDataFileName(datafile_name)
    ext_loads.printToXML(filepath)
