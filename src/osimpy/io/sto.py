"""OpenSim file reading functionality."""

import polars as pl
import polars.selectors as cs
from pathlib import Path
from typing import Literal, Any
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class STOMetadata:
    """
    https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53089996/Storage+.sto+Files

    Currently encapsulates MOTMetadaata as well since the formats are very similar
    """

    name: str = ""
    version: int | None = None
    nRows: int | None = None
    nColumns: int | None = None
    inDegrees: Literal["yes", "no", ""] = ""

    comments: list[str] = field(default_factory=list)
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def header(self):
        lines = []
        if self.name:
            lines.append(self.name)
        if self.version is not None:
            lines.append(f"version={self.version}")
        if self.nRows is not None:
            lines.append(f"nRows={self.nRows}")
        if self.nColumns is not None:
            lines.append(f"nColumns={self.nColumns}")
        if self.inDegrees:
            lines.append(f"inDegrees={self.inDegrees}")
        lines.extend(f"{key}={value}" for key, value in self.extra.items())
        if self.comments:
            lines.extend(self.comments)
        lines.append("endheader")
        return "\n".join(lines) + "\n"


# class MOTMetadata(STOMetadata):
#     """
#     https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53089415/Motion+.mot+Files
#     Currently this only implements option 2 from the documentation as the previous format is rarely used.
#     """


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_in_degrees(value: str | None) -> Literal["yes", "no", ""]:
    if value == "yes":
        return "yes"
    if value == "no":
        return "no"
    return ""


def sto_to_df(filepath: Path) -> tuple[pl.DataFrame, STOMetadata]:
    raw_metadata: dict[str, str] = {}
    comments: list[str] = []
    name = ""
    lines_to_skip = 0

    with open(filepath, "r") as f:
        for line in f:
            lines_to_skip += 1
            stripped = line.strip()

            if stripped.lower() == "endheader":
                break

            if not stripped:
                continue

            if "=" in stripped:
                key, value = stripped.split("=", 1)
                if key and value:
                    raw_metadata[key.strip()] = value.strip()
            elif not name:
                name = stripped
            else:
                comments.append(stripped)
        else:
            raise ValueError(f"Missing endheader in STO file: {filepath}")

    df = pl.read_csv(
        filepath, separator="\t", skip_lines=lines_to_skip, truncate_ragged_lines=True
    )

    # Drop phantom columns created by trailing tabs in the header line
    phantom_columns = [col for col in df.columns if col == ""]
    if phantom_columns:
        df = df.drop(phantom_columns)

    # Strip whitespace from columns
    df = df.with_columns(cs.string().str.strip_chars().cast(pl.Float64, strict=False))

    metadata = STOMetadata(
        name=name,
        version=_safe_int(raw_metadata.pop("version", None)),
        nRows=_safe_int(raw_metadata.pop("nRows", None)),
        nColumns=_safe_int(raw_metadata.pop("nColumns", None)),
        inDegrees=_normalize_in_degrees(raw_metadata.pop("inDegrees", "")),
        comments=comments,
        extra=raw_metadata,
    )

    actual_rows = len(df)
    actual_columns = len(df.columns)

    if metadata.nRows is not None and metadata.nRows != actual_rows:
        logger.warning(
            "Metadata 'nRows' does not match parsed row count: %s != %s",
            metadata.nRows,
            actual_rows,
        )
    if metadata.nColumns is not None and metadata.nColumns != actual_columns:
        logger.warning(
            "Metadata 'nColumns' does not match parsed column count: %s != %s",
            metadata.nColumns,
            actual_columns,
        )

    metadata.nRows = actual_rows
    metadata.nColumns = actual_columns

    return df, metadata


def df_to_sto(
    filepath: Path,
    df: pl.DataFrame,
    metadata: STOMetadata | None = None,
    precision: int = 8,
) -> None:
    """Can be used for both .sto and .mot files since the formats are very similar. Writes a Polars DataFrame to a .sto file with the appropriate header."""
    if metadata is None:
        metadata = STOMetadata(
            name=filepath.stem, nRows=len(df), nColumns=len(df.columns)
        )
    elif metadata.nRows is None:
        metadata.nRows = len(df)
    elif metadata.nRows != len(df):
        raise ValueError("Row count in metadata does not match DataFrame length")
    if metadata.nColumns is None:
        metadata.nColumns = len(df.columns)
    elif metadata.nColumns != len(df.columns):
        raise ValueError("Column count in metadata does not match DataFrame columns")
    with open(filepath, "w") as f:
        f.write(metadata.header)
        df.write_csv(f, separator="\t", include_header=True, float_precision=precision)


# Prefer df_to_sto
def export_mot(
    filepath: str,
    data: pl.DataFrame,
    metadata: dict[str, Any] = {},
    nans_as_zero: bool = True,
):
    import opensim as osim

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
