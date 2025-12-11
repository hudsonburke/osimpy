"""OpenSim file reading functionality."""
import numpy as np
import polars as pl
from pydantic import BaseModel

# TODO: Maybe use this for the metadata return
class BaseMetadata(BaseModel):
    name: str = ''
    version: str = ''

class STOMetadata(BaseMetadata):
    pass
class TRCMetadata(BaseMetadata):
    pass

def sto_to_numpy(file_path: str) -> np.ndarray: #TODO
    return np.zeros(1)

def sto_to_df(file_path: str) -> tuple[pl.DataFrame, dict[str, str]]:
    """
    Reads a .sto or .mot file and returns a Polars DataFrame.

    Args:
        file_path (str): Path to the .sto or .mot file.

    Returns:
        tuple: A tuple containing a Polars DataFrame with the data and a dictionary with metadata.
    """
    # Read the header of the file to determine number of lines to skip
    file_metadata = {"name": "", "comments": []}
    lines_to_skip = 1
    with open(file_path, "r") as f:
        line = f.readline()
        if "=" in line:
            key, value = line.split("=", 1)
            if key and value:
                file_metadata[key.lower()] = value.strip()
            line = f.readline()  # Read the next line after the key-value pair
            lines_to_skip += 1
        elif not line.startswith("endheader"):
            file_metadata["name"] = line.strip()
            line = f.readline()  # Second line should start the key value pairs
            lines_to_skip += 1
        else:  # If the first line is 'endheader', do not enter the loop
            file_metadata["name"] = "Unnamed File"
        while line and not line.startswith("endheader"):
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                if key and value:
                    file_metadata[key.lower()] = value.strip()
            elif line:  # Treat as a comment or empty line
                file_metadata["comments"].append(line)
            line = f.readline()  # Read until 'endheader'
            lines_to_skip += 1

    df = pl.read_csv(
        file_path, separator="\t", skip_lines=lines_to_skip, truncate_ragged_lines=True
    )
    # Strip whitespace from columns
    df = df.with_columns(
        [
            pl.col(col).cast(pl.String).str.strip_chars().cast(pl.Float64)
            for col in df.columns
        ]
    )
    return df, file_metadata
