from pydantic import BaseModel, model_validator
from typing import Literal


class BaseMetadata(BaseModel):
    name: str = ""
    version: str = ""


class STOMetadata(BaseMetadata):
    """
    https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53089996/Storage+.sto+Files
    """

    nRows: int
    nColumns: int
    inDegrees: Literal["yes", "no"] = "no"


class MOTMetadata(STOMetadata):
    """
    https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53089415/Motion+.mot+Files
    Currently this only implements option 2 from the documentation as the previous format is rarely used.
    """

    comments: list[str] | None = None


class TRCMetadata(BaseMetadata):
    # Header
    PathFileType: int = 4
    FileName: str  # ?

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
