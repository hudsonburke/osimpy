import numpy as np
from typing import Literal
from pydantic import (
    BaseModel,
    Field,
    model_validator,
    PositiveFloat,
)
from numpydantic import NDArray

Array3D = NDArray[Literal["* frames, 3 xyz"], np.float64]

Origin = NDArray[Literal["3 xyz"], np.float64]
Corners = NDArray[Literal["4 corners, 3 xyz"], np.float64]
CalMatrix = NDArray[Literal["6 rows, 6 columns"], np.float64]


class ForceplateData(BaseModel):
    """Force plate data structure."""

    forces: Array3D = Field(
        description="Force vectors array of shape (n_frames, 3) - xyz components"
    )
    moments: Array3D = Field(
        description="Moment vectors array of shape (n_frames, 3) - xyz components"
    )
    cop: Array3D = Field(
        description="Center of pressure array of shape (n_frames, 3) - xyz coordinates"
    )
    cal_matrix: CalMatrix = Field(description="Calibration matrix of shape (6, 6)")
    corners: Corners = Field(
        description="Corner coordinates of shape (4, 3) - xyz for each corner"
    )
    origin: Origin = Field(description="Origin coordinates of shape (3,) - xyz")
    rate: PositiveFloat = Field(description="Sampling rate in Hz")
    unit_force: str = Field(description="Force units (e.g., 'N')", default="N")
    unit_moment: str = Field(description="Moment units (e.g., 'Nm')", default="Nm")
    unit_position: str = Field(
        description="Position units (e.g., 'm', 'mm')", default="m"
    )

    @model_validator(mode="after")
    def check_shapes(self) -> "ForceplateData":
        forces = np.asarray(self.forces)
        force_frames, _ = forces.shape

        moments = np.asarray(self.moments)
        moment_frames, _ = moments.shape
        if moment_frames != force_frames:
            raise ValueError(
                f"Moments frames {moment_frames} do not match forces frames {force_frames}."
            )
        cop = np.asarray(self.cop)
        cop_frames, _ = cop.shape
        if cop_frames != force_frames:
            raise ValueError(
                f"CoP frames {cop_frames} do not match forces frames {force_frames}."
            )
        return self
