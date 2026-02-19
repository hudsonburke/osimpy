import numpy as np
from functools import cached_property
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    PositiveFloat,
    model_validator,
)
from numpydantic import NDArray
from typing import Literal  # Better for static type checking than numpydantic.Shape

SingleMarkerArray = NDArray[Literal["* frames, 3 xyz"], np.float64]
MarkerArray = NDArray[Literal["* frames, * markers, 3 xyz"], np.float64]
ResidualsArray = NDArray[Literal["* frames, * markers"], np.float64]
SingleResidualsArray = NDArray[Literal["* frames"], np.float64]
TimeVector = NDArray[Literal["* frames"], np.float64]


class MarkerData(BaseModel):
    """Marker trajectory data structure."""

    data: MarkerArray = Field(
        description="Marker positions array of shape (n_frames, n_markers, 3) - xyz coordinates"
    )
    marker_names: list[str] = Field(
        description="List of marker names corresponding to second dimension of data"
    )
    rate: PositiveFloat = Field(description="Sampling rate in Hz")
    units: str = Field(description="Position units (e.g., 'mm', 'm')")
    first_frame: PositiveInt = Field(
        description="First frame number in the trial", default=1
    )
    residuals: ResidualsArray | None = Field(
        description="Optional residuals array of shape (n_frames, n_markers)",
        default=None,
    )

    def get_marker_index(self, marker_name: str) -> int:
        """Get the index of a marker by name."""
        try:
            return self.marker_names.index(marker_name)
        except ValueError:
            raise ValueError(f"Marker name '{marker_name}' not found in marker_names.")

    def get_marker_data(self, marker_name: str) -> SingleMarkerArray:
        index = self.get_marker_index(marker_name)
        return np.asarray(self.data)[:, index, :]

    @cached_property
    def num_frames(self) -> int:
        """Return the number of frames in the data."""
        return np.asarray(self.data).shape[0]

    @cached_property
    def time_vector(self) -> TimeVector:
        """Generate time vector based on rate and number of frames."""
        return (
            np.arange(self.num_frames) / self.rate
        )  # TODO: should first_frame be considered?

    @model_validator(mode="after")
    def check_shapes(self) -> "MarkerData":
        """Validate array dimensions match metadata."""
        data = np.asarray(self.data)
        n_frames, n_markers, dims = data.shape

        if dims != 3:
            raise ValueError(
                f"Data must be 3D (n_frames, n_markers, 3). Got shape {data.shape}"
            )

        if len(self.marker_names) != n_markers:
            raise ValueError(
                f"Mismatch: {len(self.marker_names)} marker names provided, "
                f"but data array has {n_markers} markers."
            )

        if self.residuals is not None:
            residuals = np.asarray(self.residuals)
            if residuals.shape != (n_frames, n_markers):
                raise ValueError(
                    f"Residuals shape {residuals.shape} does not match "
                    f"expected data shape ({n_frames}, {n_markers})."
                )
        return self
