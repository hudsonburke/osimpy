import numpy as np
from functools import cached_property
from pydantic import (
    BaseModel,
    Field,
    model_validator,
    PositiveInt,
    PositiveFloat,
)
from numpydantic import NDArray
from typing import Literal  # Better for static type checking than numpydantic.Shape

AnalogArray = NDArray[Literal["* frames, * channels"], np.float64]
Array1D = NDArray[Literal["* frames"], np.float64]


class AnalogData(BaseModel):
    """Analog signal data structure."""

    data: AnalogArray = Field(
        description="Analog signals array of shape (n_frames, n_channels)"
    )
    channel_names: list[str] = Field(
        description="List of channel names corresponding to second dimension of data"
    )
    rate: PositiveFloat = Field(description="Sampling rate in Hz")
    units: str = Field(description="Signal units (e.g., 'V', 'mV')")
    first_frame: PositiveInt = Field(description="First frame number in the trial")

    def get_channel_index(self, channel_name: str) -> int:
        """Get the index of a channel by name."""
        try:
            return self.channel_names.index(channel_name)
        except ValueError:
            raise ValueError(
                f"Channel name '{channel_name}' not found in channel_names."
            )

    def get_channel_data(self, channel_name: str) -> Array1D:
        index = self.get_channel_index(channel_name)
        return np.asarray(self.data)[:, index]

    @cached_property
    def num_frames(self) -> int:
        """Return the number of frames in the data."""
        return np.asarray(self.data).shape[0]

    @cached_property
    def time_vector(self) -> Array1D:
        """Generate time vector based on rate and number of frames."""
        return np.arange(self.num_frames) / self.rate

    @model_validator(mode="after")
    def check_shapes(self) -> "AnalogData":
        """Validate array dimensions match metadata."""
        data = np.asarray(self.data)
        _, n_channels = data.shape

        if len(self.channel_names) != n_channels:
            raise ValueError(
                f"Mismatch: {len(self.channel_names)} channel names provided, "
                f"but data array has {n_channels} channels."
            )
        return self
