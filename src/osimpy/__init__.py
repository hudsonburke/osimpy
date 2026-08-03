from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("osimpy")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .moco import (
    CoordinateReserveOptimalForceOverride,
    MocoInverseResult,
    MocoInverseSettings,
    MocoTrackResult,
    MocoTrackSettings,
)
from .signals import (
    ResolvedSignal,
    SignalDescriptor,
    build_resolved_signal_index,
    build_signal_map,
    build_signal_match_index,
    common_signal_keys,
    infer_coordinate_file,
    infer_default_quantity,
    infer_motion_type,
    iter_signal_rows,
    normalize_signal_values,
    parse_signal_name,
    parse_signal_names,
    resolve_signal_units,
)
from .tools import (
    CMCResult,
    CMCSettings,
    IDResult,
    IDSettings,
    IKResult,
    IKSettings,
    RRAResult,
    RRASettings,
    ScaleResult,
    ScaleSettings,
    SOResult,
    SOSettings,
    ToolResult,
    ToolSettings,
)

__all__ = [
    "__version__",
    "CMCResult",
    "CMCSettings",
    "CoordinateReserveOptimalForceOverride",
    "IDResult",
    "IDSettings",
    "IKResult",
    "IKSettings",
    "MocoInverseResult",
    "MocoInverseSettings",
    "MocoTrackResult",
    "MocoTrackSettings",
    "RRAResult",
    "RRASettings",
    "ScaleResult",
    "ScaleSettings",
    "SOResult",
    "SOSettings",
    "ResolvedSignal",
    "SignalDescriptor",
    "build_resolved_signal_index",
    "ToolResult",
    "ToolSettings",
    "build_signal_map",
    "build_signal_match_index",
    "common_signal_keys",
    "infer_coordinate_file",
    "infer_default_quantity",
    "infer_motion_type",
    "iter_signal_rows",
    "normalize_signal_values",
    "parse_signal_name",
    "parse_signal_names",
    "resolve_signal_units",
]
