from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Literal, Sequence, cast


SignalQuantity = Literal[
    "value",
    "speed",
    "activation",
    "fiber_length",
    "force",
    "moment",
    "control",
    "time",
    "unknown",
]
SignalSourceStyle = Literal["moco_path", "flat_motion", "flat_analysis", "unknown"]
DefaultQuantity = Literal["value", "speed", "activation", "control", "unknown"]
MotionType = Literal["rotational", "translational", "other"]
SignalUnit = Literal[
    "seconds",
    "radians",
    "degrees",
    "rad/s",
    "deg/s",
    "meters",
    "m/s",
    "activation",
    "force",
    "moment",
    "fiber_length",
    "control",
    "unknown",
]


TRANSLATIONAL_ENTITY_NAMES = {
    "pelvis_tx",
    "pelvis_ty",
    "pelvis_tz",
    "sacrum_x",
    "sacrum_y",
    "sacrum_z",
}

FLAT_COORDINATE_FILE_HINTS = (
    "_ik.mot",
    "_ik.sto",
    "_kinematics_q.sto",
    "_kinematics_u.sto",
    "_q.sto",
    "_u.sto",
    "coordinates.mot",
)


@dataclass(frozen=True, slots=True)
class SignalDescriptor:
    raw_name: str
    source_style: SignalSourceStyle
    signal_type: SignalQuantity
    entity_name: str
    component_path: tuple[str, ...] = ()

    @property
    def canonical_key(self) -> str:
        if self.signal_type == "time" and self.entity_name == "time":
            return "time"
        return f"{self.entity_name}/{self.signal_type}"


@dataclass(frozen=True, slots=True)
class ResolvedSignal:
    descriptor: SignalDescriptor
    motion_type: MotionType
    native_unit: SignalUnit
    normalized_unit: SignalUnit

    @property
    def canonical_key(self) -> str:
        return self.descriptor.canonical_key


def infer_default_quantity(file_path: str | Path | None) -> DefaultQuantity:
    if file_path is None:
        return "value"

    name = Path(file_path).name.lower()
    if name.endswith("_activation.sto"):
        return "activation"
    if name.endswith("_controls.sto") or "control" in name:
        return "control"
    if name.endswith("_u.sto") or "speed" in name:
        return "speed"
    return "value"


def infer_coordinate_file(file_path: str | Path | None) -> bool:
    if file_path is None:
        return False

    name = Path(file_path).name.lower()
    return any(name.endswith(hint) for hint in FLAT_COORDINATE_FILE_HINTS)


def parse_signal_name(
    column_name: str,
    default_quantity: DefaultQuantity = "value",
) -> SignalDescriptor:
    stripped = column_name.strip()
    lowered = stripped.lower()

    if lowered == "time":
        return SignalDescriptor(
            raw_name=column_name,
            source_style="flat_motion",
            signal_type="time",
            entity_name="time",
        )

    if stripped.startswith("/"):
        parts = tuple(part for part in stripped.split("/") if part)
        if (
            len(parts) >= 4
            and parts[0] == "jointset"
            and parts[-1] in {"value", "speed"}
        ):
            return SignalDescriptor(
                raw_name=column_name,
                source_style="moco_path",
                signal_type=cast(SignalQuantity, parts[-1]),
                entity_name=parts[-2],
                component_path=parts,
            )

        if len(parts) >= 3 and parts[0] == "forceset":
            if parts[-1] in {"activation", "fiber_length"}:
                return SignalDescriptor(
                    raw_name=column_name,
                    source_style="moco_path",
                    signal_type=cast(SignalQuantity, parts[-1]),
                    entity_name=parts[-2],
                    component_path=parts,
                )

            return SignalDescriptor(
                raw_name=column_name,
                source_style="moco_path",
                signal_type="control",
                entity_name=parts[-1],
                component_path=parts,
            )

        return SignalDescriptor(
            raw_name=column_name,
            source_style="moco_path",
            signal_type="unknown",
            entity_name=parts[-1] if parts else stripped,
            component_path=parts,
        )

    if stripped.endswith("_moment"):
        return SignalDescriptor(
            raw_name=column_name,
            source_style="flat_analysis",
            signal_type="moment",
            entity_name=stripped.removesuffix("_moment"),
        )

    if stripped.endswith("_force"):
        return SignalDescriptor(
            raw_name=column_name,
            source_style="flat_analysis",
            signal_type="force",
            entity_name=stripped.removesuffix("_force"),
        )

    signal_type: SignalQuantity
    if default_quantity == "value":
        signal_type = "value"
    elif default_quantity == "speed":
        signal_type = "speed"
    elif default_quantity == "activation":
        signal_type = "activation"
    elif default_quantity == "control":
        signal_type = "control"
    else:
        signal_type = "unknown"

    source_style: SignalSourceStyle = (
        "flat_analysis" if signal_type in {"activation", "control"} else "flat_motion"
    )
    entity_name = stripped.removesuffix("_u") if signal_type == "speed" else stripped

    return SignalDescriptor(
        raw_name=column_name,
        source_style=source_style,
        signal_type=signal_type,
        entity_name=entity_name,
    )


def parse_signal_names(
    column_names: Sequence[str],
    default_quantity: DefaultQuantity = "value",
) -> list[SignalDescriptor]:
    return [
        parse_signal_name(name, default_quantity=default_quantity)
        for name in column_names
    ]


def infer_motion_type(descriptor: SignalDescriptor) -> MotionType:
    if descriptor.signal_type not in {"value", "speed"}:
        return "other"
    if descriptor.source_style == "moco_path" and descriptor.component_path[:1] == (
        "jointset",
    ):
        if descriptor.entity_name in TRANSLATIONAL_ENTITY_NAMES:
            return "translational"
        return "rotational"
    if descriptor.entity_name in TRANSLATIONAL_ENTITY_NAMES:
        return "translational"
    return "other"


def resolve_signal_units(
    descriptor: SignalDescriptor,
    in_degrees: Literal["yes", "no", ""] = "",
    assume_flat_coordinates: bool = False,
) -> ResolvedSignal:
    if descriptor.signal_type == "time":
        return ResolvedSignal(descriptor, "other", "seconds", "seconds")

    motion_type = infer_motion_type(descriptor)
    if descriptor.signal_type == "value":
        if motion_type == "other" and assume_flat_coordinates:
            motion_type = "rotational"
            if descriptor.entity_name in TRANSLATIONAL_ENTITY_NAMES:
                motion_type = "translational"
        if motion_type == "rotational":
            native_unit: SignalUnit = "degrees" if in_degrees == "yes" else "radians"
            return ResolvedSignal(descriptor, motion_type, native_unit, "radians")
        if motion_type == "translational":
            return ResolvedSignal(descriptor, motion_type, "meters", "meters")
    elif descriptor.signal_type == "speed":
        if motion_type == "other" and assume_flat_coordinates:
            motion_type = "rotational"
            if descriptor.entity_name in TRANSLATIONAL_ENTITY_NAMES:
                motion_type = "translational"
        if motion_type == "rotational":
            native_unit = "deg/s" if in_degrees == "yes" else "rad/s"
            return ResolvedSignal(descriptor, motion_type, native_unit, "rad/s")
        if motion_type == "translational":
            return ResolvedSignal(descriptor, motion_type, "m/s", "m/s")
    elif descriptor.signal_type == "activation":
        return ResolvedSignal(descriptor, motion_type, "activation", "activation")
    elif descriptor.signal_type == "fiber_length":
        return ResolvedSignal(descriptor, motion_type, "fiber_length", "fiber_length")
    elif descriptor.signal_type == "force":
        return ResolvedSignal(descriptor, motion_type, "force", "force")
    elif descriptor.signal_type == "moment":
        return ResolvedSignal(descriptor, motion_type, "moment", "moment")
    elif descriptor.signal_type == "control":
        return ResolvedSignal(descriptor, motion_type, "control", "control")

    return ResolvedSignal(descriptor, motion_type, "unknown", "unknown")


def normalize_signal_values(
    values: Sequence[float | None],
    resolved_signal: ResolvedSignal,
) -> list[float | None]:
    if resolved_signal.native_unit == resolved_signal.normalized_unit:
        return list(values)

    if (
        resolved_signal.native_unit == "degrees"
        and resolved_signal.normalized_unit == "radians"
    ):
        return [None if value is None else math.radians(value) for value in values]

    if (
        resolved_signal.native_unit == "deg/s"
        and resolved_signal.normalized_unit == "rad/s"
    ):
        return [None if value is None else math.radians(value) for value in values]

    return list(values)


def build_signal_map(
    column_names: Sequence[str],
    default_quantity: DefaultQuantity = "value",
) -> dict[str, list[SignalDescriptor]]:
    signal_map: dict[str, list[SignalDescriptor]] = {}
    for descriptor in parse_signal_names(
        column_names, default_quantity=default_quantity
    ):
        signal_map.setdefault(descriptor.canonical_key, []).append(descriptor)
    return signal_map


def common_signal_keys(
    column_groups: Sequence[Sequence[str]],
    default_quantities: Sequence[DefaultQuantity] | None = None,
) -> list[str]:
    if not column_groups:
        return []

    quantities: Sequence[DefaultQuantity]
    if default_quantities is None:
        quantities = [cast(DefaultQuantity, "value") for _ in column_groups]
    else:
        quantities = default_quantities

    common_keys = set(build_signal_map(column_groups[0], quantities[0]).keys())
    for columns, default_quantity in zip(column_groups[1:], quantities[1:]):
        common_keys &= set(build_signal_map(columns, default_quantity).keys())
    return sorted(common_keys)


def build_signal_match_index(
    column_names: Sequence[str],
    default_quantity: DefaultQuantity = "value",
) -> dict[str, str]:
    signal_index: dict[str, str] = {}
    for descriptor in parse_signal_names(
        column_names, default_quantity=default_quantity
    ):
        signal_index.setdefault(descriptor.canonical_key, descriptor.raw_name)
    return signal_index


def build_resolved_signal_index(
    column_names: Sequence[str],
    default_quantity: DefaultQuantity = "value",
    in_degrees: Literal["yes", "no", ""] = "",
    assume_flat_coordinates: bool = False,
) -> dict[str, ResolvedSignal]:
    resolved_index: dict[str, ResolvedSignal] = {}
    for descriptor in parse_signal_names(
        column_names, default_quantity=default_quantity
    ):
        resolved_index.setdefault(
            descriptor.canonical_key,
            resolve_signal_units(
                descriptor,
                in_degrees=in_degrees,
                assume_flat_coordinates=assume_flat_coordinates,
            ),
        )
    return resolved_index


def iter_signal_rows(
    column_names: Iterable[str],
    default_quantity: DefaultQuantity = "value",
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for descriptor in parse_signal_names(
        list(column_names), default_quantity=default_quantity
    ):
        rows.append(
            {
                "raw_name": descriptor.raw_name,
                "canonical_key": descriptor.canonical_key,
                "signal_type": descriptor.signal_type,
                "source_style": descriptor.source_style,
                "entity_name": descriptor.entity_name,
            }
        )
    return rows
