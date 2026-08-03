import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SIGNALS_MODULE_PATH = REPO_ROOT / "src" / "osimpy" / "signals.py"


def load_signals_module():
    spec = importlib.util.spec_from_file_location("osimpy_signals", SIGNALS_MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["osimpy_signals"] = module
    spec.loader.exec_module(module)
    return module


def test_parse_moco_joint_value_matches_flat_motion_name():
    module = load_signals_module()

    moco_signal = module.parse_signal_name("/jointset/hip_r/hip_r_flx/value")
    flat_signal = module.parse_signal_name("hip_r_flx")

    assert moco_signal.entity_name == "hip_r_flx"
    assert moco_signal.signal_type == "value"
    assert moco_signal.canonical_key == flat_signal.canonical_key


def test_parse_moco_joint_speed_differs_from_value():
    module = load_signals_module()

    speed_signal = module.parse_signal_name("/jointset/hip_r/hip_r_flx/speed")
    value_signal = module.parse_signal_name("/jointset/hip_r/hip_r_flx/value")

    assert speed_signal.signal_type == "speed"
    assert speed_signal.canonical_key != value_signal.canonical_key


def test_parse_activation_matches_flat_activation_with_context():
    module = load_signals_module()

    moco_signal = module.parse_signal_name("/forceset/R_BFa/activation")
    flat_signal = module.parse_signal_name("R_BFa", default_quantity="activation")

    assert moco_signal.signal_type == "activation"
    assert moco_signal.canonical_key == flat_signal.canonical_key


def test_common_signal_keys_respects_default_quantities():
    module = load_signals_module()

    common_keys = module.common_signal_keys(
        [
            [
                "time",
                "/jointset/hip_r/hip_r_flx/value",
                "/jointset/hip_r/hip_r_flx/speed",
            ],
            ["time", "hip_r_flx"],
        ],
        default_quantities=["value", "value"],
    )

    assert common_keys == ["hip_r_flx/value", "time"]


def test_parse_flat_speed_matches_moco_speed():
    module = load_signals_module()

    flat_speed = module.parse_signal_name("hip_r_flx_u", default_quantity="speed")
    moco_speed = module.parse_signal_name("/jointset/hip_r/hip_r_flx/speed")

    assert flat_speed.signal_type == "speed"
    assert flat_speed.entity_name == "hip_r_flx"
    assert flat_speed.canonical_key == moco_speed.canonical_key


def test_resolve_rotational_value_units_from_in_degrees():
    module = load_signals_module()

    descriptor = module.parse_signal_name("hip_r_flx")
    resolved = module.resolve_signal_units(
        descriptor,
        in_degrees="yes",
        assume_flat_coordinates=True,
    )

    assert resolved.motion_type == "rotational"
    assert resolved.native_unit == "degrees"
    assert resolved.normalized_unit == "radians"


def test_resolve_translational_value_units_ignore_in_degrees():
    module = load_signals_module()

    descriptor = module.parse_signal_name("sacrum_x")
    resolved = module.resolve_signal_units(descriptor, in_degrees="yes")

    assert resolved.motion_type == "translational"
    assert resolved.native_unit == "meters"
    assert resolved.normalized_unit == "meters"


def test_resolve_unknown_flat_signal_stays_unconverted_by_default():
    module = load_signals_module()

    descriptor = module.parse_signal_name("mystery_signal")
    resolved = module.resolve_signal_units(descriptor, in_degrees="yes")

    assert resolved.motion_type == "other"
    assert resolved.native_unit == "unknown"
    assert resolved.normalized_unit == "unknown"


def test_resolve_flat_coordinate_can_opt_in_to_conversion():
    module = load_signals_module()

    descriptor = module.parse_signal_name("hip_r_flx")
    resolved = module.resolve_signal_units(
        descriptor,
        in_degrees="yes",
        assume_flat_coordinates=True,
    )

    assert resolved.motion_type == "rotational"
    assert resolved.native_unit == "degrees"
    assert resolved.normalized_unit == "radians"


def test_normalize_rotational_speed_values_to_radians_per_second():
    module = load_signals_module()

    descriptor = module.parse_signal_name("hip_r_flx_u", default_quantity="speed")
    resolved = module.resolve_signal_units(
        descriptor,
        in_degrees="yes",
        assume_flat_coordinates=True,
    )
    normalized = module.normalize_signal_values([180.0, 90.0], resolved)

    assert normalized == [3.141592653589793, 1.5707963267948966]


def test_build_signal_match_index_preserves_first_raw_column():
    module = load_signals_module()

    match_index = module.build_signal_match_index(
        ["time", "/jointset/hip_r/hip_r_flx/value", "/jointset/hip_r/hip_r_flx/speed"]
    )

    assert match_index["time"] == "time"
    assert match_index["hip_r_flx/value"] == "/jointset/hip_r/hip_r_flx/value"
    assert match_index["hip_r_flx/speed"] == "/jointset/hip_r/hip_r_flx/speed"
