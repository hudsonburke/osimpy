import importlib.util
import json
import io
import sys
import types
import contextlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TRACK_MODULE_PATH = SRC_ROOT / "osimpy" / "moco" / "track.py"


def load_track_module():
    opensim = types.ModuleType("opensim")
    sys.modules["opensim"] = opensim

    osimpy_pkg = types.ModuleType("osimpy")
    osimpy_pkg.__path__ = [str(SRC_ROOT / "osimpy")]
    sys.modules["osimpy"] = osimpy_pkg

    moco_pkg = types.ModuleType("osimpy.moco")
    moco_pkg.__path__ = [str(SRC_ROOT / "osimpy" / "moco")]
    sys.modules["osimpy.moco"] = moco_pkg

    io_pkg = types.ModuleType("osimpy.io")
    io_pkg.STOMetadata = object
    sys.modules["osimpy.io"] = io_pkg

    sto_pkg = types.ModuleType("osimpy.io.sto")
    sto_pkg.sto_to_df = lambda path: (path, None)
    sys.modules["osimpy.io.sto"] = sto_pkg

    def build_moco_model_processor(**kwargs):
        build_moco_model_processor.last_kwargs = kwargs
        return "processor"

    helper_spec = importlib.util.spec_from_file_location(
        "osimpy.moco.model_processing", SRC_ROOT / "osimpy" / "moco" / "model_processing.py"
    )
    assert helper_spec is not None and helper_spec.loader is not None
    helper_module = importlib.util.module_from_spec(helper_spec)
    sys.modules["osimpy.moco.model_processing"] = helper_module
    helper_spec.loader.exec_module(helper_module)
    helper_module.build_moco_model_processor = build_moco_model_processor

    spec = importlib.util.spec_from_file_location("osimpy.moco.track", TRACK_MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["osimpy.moco.track"] = module
    spec.loader.exec_module(module)
    return module, build_moco_model_processor


def test_track_build_model_processor_forwards_new_knobs_and_preserves_default_reserve(tmp_path):
    module, helper = load_track_module()

    model_path = tmp_path / "model.osim"
    coords_path = tmp_path / "coords.mot"
    loads_path = tmp_path / "loads.xml"
    model_path.write_text("model", encoding="utf-8")
    coords_path.write_text("coords", encoding="utf-8")
    loads_path.write_text("loads", encoding="utf-8")

    settings = module.MocoTrackSettings(
        model_path=model_path,
        coordinates_path=coords_path,
        external_loads_path=loads_path,
        rigid_tendon_muscle_names=["L_GS", "R_GS"],
        dgf_fiber_damping=0.01,
        coordinate_reserve_optimal_force_overrides=[
            module.CoordinateReserveOptimalForceOverride(
                coordinate="hip_r_flx", optimal_force=0.5
            )
        ],
    )

    assert settings._build_model_processor() == "processor"
    assert helper.last_kwargs["rigid_tendon_muscle_names"] == ["L_GS", "R_GS"]
    assert helper.last_kwargs["dgf_fiber_damping"] == 0.01
    assert helper.last_kwargs["reserve_optimal_force"] == 0.1


def test_track_cli_cmd_prints_result_json(tmp_path):
    module, _ = load_track_module()

    model_path = tmp_path / "model.osim"
    coords_path = tmp_path / "coords.mot"
    model_path.write_text("model", encoding="utf-8")
    coords_path.write_text("coords", encoding="utf-8")

    settings = module.MocoTrackSettings(
        model_path=model_path,
        coordinates_path=coords_path,
        results_directory=tmp_path,
    )

    original_run = module.MocoTrackSettings.run
    module.MocoTrackSettings.run = lambda self: module.MocoTrackResult(
        success=True,
        results_directory=tmp_path,
        solution_file=coords_path,
    )

    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            settings.cli_cmd()
    finally:
        module.MocoTrackSettings.run = original_run

    payload = json.loads(stdout.getvalue())
    assert payload["success"] is True
    assert payload["solution_file"] == str(coords_path)
