import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
from pathlib import Path

from pydantic import BaseModel, Field, FilePath


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TOOL_MODULE_PATH = SRC_ROOT / "osimpy" / "tools" / "tool.py"
CLI_MODULE_PATH = SRC_ROOT / "osimpy" / "cli.py"


def load_tool_module():
    pydantic_settings_pkg = types.ModuleType("pydantic_settings")

    class CliApp:
        @staticmethod
        def run(cls, cli_args=None):
            if cli_args is None:
                cli_args = []
            parsed = {}
            index = 0
            while index < len(cli_args):
                key = cli_args[index]
                if not key.startswith("--"):
                    index += 1
                    continue
                parsed[key[2:].replace("-", "_")] = cli_args[index + 1]
                index += 2
            instance = cls(**parsed)
            instance.cli_cmd()
            return instance

    setattr(pydantic_settings_pkg, "CliApp", CliApp)
    sys.modules["pydantic_settings"] = pydantic_settings_pkg

    osimpy_pkg = types.ModuleType("osimpy")
    osimpy_pkg.__path__ = [str(SRC_ROOT / "osimpy")]
    sys.modules["osimpy"] = osimpy_pkg

    tools_pkg = types.ModuleType("osimpy.tools")
    tools_pkg.__path__ = [str(SRC_ROOT / "osimpy" / "tools")]
    sys.modules["osimpy.tools"] = tools_pkg

    io_pkg = types.ModuleType("osimpy.io")
    io_pkg.__path__ = [str(SRC_ROOT / "osimpy" / "io")]

    class STOMetadata(BaseModel):
        pass

    setattr(io_pkg, "STOMetadata", STOMetadata)
    sys.modules["osimpy.io"] = io_pkg

    spec = importlib.util.spec_from_file_location("osimpy.tools.tool", TOOL_MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["osimpy.tools.tool"] = module
    spec.loader.exec_module(module)
    return module


def test_make_cli_smoke():
    module = load_tool_module()
    ToolResult = module.ToolResult
    ToolSettings = module.ToolSettings

    class DummyResult(ToolResult):
        marker_file: FilePath | None = Field(None, description="Dummy output file")

    class DummyTool:
        def __init__(self, marker_name: str):
            self.marker_name = marker_name

        def printToXML(self, path: str) -> None:
            Path(path).write_text("<dummy />", encoding="utf-8")

        def run(self) -> None:
            Path(self.marker_name).write_text("done", encoding="utf-8")

    class DummySettings(ToolSettings[DummyResult]):
        marker_output: str = Field(description="Marker output file name")

        def create_tool(self):
            return DummyTool(self.marker_output)

        def resolve_output_files(self):
            return {"marker_file": self._resolve_output(self.marker_output)}

    with tempfile.TemporaryDirectory(prefix="osimpy-cli-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        model_path = tmp_path / "model.osim"
        model_path.write_text("model", encoding="utf-8")
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        cli = DummySettings.make_cli()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = cli(
                [
                    "--name",
                    "demo",
                    "--model-path",
                    str(model_path),
                    "--results-directory",
                    str(results_dir),
                    "--marker-output",
                    "marker.txt",
                ]
            )

        payload = stdout.getvalue().strip()
        parsed = json.loads(payload)

        assert result.success is True
        assert result.marker_file is not None
        assert Path(result.marker_file).name == "marker.txt"
        assert Path(parsed["setup_file"]).exists()
        assert Path(parsed["marker_file"]).exists()


def test_top_level_cli_registers_moco_subcommands():
    pydantic_settings_pkg = types.ModuleType("pydantic_settings")

    class CliApp:
        @staticmethod
        def run_subcommand(instance):
            return instance

    class CliSubCommand:
        def __class_getitem__(cls, item):
            return item

    setattr(pydantic_settings_pkg, "CliApp", CliApp)
    setattr(pydantic_settings_pkg, "CliSubCommand", CliSubCommand)
    sys.modules["pydantic_settings"] = pydantic_settings_pkg

    osimpy_pkg = types.ModuleType("osimpy")
    osimpy_pkg.__path__ = [str(SRC_ROOT / "osimpy")]
    sys.modules["osimpy"] = osimpy_pkg

    tools_pkg = types.ModuleType("osimpy.tools")
    tools_pkg.__path__ = [str(SRC_ROOT / "osimpy" / "tools")]
    sys.modules["osimpy.tools"] = tools_pkg

    moco_pkg = types.ModuleType("osimpy.moco")
    moco_pkg.__path__ = [str(SRC_ROOT / "osimpy" / "moco")]
    sys.modules["osimpy.moco"] = moco_pkg

    class DummySettings(BaseModel):
        def cli_cmd(self) -> None:
            return None

    for module_name, attr_name in [
        ("osimpy.tools.scale", "ScaleSettings"),
        ("osimpy.tools.ik", "IKSettings"),
        ("osimpy.tools.id", "IDSettings"),
        ("osimpy.tools.cmc", "CMCSettings"),
        ("osimpy.moco.inverse", "MocoInverseSettings"),
        ("osimpy.moco.track", "MocoTrackSettings"),
    ]:
        module = types.ModuleType(module_name)
        setattr(module, attr_name, DummySettings)
        sys.modules[module_name] = module

    spec = importlib.util.spec_from_file_location("osimpy.cli", CLI_MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["osimpy.cli"] = module
    spec.loader.exec_module(module)

    model_fields = module.OsimPyCli.model_fields
    assert model_fields["moco_inverse"].alias == "moco-inverse"
    assert model_fields["moco_track"].alias == "moco-track"
