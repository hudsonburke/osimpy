import importlib.util
from pathlib import Path

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
STO_MODULE_PATH = REPO_ROOT / "src" / "osimpy" / "io" / "sto.py"


def load_sto_module():
    spec = importlib.util.spec_from_file_location("osimpy_io_sto", STO_MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_sto(tmp_path: Path, name: str, contents: str) -> Path:
    path = tmp_path / name
    path.write_text(contents, encoding="utf-8")
    return path


def test_sto_to_df_reads_conventional_sto_metadata(tmp_path):
    module = load_sto_module()
    path = write_sto(
        tmp_path,
        "classic.sto",
        """Inverse Dynamics Generalized Forces
version=1
nRows=2
nColumns=3
inDegrees=no
endheader
time\thip_flexion\tknee_angle
0.0\t1.0\t2.0
0.1\t3.0\t4.0
""",
    )

    df, metadata = module.sto_to_df(path)

    assert isinstance(metadata, module.STOMetadata)
    assert metadata.name == "Inverse Dynamics Generalized Forces"
    assert metadata.version == 1
    assert metadata.nRows == 2
    assert metadata.nColumns == 3
    assert metadata.inDegrees == "no"
    assert metadata.extra == {}
    assert df.columns == ["time", "hip_flexion", "knee_angle"]
    assert df.shape == (2, 3)


def test_sto_to_df_reads_moco_metadata_into_extra(tmp_path):
    module = load_sto_module()
    path = write_sto(
        tmp_path,
        "moco_solution.sto",
        """inDegrees=no
num_controls=2
num_derivatives=0
num_input_controls=0
num_iterations=611
num_multipliers=0
num_parameters=0
num_slacks=0
num_states=3
objective=0.009539
solver_duration=4787.070201
status=Solve_Succeeded
success=true
DataType=double
version=3
OpenSimVersion=4.5.2
endheader
time\t/state_1\t/state_2\t/state_3\t/control_1\t/control_2
0.0\t1\t2\t3\t4\t5
0.1\t6\t7\t8\t9\t10
""",
    )

    df, metadata = module.sto_to_df(path)

    assert isinstance(metadata, module.STOMetadata)
    assert metadata.name == ""
    assert metadata.version == 3
    assert metadata.nRows == 2
    assert metadata.nColumns == 6
    assert metadata.inDegrees == "no"
    assert metadata.extra["num_states"] == "3"
    assert metadata.extra["num_controls"] == "2"
    assert metadata.extra["status"] == "Solve_Succeeded"
    assert metadata.extra["success"] == "true"
    assert metadata.extra["OpenSimVersion"] == "4.5.2"
    assert df.columns == [
        "time",
        "/state_1",
        "/state_2",
        "/state_3",
        "/control_1",
        "/control_2",
    ]
    assert df.shape == (2, 6)


def test_df_to_sto_roundtrip_preserves_extra_metadata(tmp_path):
    module = load_sto_module()
    df = pl.DataFrame({"time": [0.0, 0.1], "value": [1.0, 2.0]})
    metadata = module.STOMetadata(
        version=3,
        inDegrees="no",
        extra={"num_states": "1", "status": "Solve_Succeeded"},
    )

    output_path = tmp_path / "roundtrip.sto"
    module.df_to_sto(output_path, df, metadata=metadata)

    written = output_path.read_text(encoding="utf-8")
    assert "version=3" in written
    assert "nRows=2" in written
    assert "nColumns=2" in written
    assert "num_states=1" in written
    assert "status=Solve_Succeeded" in written
    assert "nRows=None" not in written
    assert "nColumns=None" not in written

    roundtrip_df, roundtrip_metadata = module.sto_to_df(output_path)
    assert roundtrip_df.shape == (2, 2)
    assert roundtrip_metadata.nRows == 2
    assert roundtrip_metadata.nColumns == 2
    assert roundtrip_metadata.extra["num_states"] == "1"
    assert roundtrip_metadata.extra["status"] == "Solve_Succeeded"


def test_sto_to_df_preserves_legitimate_column_prefixes(tmp_path):
    module = load_sto_module()
    path = write_sto(
        tmp_path,
        "column_prefix.sto",
        """version=1
endheader
time\tcolumn_1\tcolumn_2
0.0\t1.0\t2.0
0.1\t3.0\t4.0
""",
    )

    df, metadata = module.sto_to_df(path)

    assert df.columns == ["time", "column_1", "column_2"]
    assert df.shape == (2, 3)
    assert metadata.nColumns == 3
