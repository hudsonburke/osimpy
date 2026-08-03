import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
MODULE_PATH = SRC_ROOT / "osimpy" / "moco" / "model_processing.py"


class FakeModOp:
    def __init__(self, *args):
        self.args = args


class FakeCoordinate:
    def __init__(self, name):
        self._name = name

    def getName(self):
        return self._name


class FakeCoordinateActuator:
    def __init__(self, coordinate_name):
        self.coordinate = FakeCoordinate(coordinate_name)
        self.optimal_force = None

    def getCoordinate(self):
        return self.coordinate

    def setOptimalForce(self, value):
        self.optimal_force = value

    @staticmethod
    def safeDownCast(obj):
        return obj if isinstance(obj, FakeCoordinateActuator) else None


class FakeMuscle:
    def __init__(self, name):
        self._name = name

    def getName(self):
        return self._name


class FakeDGFMuscle(FakeMuscle):
    def __init__(self, name):
        super().__init__(name)
        self.fiber_damping = None
        self.ignore_tendon_compliance = False

    def set_fiber_damping(self, value):
        self.fiber_damping = value

    def set_ignore_tendon_compliance(self, value):
        self.ignore_tendon_compliance = value


class FakeCollection:
    def __init__(self, items):
        self._items = items

    def getSize(self):
        return len(self._items)

    def get(self, index):
        return self._items[index]


class FakeModel:
    def __init__(self):
        self.muscles = FakeCollection([FakeDGFMuscle("R_GS"), FakeDGFMuscle("L_GS")])
        self.forces = FakeCollection([
            FakeCoordinateActuator("sacrum_y"),
            FakeCoordinateActuator("hip_r_flx"),
        ])
        self.printed_to = None

    def initSystem(self):
        return None

    def updMuscles(self):
        return self.muscles

    def updForceSet(self):
        return self.forces

    def printToXML(self, path):
        self.printed_to = path
        Path(path).write_text("<OpenSimDocument><datafile>forces.mot</datafile><data_source_name>forces</data_source_name></OpenSimDocument>", encoding="utf-8")


class FakeModelProcessor:
    instances = []

    def __init__(self, path):
        self.path = path
        self.ops = []
        self.model = FakeModel()
        FakeModelProcessor.instances.append(self)

    def append(self, op):
        self.ops.append(op)

    def process(self):
        return self.model


def load_module():
    opensim = types.ModuleType("opensim")
    opensim.ModelProcessor = FakeModelProcessor
    opensim.ModOpAddExternalLoads = type("ModOpAddExternalLoads", (FakeModOp,), {})
    opensim.ModOpReplaceMusclesWithDeGrooteFregly2016 = type(
        "ModOpReplaceMusclesWithDeGrooteFregly2016", (FakeModOp,), {}
    )
    opensim.ModOpUseImplicitTendonComplianceDynamicsDGF = type(
        "ModOpUseImplicitTendonComplianceDynamicsDGF", (FakeModOp,), {}
    )
    opensim.ModOpIgnoreTendonCompliance = type("ModOpIgnoreTendonCompliance", (FakeModOp,), {})
    opensim.ModOpIgnorePassiveFiberForcesDGF = type(
        "ModOpIgnorePassiveFiberForcesDGF", (FakeModOp,), {}
    )
    opensim.ModOpScaleActiveFiberForceCurveWidthDGF = type(
        "ModOpScaleActiveFiberForceCurveWidthDGF", (FakeModOp,), {}
    )
    opensim.ModOpReplacePathsWithFunctionBasedPaths = type(
        "ModOpReplacePathsWithFunctionBasedPaths", (FakeModOp,), {}
    )
    opensim.ModOpAddReserves = type("ModOpAddReserves", (FakeModOp,), {})
    opensim.CoordinateActuator = FakeCoordinateActuator
    opensim.DeGrooteFregly2016Muscle = types.SimpleNamespace(
        safeDownCast=lambda obj: obj if isinstance(obj, FakeDGFMuscle) else None
    )
    sys.modules["opensim"] = opensim

    spec = importlib.util.spec_from_file_location("osimpy.moco.model_processing", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["osimpy.moco.model_processing"] = module
    spec.loader.exec_module(module)
    return module


def test_build_model_processor_uses_default_ignore_tendon_compliance(tmp_path):
    FakeModelProcessor.instances.clear()
    module = load_module()

    processor = module.build_moco_model_processor(
        model_path=tmp_path / "model.osim",
        external_loads_path=None,
        replace_muscles_with_dgf=True,
        ignore_tendon_compliance=True,
        ignore_passive_fiber_forces=True,
        active_fiber_force_scale_width=1.5,
        reserve_optimal_force=0.1,
        muscle_path_set_file=None,
        rigid_tendon_muscle_names=[],
        dgf_fiber_damping=None,
        coordinate_reserve_optimal_force_overrides=[],
        results_directory=tmp_path,
        temp_model_stem="demo",
    )

    assert processor is FakeModelProcessor.instances[0]
    assert any(op.__class__.__name__ == "ModOpIgnoreTendonCompliance" for op in processor.ops)


def test_build_model_processor_materializes_selected_rigid_tendons_and_overrides(tmp_path):
    FakeModelProcessor.instances.clear()
    module = load_module()
    external_loads = tmp_path / "loads.xml"
    external_loads.write_text("<OpenSimDocument><datafile>forces.mot</datafile></OpenSimDocument>", encoding="utf-8")

    processor = module.build_moco_model_processor(
        model_path=tmp_path / "model.osim",
        external_loads_path=external_loads,
        replace_muscles_with_dgf=True,
        ignore_tendon_compliance=False,
        ignore_passive_fiber_forces=True,
        active_fiber_force_scale_width=1.5,
        reserve_optimal_force=0.1,
        muscle_path_set_file=None,
        rigid_tendon_muscle_names=["R_GS"],
        dgf_fiber_damping=0.01,
        coordinate_reserve_optimal_force_overrides=[
            module.CoordinateReserveOptimalForceOverride(
                coordinate="sacrum_y", optimal_force=3.0
            )
        ],
        results_directory=tmp_path,
        temp_model_stem="demo",
    )

    runtime_model = FakeModelProcessor.instances[0].model
    rigid_target = runtime_model.updMuscles().get(0)
    reserve = runtime_model.updForceSet().get(0)

    assert rigid_target.ignore_tendon_compliance is True
    assert rigid_target.fiber_damping == 0.01
    assert reserve.optimal_force == 3.0
    assert processor is not FakeModelProcessor.instances[0]
    assert Path(processor.path).name.startswith("demo_")
    assert Path(processor.path).name.endswith("_runtime_model.osim")


def test_build_model_processor_rejects_unknown_rigid_tendon_name(tmp_path):
    module = load_module()

    try:
        module.build_moco_model_processor(
            model_path=tmp_path / "model.osim",
            external_loads_path=None,
            replace_muscles_with_dgf=True,
            ignore_tendon_compliance=False,
            ignore_passive_fiber_forces=True,
            active_fiber_force_scale_width=1.5,
            reserve_optimal_force=0.1,
            muscle_path_set_file=None,
            rigid_tendon_muscle_names=["BAD_NAME"],
            dgf_fiber_damping=None,
            coordinate_reserve_optimal_force_overrides=[],
            results_directory=tmp_path,
            temp_model_stem="demo",
        )
    except ValueError as exc:
        assert "BAD_NAME" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown rigid tendon name")


def test_build_model_processor_rejects_duplicate_coordinate_overrides(tmp_path):
    module = load_module()

    try:
        module.build_moco_model_processor(
            model_path=tmp_path / "model.osim",
            external_loads_path=None,
            replace_muscles_with_dgf=True,
            ignore_tendon_compliance=False,
            ignore_passive_fiber_forces=True,
            active_fiber_force_scale_width=1.5,
            reserve_optimal_force=0.1,
            muscle_path_set_file=None,
            rigid_tendon_muscle_names=[],
            dgf_fiber_damping=None,
            coordinate_reserve_optimal_force_overrides=[
                module.CoordinateReserveOptimalForceOverride(
                    coordinate="sacrum_y", optimal_force=3.0
                ),
                module.CoordinateReserveOptimalForceOverride(
                    coordinate="sacrum_y", optimal_force=4.0
                ),
            ],
            results_directory=tmp_path,
            temp_model_stem="demo",
        )
    except ValueError as exc:
        assert "Duplicate coordinate names" in str(exc)
    else:
        raise AssertionError("Expected ValueError for duplicate coordinate overrides")
