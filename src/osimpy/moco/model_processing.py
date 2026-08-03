from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import opensim as osim
from pydantic import BaseModel


class CoordinateReserveOptimalForceOverride(BaseModel):
    coordinate: str
    optimal_force: float


def validate_requested_muscle_names(requested_names: set[str], muscles) -> None:
    if not requested_names:
        return
    available_names = {muscles.get(index).getName() for index in range(muscles.getSize())}
    unknown = sorted(requested_names - available_names)
    if unknown:
        raise ValueError(
            "Unknown muscle names in rigid_tendon_muscle_names: " + ", ".join(unknown)
        )


def validate_coordinate_force_overrides(overrides, forces) -> None:
    if not overrides:
        return

    duplicate_coordinates = sorted(
        {
            item.coordinate
            for item in overrides
            if sum(1 for other in overrides if other.coordinate == item.coordinate) > 1
        }
    )
    if duplicate_coordinates:
        raise ValueError(
            "Duplicate coordinate names in coordinate_reserve_optimal_force_overrides: "
            + ", ".join(duplicate_coordinates)
        )

    available_coordinates = set()
    for index in range(forces.getSize()):
        actuator = osim.CoordinateActuator.safeDownCast(forces.get(index))
        if actuator is None:
            continue
        available_coordinates.add(actuator.getCoordinate().getName())

    unknown = sorted({item.coordinate for item in overrides} - available_coordinates)
    if unknown:
        raise ValueError(
            "Unknown coordinate names in coordinate_reserve_optimal_force_overrides: "
            + ", ".join(unknown)
        )


def rewrite_external_load_paths(model_file: Path, external_loads_path: Path | None) -> None:
    if external_loads_path is None or not external_loads_path.exists():
        return

    external_loads_text = external_loads_path.read_text()
    match = re.search(r"<datafile>(.*?)</datafile>", external_loads_text)
    if match is None:
        return

    datafile = match.group(1).strip()
    absolute_datafile = (external_loads_path.parent / datafile).resolve()

    model_text = model_file.read_text()
    model_text = re.sub(r"<datafile>.*?</datafile>", f"<datafile>{absolute_datafile}</datafile>", model_text)
    data_source_name = absolute_datafile.as_posix()
    model_text = re.sub(
        r"<data_source_name>.*?</data_source_name>",
        f"<data_source_name>{data_source_name}</data_source_name>",
        model_text,
    )
    model_file.write_text(model_text)


def build_runtime_model_path(results_directory: Path, temp_model_stem: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    return results_directory / f"{temp_model_stem}_{timestamp}_runtime_model.osim"


def build_moco_model_processor(
    *,
    model_path: Path,
    external_loads_path: Path | None,
    replace_muscles_with_dgf: bool,
    ignore_tendon_compliance: bool,
    ignore_passive_fiber_forces: bool,
    active_fiber_force_scale_width: float,
    reserve_optimal_force: float,
    muscle_path_set_file: Path | None,
    rigid_tendon_muscle_names: list[str],
    dgf_fiber_damping: float | None,
    coordinate_reserve_optimal_force_overrides: list[CoordinateReserveOptimalForceOverride],
    results_directory: Path,
    temp_model_stem: str,
) -> osim.ModelProcessor:
    selected_rigid_names = {name for name in rigid_tendon_muscle_names if name}
    requires_materialized_model = bool(
        selected_rigid_names
        or dgf_fiber_damping is not None
        or coordinate_reserve_optimal_force_overrides
    )

    if selected_rigid_names and ignore_tendon_compliance:
        raise ValueError(
            "ignore_tendon_compliance=True cannot be combined with rigid_tendon_muscle_names. "
            "Set ignore_tendon_compliance=False to use mixed compliant/rigid tendons."
        )

    if (selected_rigid_names or dgf_fiber_damping is not None) and not replace_muscles_with_dgf:
        raise ValueError(
            "rigid_tendon_muscle_names and dgf_fiber_damping require replace_muscles_with_dgf=True"
        )

    mp = osim.ModelProcessor(str(model_path))
    if external_loads_path is not None:
        mp.append(osim.ModOpAddExternalLoads(str(external_loads_path)))
    if replace_muscles_with_dgf:
        mp.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    if selected_rigid_names:
        mp.append(osim.ModOpUseImplicitTendonComplianceDynamicsDGF())
    elif ignore_tendon_compliance:
        mp.append(osim.ModOpIgnoreTendonCompliance())
    if ignore_passive_fiber_forces:
        mp.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    if active_fiber_force_scale_width != 1.0:
        mp.append(
            osim.ModOpScaleActiveFiberForceCurveWidthDGF(active_fiber_force_scale_width)
        )
    if muscle_path_set_file is not None:
        mp.append(osim.ModOpReplacePathsWithFunctionBasedPaths(str(muscle_path_set_file)))
    mp.append(osim.ModOpAddReserves(reserve_optimal_force))

    if not requires_materialized_model:
        return mp

    runtime_model = mp.process()
    runtime_model.initSystem()
    muscles = runtime_model.updMuscles()
    forces = runtime_model.updForceSet()

    validate_requested_muscle_names(selected_rigid_names, muscles)
    validate_coordinate_force_overrides(coordinate_reserve_optimal_force_overrides, forces)

    for index in range(muscles.getSize()):
        muscle = muscles.get(index)
        dgf = osim.DeGrooteFregly2016Muscle.safeDownCast(muscle)
        if dgf is None:
            continue
        if dgf_fiber_damping is not None:
            dgf.set_fiber_damping(dgf_fiber_damping)
        if muscle.getName() in selected_rigid_names:
            dgf.set_ignore_tendon_compliance(True)

    override_map = {
        item.coordinate: item.optimal_force
        for item in coordinate_reserve_optimal_force_overrides
    }
    for index in range(forces.getSize()):
        actuator = osim.CoordinateActuator.safeDownCast(forces.get(index))
        if actuator is None:
            continue
        coordinate_name = actuator.getCoordinate().getName()
        override_force = override_map.get(coordinate_name)
        if override_force is not None:
            actuator.setOptimalForce(override_force)

    results_directory.mkdir(parents=True, exist_ok=True)
    temp_model_path = build_runtime_model_path(results_directory, temp_model_stem)
    runtime_model.printToXML(str(temp_model_path))
    rewrite_external_load_paths(temp_model_path, external_loads_path)
    return osim.ModelProcessor(str(temp_model_path))
