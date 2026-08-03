from dataclasses import dataclass
import opensim as osim


@dataclass
class OpenSimExternalForce:
    name: str
    applied_to_body: str
    force_expressed_in_body: str = "ground"
    point_expressed_in_body: str = "ground"
    force_identifier: str = r"force_v"
    point_identifier: str = r"force_p"
    torque_identifier: str = r"moment_"
    data_source_name: str | None = None


def force_to_opensim(force: OpenSimExternalForce) -> osim.ExternalForce:
    """
    Convert to OpenSim ExternalForce object.
    """
    ext_force = osim.ExternalForce()
    ext_force.setName(force.name)
    ext_force.setAppliedToBodyName(force.applied_to_body)
    ext_force.setForceExpressedInBodyName(force.force_expressed_in_body)
    ext_force.setPointExpressedInBodyName(force.point_expressed_in_body)
    ext_force.setForceIdentifier(force.force_identifier)
    ext_force.setPointIdentifier(force.point_identifier)
    ext_force.setTorqueIdentifier(force.torque_identifier)

    if force.data_source_name is not None:
        ext_force.set_data_source_name(force.data_source_name)

    return ext_force


def export_external_loads(
    filepath: str,
    external_forces: list[OpenSimExternalForce],
    datafile_name: str | None = None,
) -> None:
    """
    Export external loads to OpenSim ExternalLoads .xml file.
    """
    ext_loads = osim.ExternalLoads()
    for force in external_forces:
        ext_loads.cloneAndAppend(force_to_opensim(force))
    if datafile_name is not None:
        ext_loads.setDataFileName(datafile_name)
    ext_loads.printToXML(filepath)
