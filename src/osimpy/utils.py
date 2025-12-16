import opensim as osim
import os
from pathlib import Path
from loguru import logger


def parse_enf_file(file_path: str, encoding: str = "utf-8") -> dict[str, str]:
    """
    Parse an .enf file and return key-value pairs.

    Args:
        file_path: Path to the .enf file
        encoding: File encoding (default: utf-8)

    Returns:
        Dictionary with lowercase keys and their values
    """
    data = {}
    with open(file_path, "r", encoding=encoding) as file:
        for line in file:
            line = line.lstrip("\ufeff").strip()
            if "=" in line:
                parts = line.split("=", 1)
                key = parts[0].strip() if len(parts) > 0 else ""
                value = parts[1].strip() if len(parts) > 1 else ""
                if key:
                    data[key.lower()] = value
    return data


def get_unit_conversion(from_units: str, to_units: str) -> float:
    if from_units == to_units:
        return 1.0
    from_u = osim.Units(from_units)
    to_u = osim.Units(to_units)
    return from_u.convertTo(to_u)


def createActuatorsFile(modelpath=None):
    # raise("Need to fix generation of Point and Torque Actuators")
    """
    Function to generate a generic OpenSim Actuator File from a Model by
    identifying the coordinates that are connected to ground and placing
    point or torque actuators on translational or rotational coordinates,
    respectively. All other coordinates will get coordinate actuators.
    Any constrained coordinates will be ignored.
    File is Printed to the same folder as the selected Model.

    Parameters:
        modelpath (str): path to an OSIM file
    """

    # If no model is input, get a path to one
    print("Loading the model...")
    if modelpath is None:
        # Note: GUI file selection would require additional imports like tkinter
        raise ValueError("Model path must be provided")
    elif not os.path.exists(modelpath):
        raise FileNotFoundError(f"Model file not found: {modelpath}")

    # Instantiate the model
    model = osim.Model(modelpath)

    # Instantiate the underlying computational System and return a handle to the State
    state = model.initSystem()

    # Get the coordinate set
    coordSet = model.getCoordinateSet()
    nCoord = coordSet.getSize()

    # Instantiate empty Vec3's
    massCenter = osim.Vec3(0, 0, 0)
    axisValues = osim.Vec3(0, 0, 0)

    # Instantiate an empty Force set
    forceSet = osim.ForceSet()

    # Set the optimal force
    optimalForce = 1

    # Process each coordinate
    for iCoord in range(nCoord):
        # Get reference to current coordinate
        coordinate = coordSet.get(iCoord)

        # Skip constrained coordinates
        if coordinate.isConstrained(state):
            continue

        # Get joint details
        joint = coordinate.getJoint()
        parentName = joint.getParentFrame().getName()
        childName = joint.getChildFrame().getName()

        # Check if parent is ground
        if parentName == model.getGround().getName():
            # Handle Custom and Free joints
            if (
                joint.getConcreteClassName() == "CustomJoint"
                or joint.getConcreteClassName() == "FreeJoint"
            ):
                motion = str(coordinate.getMotionType())

                # Get coordinate transform axis
                if joint.getConcreteClassName() == "CustomJoint":
                    concreteJoint = osim.CustomJoint.safeDownCast(joint)
                else:
                    concreteJoint = osim.FreeJoint.safeDownCast(joint)
                sptr = concreteJoint.getSpatialTransform()
                for ip in range(6):
                    if str(sptr.getCoordinateNames().get(ip)) == str(
                        coordinate.getName()
                    ):
                        sptr.getTransformAxis(ip).getAxis(axisValues)
                        break

                # Create appropriate actuator based on motion type
                if motion.lower() == "rotational":
                    newActuator = osim.TorqueActuator(
                        joint.getParentFrame(), joint.getParentFrame(), axisValues
                    )
                elif motion.lower() == "translational":
                    newActuator = osim.PointActuator()
                    newActuator.set_body(str(joint.getChildFrame().getName()))
                    newActuator.set_point(massCenter)
                    newActuator.set_point_is_global(False)
                    newActuator.set_direction(axisValues)
                    newActuator.set_force_is_global(True)
                else:
                    newActuator = osim.CoordinateActuator()
            else:
                newActuator = osim.CoordinateActuator()
        else:
            newActuator = osim.CoordinateActuator(str(coordinate.getName()))

        # Set actuator properties
        newActuator.setOptimalForce(optimalForce)
        newActuator.setName(coordinate.getName())
        newActuator.setMaxControl(float("inf"))
        newActuator.setMinControl(float("-inf"))

        # Add to force set
        forceSet.cloneAndAppend(newActuator)

    # Create output file path
    filepath = Path(modelpath)
    output_path = filepath.parent / f"{filepath.stem}_actuators.xml"

    # Print the actuators xml file
    forceSet.printToXML(str(output_path))
    print(f"Printed actuators to {output_path}")


def createCMCTaskSet(model_path):
    """
    Create a CMC TaskSet from an OpenSim model file.
    """
    # Load the model
    model = osim.Model(model_path)

    # Create a CMC TaskSet
    task_set = osim.CMC_TaskSet()

    coord_set: osim.SetCoordinates = model.getCoordinateSet()
    nCoord = coord_set.getSize()

    # Add a CMC_Joint to the TaskSet for each coordinate
    for i in range(nCoord):
        coord: osim.Coordinate = coord_set.get(i)
        coord_name = coord.getName()
        cmc_joint = osim.CMC_Joint()
        cmc_joint.setName(coord_name)
        # on - Flag (true or false) indicating whether or not a task is enabled
        cmc_joint.setOn(True)  # not coord.isConstrained())
        # weight - Weight with which a task is tracked relative to other tasks. To track a task more tightly, make the weight larger.
        cmc_joint.setWeight(1)

        # wrt_body - Name of body frame with respect to which a tracking objective is specified. The special name 'center_of_mass' refers to the system center of mass. This property is not used for tracking joint angles.
        cmc_joint.setWRTBodyName("-1")
        # express_body - Name of body frame in which the tracking objectives are expressed.  This property is not used for tracking joint angles
        cmc_joint.setExpressBodyName("-1")

        # Active - Array of 3 flags (each true or false) specifying whether a component of a task is active.  For example, tracking the trajectory of a point in space could have three components (x,y,z).  This allows each of those to be made active (true) or inactive (false).  A task for tracking a joint coordinate only has one component
        cmc_joint.setActive(True, False, False)
        # Kp - Position error feedback gain (stiffness). To achieve critical damping of errors, choose kv = 2*sqrt(kp).
        cmc_joint.setKP(100.0, 1.0, 1.0)
        # Kv - Velocity error feedback gain (damping). To achieve critical damping of errors, choose kv = 2*sqrt(kp).
        cmc_joint.setKV(20.0, 1.0, 1.0)
        # Ka - Feedforward acceleration gain.  This is normally set to 1.0, so no gain.
        cmc_joint.setKA(1.0, 1.0, 1.0)
        # r0 - Direction vector[3] for component 0 of a task. Joint tasks do not use this property
        cmc_joint.setDirection_0(osim.Vec3(0, 0, 0))
        # r1 - Direction vector[3] for component 1 of a task. Joint tasks do not use this property
        cmc_joint.setDirection_1(osim.Vec3(0, 0, 0))
        # r2 - Direction vector[3] for component 2 of a task. Joint tasks do not use this property
        cmc_joint.setDirection_2(osim.Vec3(0, 0, 0))

        # coordinate - Name of the coordinate to be tracked
        cmc_joint.setCoordinateName(coord_name)

        # limit - Error limit on the tracking accuracy for this coordinate. If the tracking errors approach this limit, the weighting for this coordinate is increased
        # There does not appear to be a method to set this property in the OpenSim API
        task_set.cloneAndAppend(cmc_joint)

    # Create output file path
    filepath = Path(model_path)
    output_path = filepath.parent / f"{filepath.stem}_taskSet.xml"
    task_set.printToXML(str(output_path))
    print(f"Printed task set to {output_path}")


# ===== Forceplate-to-Body Mapping Utilities =====


def get_forceplate_body_mapping_from_enf(
    enf_path: str, body_mapping: dict[str, str] = {"Left": "foot_l", "Right": "foot_r"}
) -> dict[int, str]:
    """
    Parse ENF file to determine which force platforms contact which bodies.

    This function reads forceplate assignments from a Vicon ENF file (e.g., FP3=Right)
    and maps them to OpenSim body names using the provided body_mapping dict.

    Args:
        enf_path: Path to the .enf file
        body_mapping: Dictionary mapping ENF context names (e.g., 'Left', 'Right')
                     to OpenSim body names (e.g., 'foot_l', 'foot_r')

    Returns:
        Dictionary mapping forceplate indices (1-based) to body names.
        Example: {3: 'foot_r', 2: 'foot_r'} means FP2 and FP3 both contact foot_r

    Example:
        >>> mapping = get_forceplate_body_mapping_from_enf(
        ...     enf_path="Walk05.Trial.enf",
        ...     body_mapping={'Left': 'foot_l', 'Right': 'foot_r'}
        ... )
        >>> # If ENF contains FP3=Right, returns {3: 'foot_r'}
    """

    # Parse ENF file
    enf_data = parse_enf_file(enf_path)

    # Find forceplate assignments (keys like 'fp1', 'fp2', 'fp3')
    fp_to_body = {}
    for key, value in enf_data.items():
        # Check if key matches pattern 'fp' followed by digits
        if key.startswith("fp") and key[2:].isdigit():
            fp_index = int(key[2:])  # Extract the number (1-based indexing)

            # Map the ENF context name to OpenSim body name
            body_name = body_mapping.get(value, None)
            if body_name is None:
                logger.warning(
                    f"ENF context '{value}' for {key.upper()} not found in body_mapping. "
                    f"Skipping this forceplate."
                )
                continue

            fp_to_body[fp_index] = body_name
            logger.debug(f"Mapped {key.upper()}={value} -> body '{body_name}'")

    if not fp_to_body:
        logger.warning(
            f"No forceplate assignments found in ENF file: {enf_path}. "
            f"Expected keys like 'FP1=Left', 'FP2=Right', etc."
        )

    return fp_to_body


def create_opensim_external_forces(
    forceplate_names: list[str],
    fp_to_body_mapping: dict[int, str],
    force_expressed_in_body: str = "ground",
    point_expressed_in_body: str = "ground",
):
    """
    Create OpenSimExternalForce objects for forceplates based on body mapping.

    This standalone function creates external force configurations for OpenSim ID analysis.
    It handles cases where multiple forceplates contact the same body.

    The fp_to_body_mapping uses lab-specific forceplate numbers (e.g., from ENF files),
    which are matched to platform indices using either:
    1. Direct index matching (FP3 → forceplate_names[2] if names are generic)
    2. Name matching (FP3 → find "Force Plate [3]" in forceplate_names)

    Args:
        forceplate_names: List of forceplate names (e.g., ["FP1", "FP2"] or
                         ["Bertec Force Plate [2]", "Bertec Force Plate [3]"])
        fp_to_body_mapping: Dictionary mapping forceplate numbers (1-based) to body names
                           Example: {3: 'foot_r', 4: 'foot_l'} means FP3→foot_r, FP4→foot_l
        force_expressed_in_body: Body frame in which forces are expressed
        point_expressed_in_body: Body frame in which application points are expressed

    Returns:
        List of OpenSimExternalForce objects

    Example:
        >>> # ENF says FP3=Right, FP4=Left
        >>> fp_mapping = {3: 'foot_r', 4: 'foot_l'}
        >>> # Forceplates named with lab numbers [2], [3], [4], [5]
        >>> forces = create_opensim_external_forces(
        ...     forceplate_names=["Bertec FP [2]", "Bertec FP [3]", "Bertec FP [4]", "Bertec FP [5]"],
        ...     fp_to_body_mapping=fp_mapping
        ... )
        >>> # Finds platform 1 for FP3, platform 2 for FP4
    """
    from loguru import logger
    from .io.write import OpenSimExternalForce

    if not fp_to_body_mapping:
        raise ValueError(
            "No forceplate-to-body mappings provided. "
            "Cannot create external forces for ID analysis."
        )

    external_forces = []

    for fp_number, body_name in fp_to_body_mapping.items():
        # Find which platform index corresponds to this forceplate number
        platform_idx = _find_platform_for_forceplate_number(fp_number, forceplate_names)

        if platform_idx is None:
            logger.warning(
                f"Could not find platform for FP{fp_number} in available forceplates: "
                f"{forceplate_names}. Skipping."
            )
            continue

        fp_name = forceplate_names[platform_idx]

        # Create force identifier prefix
        # Names are already sanitized in C3D adapter, so no need to clean here
        fp_prefix = fp_name

        ext_force = OpenSimExternalForce(
            name=f"{fp_prefix}_{body_name}",
            applied_to_body=body_name,
            force_expressed_in_body=force_expressed_in_body,
            point_expressed_in_body=point_expressed_in_body,
            force_identifier=f"{fp_prefix}_force_v",
            point_identifier=f"{fp_prefix}_force_p",
            torque_identifier=f"{fp_prefix}_moment_",
        )

        external_forces.append(ext_force)
        logger.info(
            f"Created external force '{ext_force.name}' for FP{fp_number} ({fp_name}) -> {body_name}"
        )

    if not external_forces:
        raise ValueError(
            "No valid external forces could be created. "
            "Check that forceplate numbers match available forceplates."
        )

    return external_forces


def _find_platform_for_forceplate_number(
    fp_number: int, forceplate_names: list[str]
) -> int | None:
    """
    Find the platform index for a given forceplate number.

    This handles two cases:
    1. Names with embedded numbers (e.g., "Bertec Force Plate [3]") - match the number
    2. Generic sequential names (e.g., "ForcePlate_0") - use direct index (fp_number - 1)

    Args:
        fp_number: Forceplate number from ENF file (1-based)
        forceplate_names: List of forceplate names

    Returns:
        Platform index (0-based) or None if not found
    """
    import re

    # First, try to find a name that contains this exact number
    for platform_idx, name in enumerate(forceplate_names):
        # Look for patterns like [3], (3), #3, _3, FP3, etc.
        # Try bracketed/delimited numbers first
        matches = re.findall(r"[\[\(#_](\d+)[\]\)]?", name)
        if matches and int(matches[-1]) == fp_number:  # Use last match (most specific)
            return platform_idx

        # Also try direct patterns like "FP3", "Plate3", etc.
        # Look for letters followed immediately by the number
        direct_matches = re.findall(r"[A-Za-z]+(\d+)", name)
        if direct_matches and int(direct_matches[-1]) == fp_number:
            return platform_idx

    # If no match found and names are generic (ForcePlate_0, ForcePlate_1, etc.),
    # use direct 1-based to 0-based conversion
    if fp_number >= 1 and fp_number <= len(forceplate_names):
        # Check if names follow the generic pattern
        if all(name.startswith("ForcePlate_") for name in forceplate_names):
            return fp_number - 1

    return None
