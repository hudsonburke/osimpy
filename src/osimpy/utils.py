import opensim as osim
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    logger.info(f"Printed actuators to {output_path}")


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
    logger.info(f"Printed task set to {output_path}")
