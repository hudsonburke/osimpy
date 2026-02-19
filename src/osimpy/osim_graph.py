import polars as pl
from itertools import product
from collections import deque, defaultdict
from pydantic import BaseModel, model_validator, Field, field_validator, ConfigDict
from typing import TypeVar, Any, Literal
import opensim as osim
import sys
from loguru import logger
import numpy as np
import math

# Type variable for OpenSim component types
T = TypeVar("T")


def _process_coord_set_parallel(
    model_path: str,
    coord_set: frozenset,
    muscles: set,
    min_points: int,
    locked_coords: set,
) -> dict:
    """
    Process a single coordinate set for parallel execution.
    Must be at module level for pickling.

    Args:
        model_path: Path to the OpenSim model file (to recreate model in subprocess)
        coord_set: Frozenset of coordinate names
        muscles: Set of muscle names
        min_points: Minimum number of points to sample
        locked_coords: Set of locked coordinate names

    Returns:
        Dictionary mapping muscle names to DataFrames
    """
    # Recreate model and graph in this process
    model = osim.Model(model_path)
    # Can't import from self, so recreate inline
    # The graph will be built automatically via the validator
    graph = OsimGraph(osim_model=model, model_path=model_path)
    state = graph.osim_model.initSystem()

    # Filter unlocked coordinates
    unlocked_coords = coord_set - locked_coords
    if not unlocked_coords:
        return {}

    df = graph.get_muscle_lengths_rom(list(muscles), min_points=min_points, state=state)
    return {muscle: df[list(unlocked_coords) + [muscle]] for muscle in muscles}


class OsimGraph(BaseModel):
    """
    A graph data structure representation of an OpenSim model with convenience functions.

    This class provides a high-performance, cached representation of an OpenSim model's
    structure, optimized for efficient queries about relationships between bodies, joints,
    muscles, coordinates, markers, and wrap objects.

    Key Features:
        - Bidirectional mappings for fast lookups (e.g., joint->bodies and bodies->joint)
        - Cached shortest paths between bodies using BFS
        - Pre-computed muscle-joint crossing relationships
        - Muscle-coordinate actuation mappings
        - Marker-body relationships
        - Wrap object associations

    Graph Structure:
        The class maintains several interconnected graphs:
        - **Rigid body graph**: Undirected graph of body connections via joints
        - **Muscle graph**: Muscle attachments, path points, and wrap objects
        - **Marker graph**: Marker positions relative to bodies
        - **Coordinate graph**: Joint DOFs and their ranges

    Memory vs. Speed Tradeoff:
        This class stores bidirectional mappings (e.g., joint_bodies + bodies_joint)
        which doubles memory usage but provides O(1) lookups in both directions.
        This is intentional for performance-critical applications.

    Example:
        >>> # Load model and build graph
        >>> graph = OsimGraph.from_file("model.osim")
        >>>
        >>> # Get model summary
        >>> summary = graph.get_summary()
        >>> print(f"Model has {summary['num_muscles']} muscles")
        >>>
        >>> # Find which muscles cross a joint
        >>> muscles = graph.get_muscles_crossing_joint("knee_r")
        >>>
        >>> # Find path between bodies
        >>> path = graph.find_path(["pelvis", "foot_r"])
        >>>
        >>> # Analyze muscle lengths across ROM
        >>> state = graph.osim_model.initSystem()
        >>> lengths = graph.get_muscle_lengths_rom(
        ...     ["soleus_r", "gastroc_r"],
        ...     min_points=20,
        ...     state=state  # Reuse state for efficiency
        ... )

    Note:
        The graph is built automatically during initialization via the build_graph()
        validator. For large models, this may take several seconds.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    osim_model: osim.Model
    model_path: str | None = Field(
        default=None,
        description="Path to the model file (needed for parallel processing)",
    )

    joint_bodies: dict[str, tuple[str, str]] = Field(
        default_factory=dict, description="Joint name -> (parent name, child name)"
    )
    bodies_joint: dict[frozenset[str], str] = Field(
        default_factory=dict, description="(parent name, child name) -> Joint name"
    )
    body_graph: dict[str, set[str]] = Field(
        default_factory=lambda: defaultdict(set), description="Adjacent body names"
    )
    muscle_attachments: dict[str, list[str]] = Field(
        default_factory=lambda: defaultdict(list),
        description="Muscle name -> body names (essentially an ordered set)",
    )
    muscle_wraps: dict[str, list[str]] = Field(
        default_factory=lambda: defaultdict(list),
        description="Muscle name -> wrap object names (essentially an ordered set)",
    )
    wraps_muscles: dict[str, set[str]] = Field(
        default_factory=lambda: defaultdict(set),
        description="Wrap object name -> muscle names",
    )
    body_wraps: dict[str, set[str]] = Field(
        default_factory=lambda: defaultdict(set),
        description="Body name -> wrap object names",
    )
    wrap_body: dict[str, str] = Field(
        default_factory=dict, description="Wrap object name -> body name"
    )
    path_cache: dict[frozenset[str], list[str]] = Field(
        default_factory=dict, description="Cached paths between body names"
    )
    muscle_crossings: dict[str, set[str]] = Field(
        default_factory=lambda: defaultdict(set),
        description="Muscle name -> crossed joint names",
    )
    crossings_muscle: dict[frozenset[str], set[str]] = Field(
        default_factory=lambda: defaultdict(set),
        description="Joint names -> muscle names",
    )
    muscle_coords: dict[str, set[str]] = Field(
        default_factory=lambda: defaultdict(set),
        description="Muscle name -> coordinate names",
    )
    coords_muscles: dict[frozenset[str], set[str]] = Field(
        default_factory=lambda: defaultdict(set),
        description="Coordinate names -> muscle names",
    )
    body_markers: dict[str, set[str]] = Field(
        default_factory=lambda: defaultdict(set),
        description="Body name -> marker names",
    )
    markers: dict[str, osim.Marker] = Field(
        default_factory=dict, description="Marker name -> marker object"
    )
    marker_bodies: dict[str, str] = Field(
        default_factory=dict, description="Marker name -> body name"
    )
    joint_coords: dict[str, set[str]] = Field(
        default_factory=lambda: defaultdict(set),
        description="Joint name -> coordinate names",
    )
    coord_ranges: dict[str, tuple[float, float]] = Field(
        default_factory=dict, description="Coordinate name -> (min, max) range"
    )
    locked_coords: set[str] = Field(
        default_factory=set, description="Set of locked coordinate names"
    )
    joint_muscles: dict[str, set[str]] = Field(
        default_factory=lambda: defaultdict(set),
        description="Joint name -> muscle names crossing it (reverse index)",
    )
    coord_muscles: dict[str, set[str]] = Field(
        default_factory=lambda: defaultdict(set),
        description="Coordinate name -> muscle names actuating it (reverse index)",
    )
    log_level: Literal[
        "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    ] = Field(default="ERROR", description="Logging level for the graph operations")

    @model_validator(mode="after")
    def build_graph(self):
        """Build all graph structures after model initialization."""
        logger.remove()
        logger.add(sys.stderr, level=self.log_level)
        self.create_rigid_graph()
        self.create_muscle_graph()
        self.create_marker_graph()
        self.cache_muscle_crossings()
        return self

    @field_validator("muscle_attachments")
    def unique_muscle_attachments(cls, v):
        """Ensure muscle attachments are unique."""
        for muscle, bodies in v.items():
            v[muscle] = list(dict.fromkeys(bodies))
        return v

    @field_validator("muscle_wraps")
    def unique_muscle_wraps(cls, v):
        """Ensure muscle wraps are unique."""
        for muscle, wraps in v.items():
            v[muscle] = list(dict.fromkeys(wraps))
        return v

    @classmethod
    def from_file(cls, model_path: str) -> "OsimGraph":
        """
        Create an OsimGraph instance from a model file.

        Args:
            model_path: Path to the OpenSim model file (.osim)

        Returns:
            OsimGraph instance with all graphs built

        Example:
            >>> graph = OsimGraph.from_file("gait2392.osim")
            >>> print(graph.get_summary())
        """
        osim_model = osim.Model(model_path)
        return cls(osim_model=osim_model, model_path=model_path)

    def _get_component(self, name: str, component_set: Any, component_type: str) -> T:
        """
        Generic method to get a component by name from an OpenSim component set.

        Args:
            name: Name of the component to retrieve
            component_set: OpenSim set object (e.g., MuscleSet, BodySet)
            component_type: Human-readable type name for error messages

        Returns:
            The requested component object

        Raises:
            ValueError: If the component is not found in the set
        """
        component = component_set.get(name)
        if not component:
            raise ValueError(f"{component_type} '{name}' not found in the model.")
        return component

    def get_muscle(self, muscle_name: str) -> osim.Muscle:
        """
        Get a muscle by its name.

        Args:
            muscle_name: Name of the muscle

        Returns:
            OpenSim Muscle object

        Raises:
            ValueError: If muscle not found in model
        """
        return self._get_component(muscle_name, self.osim_model.getMuscles(), "Muscle")

    def get_body(self, body_name: str) -> osim.Body:
        """
        Get a body by its name.

        Args:
            body_name: Name of the body

        Returns:
            OpenSim Body object

        Raises:
            ValueError: If body not found in model
        """
        return self._get_component(body_name, self.osim_model.getBodySet(), "Body")

    def get_joint(self, joint_name: str) -> osim.Joint:
        """
        Get a joint by its name.

        Args:
            joint_name: Name of the joint

        Returns:
            OpenSim Joint object

        Raises:
            ValueError: If joint not found in model
        """
        return self._get_component(joint_name, self.osim_model.getJointSet(), "Joint")

    def get_coordinate(self, coord_name: str) -> osim.Coordinate:
        """
        Get a coordinate by its name.

        Args:
            coord_name: Name of the coordinate

        Returns:
            OpenSim Coordinate object

        Raises:
            ValueError: If coordinate not found in model
        """
        return self._get_component(
            coord_name, self.osim_model.getCoordinateSet(), "Coordinate"
        )

    def get_wrap(self, wrap_name: str) -> osim.WrapObject:
        """
        Get a wrap object by its name.

        Args:
            wrap_name: Name of the wrap object

        Returns:
            OpenSim WrapObject

        Raises:
            ValueError: If wrap object not found in model
        """
        body = self.wrap_body.get(wrap_name)
        if not body:
            raise ValueError(f"Wrap object '{wrap_name}' not found in the model.")
        return self.get_body(body).getWrapObject(wrap_name)

    def _validate_state_realized(self, state: osim.State) -> None:
        """
        Validate that state is realized to at least Position stage.

        Args:
            state: OpenSim State to validate

        Raises:
            RuntimeError: If state is not realized to at least Position stage

        Note:
            This validation helps provide clearer error messages when users
            attempt to query muscle lengths without properly realizing the state.
        """
        try:
            # Try to check the realization stage
            # OpenSim's State.getSystemStage() returns the current stage
            stage = state.getSystemStage()
            # Stage values: Topology=0, Model=1, Instance=2, Time=3, Position=4, Velocity=5, etc.
            # We need at least Position (4) for muscle length queries
            if stage.getValue() < 4:  # Less than Position stage
                raise RuntimeError(
                    f"State must be realized to at least Position stage for muscle length queries. "
                    f"Current stage: {stage}. Call model.realizePosition(state) first."
                )
        except Exception as e:
            # If we can't check the stage, log a warning but don't fail
            logger.warning(f"Could not validate state realization: {e}")

    def _process_body_wraps(self, body_name: str, body: osim.Frame):
        """
        Process wrap objects for a body.

        Args:
            body_name: Name of the body
            body: OpenSim Frame object to process

        Note:
            This method safely attempts to downcast Frame to Body and extract
            wrap objects. If the frame is not a Body or has no wrap objects,
            it silently continues without error.
        """
        try:
            body_obj: osim.Body = osim.Body.safeDownCast(body)
            if body_obj:
                wrap_objects: osim.WrapObjectSet = body_obj.getWrapObjectSet()
                wrap_names = {wrap.getName() for wrap in wrap_objects}
                self.body_wraps[body_name] = wrap_names
                for wrap_name in wrap_names:
                    self.wrap_body[wrap_name] = body_name
        except (AttributeError, RuntimeError) as e:
            # AttributeError: Frame doesn't support wrap objects
            # RuntimeError: OpenSim-specific errors during wrap access
            logger.warning(
                f"Could not process wrap objects for body '{body_name}': {e}. "
                "This body may not support wrap objects or has an incompatible frame type.",
            )
        except Exception as e:
            # Unexpected error - log at error level and include more context
            logger.error(
                f"Unexpected error processing wrap objects for body '{body_name}': {type(e).__name__}: {e}",
            )

    def create_rigid_graph(self):
        """
        Build the rigid body graph structure from the OpenSim model.

        This method populates the following dictionaries:
        - joint_bodies: Maps joint names to (parent, child) body tuples
        - bodies_joint: Maps body pairs to joint names
        - body_graph: Creates undirected graph of body adjacencies
        - joint_coords: Maps joints to their coordinate sets
        - coord_ranges: Stores coordinate min/max ranges
        - locked_coords: Set of locked coordinate names
        - body_wraps: Maps bodies to their wrap objects
        - wrap_body: Maps wrap objects back to their bodies

        This method is called automatically during initialization.
        """
        joints: osim.JointSet = self.osim_model.getJointSet()

        for i in range(joints.getSize()):
            joint: osim.Joint = joints.get(i)
            # Get the actual bodies that the joint connects
            parent: osim.Frame = joint.getParentFrame().findBaseFrame()
            child: osim.Frame = joint.getChildFrame().findBaseFrame()

            parent_name = parent.getName()
            child_name = child.getName()
            joint_name = joint.getName()

            # Store the joint and its body connections
            self.joint_bodies[joint_name] = (parent_name, child_name)
            self.bodies_joint[frozenset([parent_name, child_name])] = joint_name

            # Add coordinates to the joint's set and check for locked status
            for j in range(joint.numCoordinates()):
                coord: osim.Coordinate = joint.get_coordinates(j)
                coord_name = coord.getName()
                self.joint_coords[joint_name].add(coord_name)
                # Store coordinate ranges
                self.coord_ranges[coord_name] = (
                    coord.getRangeMin(),
                    coord.getRangeMax(),
                )
                # Pre-compute locked status
                if coord.getDefaultLocked():
                    self.locked_coords.add(coord_name)
                logger.debug(
                    f"Coordinate {coord_name} range: {self.coord_ranges[coord_name]}, locked: {coord.getDefaultLocked()}"
                )

            # Build undirected graph of body connections
            self.body_graph[parent_name].add(child_name)
            self.body_graph[child_name].add(parent_name)

            if i == 0:
                self._process_body_wraps(parent_name, parent)
            self._process_body_wraps(child_name, child)

            logger.debug(f"Joint {joint_name} connects {parent_name} and {child_name}")

    def create_muscle_graph(self):
        """
        Build muscle attachment and wrap object relationships.

        This method populates the following dictionaries:
        - muscle_attachments: Maps muscles to ordered list of body attachments
        - muscle_wraps: Maps muscles to list of wrap objects they interact with
        - wraps_muscles: Maps wrap objects to muscles that use them

        This method is called automatically during initialization.
        """
        muscles: osim.SetMuscles = self.osim_model.getMuscles()

        for i in range(muscles.getSize()):
            muscle: osim.Muscle = muscles.get(i)
            muscle_name = muscle.getName()
            path: osim.GeometryPath = muscle.getGeometryPath()

            # Store bodies this muscle attaches to (use dict.fromkeys to preserve order and uniqueness)
            path_points: osim.PathPointSet = path.getPathPointSet()
            body_names_seen = {}
            for j in range(path_points.getSize()):
                path_point: osim.PathPoint = path_points.get(j)
                frame: osim.Frame = path_point.getParentFrame()
                body: osim.Frame = frame.findBaseFrame()
                body_name = body.getName()
                body_names_seen[body_name] = None
            attached_bodies = list(body_names_seen.keys())
            self.muscle_attachments[muscle_name] = attached_bodies

            # Store wrap objects (use dict.fromkeys to preserve order and uniqueness)
            path_wraps: osim.PathWrapSet = path.getWrapSet()
            wrap_names_seen = {}
            for j in range(path_wraps.getSize()):
                path_wrap: osim.PathWrap = path_wraps.get(j)
                wrap_object_name = path_wrap.getWrapObjectName()
                wrap_names_seen[wrap_object_name] = None
                self.wraps_muscles[wrap_object_name].add(muscle_name)
            wrap_objects = list(wrap_names_seen.keys())
            self.muscle_wraps[muscle_name] = wrap_objects

            logger.debug(
                f"Muscle {muscle_name} attaches to bodies {attached_bodies} and wraps {wrap_objects}"
            )

    def create_marker_graph(self):
        """
        Build marker-body relationships from the OpenSim model.

        Populates body_markers, markers, and marker_bodies dictionaries
        with information about all markers in the model.
        """
        marker_set: osim.MarkerSet = self.osim_model.getMarkerSet()

        for i in range(marker_set.getSize()):
            marker: osim.Marker = marker_set.get(i)
            marker_name = marker.getName()

            # Get the body this marker is attached to
            frame: osim.Frame = marker.getParentFrame()
            body: osim.Frame = frame.findBaseFrame()
            body_name = body.getName()

            # Store the relationships
            self.markers[marker_name] = marker
            self.marker_bodies[marker_name] = body_name
            self.body_markers[body_name].add(marker_name)

            logger.debug(f"Marker {marker_name} attached to body {body_name}")

    def find_path(self, bodies: list[str]) -> list[str]:
        """
        Find the shortest path between bodies using BFS with parent tracking.
        Uses a frozenset for order-independent caching of paths.

        Args:
            bodies: list of body names to find a path between. Currently handles two bodies.

        Returns:
            list of body names representing the path, or empty list if no path exists.

        Raises:
            ValueError: If less than 2 bodies are provided.
        """
        if len(bodies) < 2:
            raise ValueError("find_path requires at least two bodies.")

        if len(bodies) > 2:
            logger.warning(
                f"find_path currently only handles two bodies. Using first and last of {bodies}."
            )

        start, end = bodies[0], bodies[-1]

        # Check cache first
        cache_key = frozenset([start, end])
        if cache_key in self.path_cache:
            path = self.path_cache[cache_key]
            # Ensure path starts with the correct body
            return path if path[0] == start else path[::-1]

        if start == end:
            return [start]

        # BFS to find shortest path using parent tracking
        visited = {start}
        parent: dict[str, str | None] = {start: None}
        queue = deque([start])

        while queue:
            current = queue.popleft()

            if current == end:
                # Reconstruct path from parent tracking
                path = []
                node: str | None = end
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()

                self.path_cache[cache_key] = path
                logger.debug(f"Path found between {start} and {end}: {path}")
                return path

            for neighbor in self.body_graph.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)

        logger.debug(f"No path found between {start} and {end}.")
        return []

    def cache_muscle_crossings(self):
        """
        Cache the joints and coordinates crossed by each muscle.

        For each muscle, this method:
        1. Finds the path between attachment bodies
        2. Identifies which joints are crossed
        3. Determines which coordinates are actuated
        4. Creates bidirectional mappings for fast lookups
        5. Builds reverse indices for O(1) joint->muscles and coord->muscles lookups

        Populates:
        - muscle_crossings: Muscle -> set of crossed joints
        - crossings_muscle: Frozenset of joints -> muscles that cross them
        - muscle_coords: Muscle -> set of actuated coordinates
        - coords_muscles: Frozenset of coords -> muscles that actuate them
        - joint_muscles: Joint -> muscles crossing it (reverse index)
        - coord_muscles: Coordinate -> muscles actuating it (reverse index)

        This method is called automatically during initialization and uses
        the find_path() method which employs BFS for shortest path finding.
        """
        for muscle_name, body_names in self.muscle_attachments.items():
            crossed_joints = set()

            path = self.find_path(list(body_names))
            for i in range(len(path) - 1):
                parent, child = path[i], path[i + 1]
                joint = self.bodies_joint.get(frozenset([parent, child]))
                if joint:
                    crossed_joints.add(joint)

            self.muscle_crossings[muscle_name] = crossed_joints
            self.crossings_muscle[frozenset(crossed_joints)].add(muscle_name)

            # Build reverse index: joint -> muscles
            for joint in crossed_joints:
                self.joint_muscles[joint].add(muscle_name)

            # Determine actuated coordinates
            self.muscle_coords[muscle_name] = set().union(
                *(self.joint_coords[joint] for joint in crossed_joints)
            )
            self.coords_muscles[frozenset(self.muscle_coords[muscle_name])].add(
                muscle_name
            )

            # Build reverse index: coord -> muscles
            for coord in self.muscle_coords[muscle_name]:
                self.coord_muscles[coord].add(muscle_name)

            logger.debug(f"Muscle {muscle_name} crosses joints {crossed_joints}")

    def get_muscles_crossing_joint(self, joint_name: str) -> set[str]:
        """
        Get all muscles that cross a specific joint.

        Args:
            joint_name: Name of the joint

        Returns:
            Set of muscle names that cross the specified joint

        Example:
            >>> muscles = graph.get_muscles_crossing_joint("knee_r")
            >>> print(f"Knee is crossed by: {muscles}")
        """
        # Use reverse index for O(1) lookup
        return self.joint_muscles.get(joint_name, set())

    def get_muscles_actuating_coord(self, coord_name: str) -> set[str]:
        """
        Get all muscles that actuate a specific coordinate.

        Args:
            coord_name: Name of the coordinate

        Returns:
            Set of muscle names that actuate the specified coordinate

        Example:
            >>> muscles = graph.get_muscles_actuating_coord("knee_angle_r")
            >>> print(f"Coordinate actuated by: {muscles}")
        """
        # Use reverse index for O(1) lookup
        return self.coord_muscles.get(coord_name, set())

    def get_joint_dof(self, joint_name: str) -> int:
        """
        Get the degrees of freedom for a joint.

        Args:
            joint_name: Name of the joint

        Returns:
            Number of coordinates (DOFs) for the joint

        Example:
            >>> dof = graph.get_joint_dof("knee_r")
            >>> print(f"Knee has {dof} DOF")
        """
        return len(self.joint_coords.get(joint_name, set()))

    def get_muscle_names(self) -> list[str]:
        """
        Get all muscle names in the model.

        Returns:
            Sorted list of muscle names.
        """
        return sorted(self.muscle_attachments.keys())

    def get_body_names(self) -> list[str]:
        """
        Get all body names in the model.

        Returns:
            Sorted list of body names.
        """
        return sorted(self.body_graph.keys())

    def get_joint_names(self) -> list[str]:
        """
        Get all joint names in the model.

        Returns:
            Sorted list of joint names.
        """
        return sorted(self.joint_bodies.keys())

    def get_coordinate_names(self) -> list[str]:
        """
        Get all coordinate names in the model.

        Returns:
            Sorted list of coordinate names.
        """
        return sorted(self.coord_ranges.keys())

    def get_marker_names(self) -> list[str]:
        """
        Get all marker names in the model.

        Returns:
            Sorted list of marker names.
        """
        return sorted(self.markers.keys())

    def get_wrap_names(self) -> list[str]:
        """
        Get all wrap object names in the model.

        Returns:
            Sorted list of wrap object names.
        """
        return sorted(self.wrap_body.keys())

    def get_summary(self) -> dict[str, int]:
        """
        Get a summary of the model structure.

        Returns:
            Dictionary containing counts of various model components:
            - num_bodies: Number of bodies in the model
            - num_joints: Number of joints in the model
            - num_muscles: Number of muscles in the model
            - num_coordinates: Number of coordinates (DOFs) in the model
            - num_markers: Number of markers in the model
            - num_wraps: Number of wrap objects in the model

        Example:
            >>> graph = OsimGraph.from_file("model.osim")
            >>> summary = graph.get_summary()
            >>> print(f"Model has {summary['num_muscles']} muscles")
        """
        return {
            "num_bodies": len(self.body_graph),
            "num_joints": len(self.joint_bodies),
            "num_muscles": len(self.muscle_attachments),
            "num_coordinates": len(self.coord_ranges),
            "num_markers": len(self.markers),
            "num_wraps": len(self.wrap_body),
        }

    def clear_caches(self):
        """
        Clear all cached data structures.

        Currently clears:
            - path_cache: Cached shortest paths between bodies

        Note:
            This does NOT rebuild the entire graph structure (joints, muscles, etc.).
            Those structures are only built once during initialization.
            If you need to rebuild the entire graph, create a new OsimGraph instance.
        """
        self.path_cache.clear()
        logger.info("Path cache cleared")

    def get_coordinate_combinations(
        self, coordinates: list[str], res: int = 2
    ) -> np.ndarray:
        """
        Get all possible combinations of coordinate values across their ranges.

        Args:
            coordinates: List of coordinate names
            res: Number of points to sample along each coordinate's range

        Returns:
            2D array of shape (res^n_coords, n_coords) containing all combinations
            where n_coords is the number of coordinates

        Note:
            The total number of points grows exponentially with the number of
            coordinates: n_points = res^n_coords. Use with caution for high
            dimensional spaces.

        Example:
            >>> # Sample 10 points along knee and ankle ranges
            >>> combos = graph.get_coordinate_combinations(
            ...     ["knee_angle_r", "ankle_angle_r"],
            ...     res=10
            ... )
            >>> print(combos.shape)  # (100, 2) = 10^2 combinations
        """
        coordinate_values = [
            np.linspace(*self.coord_ranges[coord], num=res) for coord in coordinates
        ]
        return np.array(list(product(*coordinate_values)))

    def get_muscle_length(self, muscle_name: str, state: osim.State) -> float:
        """
        Get the length of a muscle at a given state.

        Args:
            muscle_name: Name of the muscle
            state: OpenSim State (must have position realized)

        Returns:
            Muscle length in meters

        Note:
            You must call model.realizePosition(state) before calling this method.
        """
        return self.get_muscle(muscle_name).getLength(state)

    def get_muscle_lengths(
        self, muscle_names: list[str], state: osim.State
    ) -> np.ndarray:
        """
        Get the lengths of multiple muscles at a given state.

        Args:
            muscle_names: List of muscle names
            state: OpenSim State (must have position realized)

        Returns:
            1D array of muscle lengths corresponding to muscle_names

        Note:
            You must call model.realizePosition(state) before calling this method.
        """
        # Cache muscle objects to avoid repeated lookups
        muscle_objs = [self.get_muscle(name) for name in muscle_names]
        return np.array([muscle.getLength(state) for muscle in muscle_objs])

    def get_muscle_lengths_coordinates(
        self,
        muscle_names: list[str],
        coordinates: list[str],
        min_points: int = 10,
        state: osim.State | None = None,
    ) -> np.ndarray:
        """
        Analyze muscle length through the range of motion of multiple coordinates.

        Args:
            muscle_names: List of muscle names to analyze
            coordinates: List of coordinate names to vary
            min_points: Minimum number of total points to sample
            state: Optional pre-initialized OpenSim State. If None, a new state
                   will be created and initialized (this is expensive).

        Returns:
            Array with shape (n_points, n_coords + n_muscles) where first columns
            are coordinate values and remaining columns are muscle lengths.

        Note:
            Creating and initializing a state is computationally expensive.
            If calling this method multiple times, consider creating a state once
            and passing it to each call.
        """
        # Total points >= min_points = points per coordinate ^ number of coordinates
        points_per_coordinate = math.ceil(min_points ** (1 / len(coordinates)))
        coordinate_values = self.get_coordinate_combinations(
            coordinates, points_per_coordinate
        )

        # Use provided state or create a new one
        if state is None:
            state = self.osim_model.initSystem()

        # Cache coordinate and muscle objects to avoid repeated lookups
        coord_objs = [self.get_coordinate(coord) for coord in coordinates]
        muscle_objs = [self.get_muscle(name) for name in muscle_names]

        data = np.zeros(
            (coordinate_values.shape[0], len(coordinates) + len(muscle_names))
        )
        data[:, : len(coordinates)] = coordinate_values
        for i, values in enumerate(coordinate_values):
            for coord_obj, value in zip(coord_objs, values):
                coord_obj.setValue(state, value)
            self.osim_model.realizePosition(state)
            for j, muscle_obj in enumerate(muscle_objs):
                data[i, len(coordinates) + j] = muscle_obj.getLength(state)
        return data

    def get_muscle_lengths_rom(
        self,
        muscle_names: list[str],
        min_points: int = 10,
        state: osim.State | None = None,
    ) -> pl.DataFrame:
        """
        Analyze muscle length through the range of motion of coordinates they cross.

        Args:
            muscle_names: List of muscle names to analyze
            min_points: Minimum number of total points to sample
            state: Optional pre-initialized OpenSim State. If None, a new state
                   will be created (expensive operation).

        Returns:
            DataFrame with coordinate columns followed by muscle length columns.

        Note:
            Locked coordinates are automatically filtered out and will not appear
            in the results. Only unlocked (variable) coordinates are included.
        """
        # Collect all coordinates that the muscles cross
        muscle_coords = list(
            set().union(*(self.muscle_coords[muscle] for muscle in muscle_names))
        )

        # Filter out locked coordinates to avoid setValue() errors
        unlocked_coords = [
            coord for coord in muscle_coords if coord not in self.locked_coords
        ]

        # Warn if any coordinates were filtered out
        locked_in_set = set(muscle_coords) - set(unlocked_coords)
        if locked_in_set:
            logger.debug(
                f"Excluding locked coordinates from ROM calculation: {locked_in_set}"
            )

        # If no unlocked coordinates remain, return empty DataFrame with just muscle lengths
        if not unlocked_coords:
            logger.warning(
                f"All coordinates for muscles {muscle_names} are locked. "
                "Returning muscle lengths at default pose only."
            )
            # Return single row with default pose
            if state is None:
                state = self.osim_model.initSystem()
            muscle_objs = [self.get_muscle(name) for name in muscle_names]
            self.osim_model.realizePosition(state)
            lengths = np.array(
                [muscle.getLength(state) for muscle in muscle_objs]
            ).reshape(1, -1)
            return pl.DataFrame(lengths, schema=muscle_names)

        # Compute muscle lengths across ROM for unlocked coordinates
        data = self.get_muscle_lengths_coordinates(
            muscle_names, unlocked_coords, min_points, state=state
        )
        return pl.DataFrame(data, schema=unlocked_coords + muscle_names)

    def get_all_muscle_lengths_rom(
        self, min_points: int = 10, use_parallel: bool = False, n_jobs: int = -1
    ) -> dict[str, pl.DataFrame]:
        """
        Analyze muscle length through the range of motion for all muscles.

        Groups muscles by their actuated coordinate sets and computes muscle
        lengths across the full range of motion for each group. This is more
        efficient than computing each muscle individually.

        Args:
            min_points: Minimum total number of points to sample
            use_parallel: Whether to use parallel processing (default: False)
            n_jobs: Number of parallel jobs (-1 uses all CPUs, default: -1)

        Returns:
            Dictionary mapping muscle names to DataFrames. Each DataFrame contains:
            - Columns for unlocked coordinates that the muscle crosses
            - A column for the muscle length values

        Note:
            - Locked coordinates are automatically excluded using pre-computed cache
            - This can be computationally expensive for models with many muscles
            - Set use_parallel=True for faster computation on multi-core systems

        Example:
            >>> results = graph.get_all_muscle_lengths_rom(min_points=20)
            >>> soleus_df = results["soleus_r"]
            >>> print(soleus_df.head())
        """
        # Create state once and reuse it for all muscle length calculations
        state = self.osim_model.initSystem()

        results = {}

        # Prepare coordinate sets for processing
        coord_set_items = list(self.coords_muscles.items())

        # Check if tqdm is available
        has_tqdm = False
        try:
            from tqdm import tqdm as tqdm_progress

            has_tqdm = True
        except ImportError:
            tqdm_progress = None
            if len(coord_set_items) > 10:  # Only warn for larger jobs
                logger.info("Install tqdm for progress bars: pip install tqdm")

        if use_parallel and len(coord_set_items) > 1:
            # Parallel processing requires model_path
            if self.model_path is None:
                raise ValueError(
                    "Parallel processing requires model_path. "
                    "Use OsimGraph.from_file() instead of direct instantiation, "
                    "or set use_parallel=False."
                )

            # Parallel processing
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(
                max_workers=n_jobs if n_jobs > 0 else None
            ) as executor:
                futures = {
                    executor.submit(
                        _process_coord_set_parallel,
                        self.model_path,
                        coord_set,
                        muscles,
                        min_points,
                        self.locked_coords,
                    ): (coord_set, muscles)
                    for coord_set, muscles in coord_set_items
                }

                future_iterator = as_completed(futures)
                if has_tqdm and tqdm_progress is not None:
                    future_iterator = tqdm_progress(
                        future_iterator,
                        total=len(futures),
                        desc="Processing coordinate sets",
                    )

                for future in future_iterator:
                    results.update(future.result())
        else:
            # Sequential processing
            if has_tqdm and tqdm_progress is not None:
                coord_iterator = tqdm_progress(
                    coord_set_items, desc="Processing coordinate sets"
                )
            else:
                coord_iterator = coord_set_items

            for coord_set, muscles in coord_iterator:
                # Filter unlocked coordinates using pre-computed cache
                unlocked_coords = coord_set - self.locked_coords
                if not unlocked_coords:
                    continue

                diff = coord_set.difference(unlocked_coords)
                if diff:
                    logger.warning(f"Locked coordinates {diff} for muscles {muscles}")

                df = self.get_muscle_lengths_rom(
                    list(muscles), min_points=min_points, state=state
                )
                # Add the coordinate values and muscle lengths to the results dictionary for each muscle
                results.update(
                    {muscle: df[list(unlocked_coords) + [muscle]] for muscle in muscles}
                )

        return results

    def get_muscle_lengths_from_data(
        self,
        muscle_names: list[str],
        data: pl.DataFrame,
        state: osim.State | None = None,
    ) -> pl.DataFrame:
        """
        Calculate muscle lengths from coordinate data.

        Args:
            muscle_names: List of muscle names to compute lengths for
            data: DataFrame with coordinate columns
            state: Optional pre-initialized OpenSim State. If None, a new state
                   will be created (expensive operation).

        Returns:
            DataFrame with muscle length columns corresponding to muscle_names.
        """
        # Use provided state or create a new one
        if state is None:
            state = self.osim_model.initSystem()

        # Cache coordinate and muscle objects to avoid repeated lookups
        coord_objs = [self.get_coordinate(coord) for coord in data.columns]
        muscle_objs = [self.get_muscle(name) for name in muscle_names]

        lengths = np.zeros((data.shape[0], len(muscle_names)))
        for i, row in enumerate(data.to_numpy()):
            for coord_obj, value in zip(coord_objs, row):
                coord_obj.setValue(state, value)
            self.osim_model.realizePosition(state)
            for j, muscle_obj in enumerate(muscle_objs):
                lengths[i, j] = muscle_obj.getLength(state)
        return pl.DataFrame(lengths, schema=muscle_names)
