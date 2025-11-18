import polars as pl
from itertools import product
from collections import deque, defaultdict
from pydantic import BaseModel, model_validator, Field, field_validator, ConfigDict
import pyopensim.simulation as osim
from loguru import logger
import numpy as np
import math

class OsimGraph(BaseModel):
    """
    A graph data structure representation of an OpenSim model with convenience functions.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    osim_model: osim.Model

    joint_bodies: dict[str, tuple[str, str]] = Field(default_factory=dict)  # Joint name -> (parent name, child name)
    bodies_joint: dict[frozenset[str], str] = Field(default_factory=dict)  # (parent name, child name) -> Joint name
    body_graph: dict[str, set[str]] = Field(default_factory=lambda: defaultdict(set))  # Adjacent body names
    muscle_attachments: dict[str, list[str]] = Field(default_factory=lambda: defaultdict(list))  # Muscle name -> body names (essentially an ordered set)
    muscle_wraps: dict[str, list[str]] = Field(default_factory=lambda: defaultdict(list))  # Muscle name -> wrap object names (essentially an ordered set)
    wraps_muscles: dict[str, set[str]] = Field(default_factory=lambda: defaultdict(set))  # Wrap object name -> muscle names
    body_wraps: dict[str, set[str]] = Field(default_factory=lambda: defaultdict(set))  # Body name -> wrap object names
    wrap_body: dict[str, str] = Field(default_factory=dict)  # Wrap object name -> body name
    path_cache: dict[frozenset[str], list[str]] = Field(default_factory=dict)  # Cached paths between body names
    muscle_crossings: dict[str, set[str]] = Field(default_factory=lambda: defaultdict(set))  # Muscle name -> crossed joint names
    crossings_muscle: dict[frozenset[str], set[str]] = Field(default_factory=lambda: defaultdict(set))  # Joint names -> muscle names
    muscle_coords: dict[str, set[str]] = Field(default_factory=lambda: defaultdict(set))  # Muscle name -> coordinate names
    coords_muscles: dict[frozenset[str], set[str]] = Field(default_factory=lambda: defaultdict(set))  # Coordinate names -> muscle names
    body_markers: dict[str, set[str]] = Field(default_factory=lambda: defaultdict(set))  # Body name -> marker names
    markers: dict[str, osim.Marker] = Field(default_factory=dict)  # Marker name -> marker object
    marker_bodies: dict[str, str] = Field(default_factory=dict)  # Marker name -> body name
    joint_coords: dict[str, set[str]] = Field(default_factory=lambda: defaultdict(set))  # Joint name -> coordinate names
    coord_ranges: dict[str, tuple[float, float]] = Field(default_factory=dict)  # Coordinate name -> (min, max) range

    @model_validator(mode='after')
    def build_graph(self):
        """Build all graph structures after model initialization."""
        self.create_rigid_graph()
        self.create_muscle_graph()
        self.cache_muscle_crossings()
        return self

    @field_validator('muscle_attachments')
    def unique_muscle_attachments(cls, v):
        """Ensure muscle attachments are unique."""
        for muscle, bodies in v.items():
            v[muscle] = list(dict.fromkeys(bodies))

    @field_validator('muscle_wraps')
    def unique_muscle_wraps(cls, v):
        """Ensure muscle wraps are unique."""
        for muscle, wraps in v.items():
            v[muscle] = list(dict.fromkeys(wraps))

    @classmethod
    def from_file(cls, model_path: str) -> 'OsimGraph':
        """Create an instance from a model file."""
        osim_model = osim.Model(model_path)
        return cls(osim_model=osim_model)

    def _get_component(self, name: str, component_set, component_type: str):
        """Generic method to get a component by name."""
        component = component_set.get(name)
        if not component:
            raise ValueError(f"{component_type} '{name}' not found in the model.")
        return component
        
    def get_muscle(self, muscle_name: str) -> osim.Muscle:
        """Get a muscle by its name."""
        return self._get_component(muscle_name, self.osim_model.getMuscles(), "Muscle")
    
    def get_body(self, body_name: str) -> osim.Body:
        """Get a body by its name."""
        return self._get_component(body_name, self.osim_model.getBodySet(), "Body")
    
    def get_joint(self, joint_name: str) -> osim.Joint:
        """Get a joint by its name."""
        return self._get_component(joint_name, self.osim_model.getJointSet(), "Joint")
    
    def get_coordinate(self, coord_name: str) -> osim.Coordinate:
        """Get a coordinate by its name."""
        return self._get_component(coord_name, self.osim_model.getCoordinateSet(), "Coordinate")
    
    def get_wrap(self, wrap_name: str) -> osim.WrapObject:
        """
        Get a wrap object by its name.
        """
        body = self.wrap_body.get(wrap_name)
        if not body:
            raise ValueError(f"Wrap object '{wrap_name}' not found in the model.")
        return self.get_body(body).getWrapObject(wrap_name)
    
    def _process_body_wraps(self, body_name: str, body: osim.Frame):
        """Process wrap objects for a body."""
        try:
            body_obj: osim.Body = osim.Body.safeDownCast(body)
            if body_obj:
                wrap_objects: osim.WrapObjectSet = body_obj.getWrapObjectSet()
                wrap_names = {wrap.getName() for wrap in wrap_objects}
                self.body_wraps[body_name] = wrap_names
                for wrap_name in wrap_names:
                    self.wrap_body[wrap_name] = body_name
        except Exception as e:
            logger.error(f"Error processing wrap objects for {body_name}: {e}")

    def create_rigid_graph(self):
        """
        Build the graph structure from the OpenSim model.
        This method populates the various dictionaries used to represent the model's structure.
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
            
            # Add coordinates to the joint's set
            for j in range(joint.numCoordinates()):
                coord: osim.Coordinate = joint.get_coordinates(j)
                coord_name = coord.getName()
                self.joint_coords[joint_name].add(coord_name)
                # Store coordinate ranges
                self.coord_ranges[coord_name] = (coord.getRangeMin(), coord.getRangeMax())
                logger.debug(f"Coordinate {coord_name} range: {self.coord_ranges[coord_name]}")
            
            # Build undirected graph of body connections
            self.body_graph[parent_name].add(child_name)
            self.body_graph[child_name].add(parent_name)

            if i == 0:
                self._process_body_wraps(parent_name, parent)
            self._process_body_wraps(child_name, child)

            logger.debug(f"Joint {joint_name} connects {parent_name} and {child_name}")
            
    def create_muscle_graph(self):
        """Cache all muscle attachments and wrap objects."""
        muscles: osim.SetMuscles = self.osim_model.getMuscles()
        
        for i in range(muscles.getSize()):
            muscle: osim.Muscle = muscles.get(i)
            muscle_name = muscle.getName()
            path: osim.GeometryPath = muscle.getGeometryPath()
            
            # Store bodies this muscle attaches to
            path_points: osim.PathPointSet = path.getPathPointSet()
            attached_bodies = []
            for j in range(path_points.getSize()):
                path_point: osim.PathPoint = path_points.get(j)
                frame: osim.Frame = path_point.getParentFrame()
                body: osim.Frame = frame.findBaseFrame()
                body_name = body.getName()
                if body_name not in attached_bodies:
                    attached_bodies.append(body_name)
            self.muscle_attachments[muscle_name] = attached_bodies

            # Store wrap objects
            path_wraps: osim.PathWrapSet = path.getWrapSet()
            wrap_objects = []
            for j in range(path_wraps.getSize()):
                path_wrap: osim.PathWrap = path_wraps.get(j)
                wrap_object_name = path_wrap.getWrapObjectName()
                if wrap_object_name not in wrap_objects:
                    wrap_objects.append(wrap_object_name)
                self.wraps_muscles[wrap_object_name].add(muscle_name)
            self.muscle_wraps[muscle_name] = wrap_objects
        
            logger.debug(f"Muscle {muscle_name} attaches to bodies {attached_bodies} and wraps {wrap_objects}")
    
    def find_path(self, bodies: list[str]) -> list[str]:
        """
        Find the shortest path between bodies using BFS.
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
            logger.warning(f"find_path currently only handles two bodies. Using first and last of {bodies}.")

        start, end = bodies[0], bodies[-1]
        
        # Check cache first
        cache_key = frozenset([start, end])
        if cache_key in self.path_cache:
            path = self.path_cache[cache_key]
            # Ensure path starts with the correct body
            return path if path[0] == start else path[::-1]

        if start == end:
            return [start]

        # BFS to find shortest path
        visited = {start}
        queue = deque([(start, [start])])
        
        while queue:
            current, path = queue.popleft()

            if current == end:
                self.path_cache[cache_key] = path
                logger.debug(f"Path found between {start} and {end}: {path}")
                return path
            
            for neighbor in self.body_graph.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
                    
        logger.debug(f"No path found between {start} and {end}.")
        return []
    
    def cache_muscle_crossings(self):
        """
        Caches the joints crossed by each muscle.
        This is used to quickly find which joints a muscle crosses without recalculating.
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
            self.muscle_coords[muscle_name] = set().union(*(self.joint_coords[joint] for joint in crossed_joints))
            self.coords_muscles[frozenset(self.muscle_coords[muscle_name])].add(muscle_name)

            logger.debug(f"Muscle {muscle_name} crosses joints {crossed_joints}")

    def get_muscles_crossing_joint(self, joint_name: str) -> set[str]:
        """Get all muscles that cross a specific joint."""
        return {muscle for muscle, joints in self.muscle_crossings.items() 
                if joint_name in joints}

    def get_muscles_actuating_coord(self, coord_name: str) -> set[str]:
        """Get all muscles that actuate a specific coordinate."""
        return {muscle for muscle, coords in self.muscle_coords.items() 
                if coord_name in coords}

    def get_joint_dof(self, joint_name: str) -> int:
        """Get the degrees of freedom for a joint."""
        return len(self.joint_coords.get(joint_name, set()))
    
    def clear_caches(self):
        """Clear all cached data structures."""
        self.path_cache.clear()
        logger.info("Caches cleared")

    def get_coordinate_combinations(self, coordinates: list[str], res: int = 2) -> np.ndarray: # TODO: range for each coordinate
        """Get all possible combinations of coordinate values."""
        coordinate_values = [np.linspace(*self.coord_ranges[coord], num=res) for coord in coordinates]
        return np.array(list(product(*coordinate_values)))

    def get_muscle_lengths_coordinates( 
        self,
        muscle_names: list[str],
        coordinates: list[str],
        min_points: int = 10,
    ) -> np.ndarray:
        """Analyze muscle length through the range of motion of multiple coordinates."""
        # Total points >= min_points = points per coordinate ^ number of coordinates
        points_per_coordinate = math.ceil(min_points ** (1 / len(coordinates)))
        coordinate_values = self.get_coordinate_combinations(coordinates, points_per_coordinate)
        state = self.osim_model.initSystem()
        data = np.zeros((coordinate_values.shape[0], len(coordinates) + len(muscle_names)))
        data[:, :len(coordinates)] = coordinate_values
        for i, values in enumerate(coordinate_values):
            for coord, value in zip(coordinates, values):
                self.get_coordinate(coord).setValue(state, value)
            self.osim_model.realizePosition(state)
            data[i, len(coordinates):] = np.array([self.get_muscle(muscle_name).getLength(state) for muscle_name in muscle_names])
        return data
    
    def get_muscle_lengths_rom(
        self,
        muscle_names: list[str],
        min_points: int = 10,
    ) -> pl.DataFrame:
        """Analyze muscle length through the range of motion of coordinates it crosses"""
        muscle_coords = list(set().union(*(self.muscle_coords[muscle] for muscle in muscle_names)))
        data = self.get_muscle_lengths_coordinates(muscle_names, muscle_coords, min_points)
        return pl.DataFrame(data, schema = muscle_coords + muscle_names)
        
    def get_all_muscle_lengths_rom(self, min_points : int = 10) -> dict[str, pl.DataFrame]:
        """
        Analyze muscle length through the range of motion of all coordinates.
        
        Returns:
            Dict[str, pd.DataFrame]: A dictionary where keys are muscle names and 
            values are DataFrames containing coordinate values and muscle lengths.
        """
        # TODO: Subsets and/or parallelization to speed up computation
        results = {}
        for coord_set, muscles in self.coords_muscles.items():
            # check for locked coordinates
            unlocked_coords = set([coord for coord in coord_set if not self.get_coordinate(coord).getDefaultLocked()])
            if not unlocked_coords:
                continue
            diff = coord_set.difference(unlocked_coords)
            # if diff:
            #     self.logger.warning(f"Locked coordinates {diff} for muscles {muscles}")
            df = self.get_muscle_lengths_rom(list(muscles), min_points=min_points)
            # Add the coordinate values and muscle lengths to the results dictionary for each muscle
            results.update({muscle: df[list(unlocked_coords) + [muscle]] for muscle in muscles})
        
        return results
    
    def get_muscle_lengths_from_data(
        self,
        muscle_names: list[str],
        data: pl.DataFrame,
    ) -> pl.DataFrame:
        """Calculate muscle lengths from coordinate data."""
        state = self.osim_model.initSystem()
        lengths = np.zeros((data.shape[0], len(muscle_names)))
        for i, row in enumerate(data.to_numpy()):
            for coord, value in zip(data.columns, row):
                self.get_coordinate(coord).setValue(state, value)
            self.osim_model.realizePosition(state)
            lengths[i] = np.array([self.get_muscle(muscle_name).getLength(state) for muscle_name in muscle_names])
        return pl.DataFrame(lengths, schema=muscle_names)
