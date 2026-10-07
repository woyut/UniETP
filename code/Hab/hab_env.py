import os
import json
from pathlib import Path, PurePosixPath
from typing import List, Optional, Union, OrderedDict, Any
from collections import defaultdict


import magnum as mn
import numpy as np


import habitat_sim
from habitat.datasets.rearrange.navmesh_utils import get_largest_island_index
from habitat.sims.habitat_simulator.sim_utilities import (
    get_obj_from_id,
    get_obj_from_handle,
)

from general_env import Environment
from scene_graph.unified_scene_graph import UnifiedSceneGraph
import numpy as np
from task.commonsense_knowledge.hab import NEW_OBJ_CATEGORIES

from Hab.hab_actions import (
    moveAhead,
    moveBack,
    moveLeft,
    moveRight,
    turnLeft,
    turnRight,
    turnToLook,
    lookLeft,
    lookRight,
    lookUp,
    lookDown,
    resetHead,
    turnTo_obj,
    turnTo_point,
    goTo_loc,
    goTo_obj,
    pick_obj,
    placeIn_recep,
    placeOn_recep,
    placeOn_roomFloor,
    placeTo_recep,
    placeTo_point,
    open_recep,
    close_recep,
)
from Hab.hab_action_utils import sync_object_bindings


class Hab_Environment(Environment):
    def __init__(
        self,
        sceneID: str,
        max_steps: int,
        max_failed_steps: int,
        hab_data_dir: str = "",
        initial_position: List[int] = [0.0, 0.0, 0.0],
        moveMagnitude: float = 0.25,
        rotateDegree: int = 30,
        rotateUpDownDegree: int = 10,
        useDepth: bool = False,
        useSeg: bool = True,
        img_w: int = 256,
        img_h: int = 256,
        embodiment: str = "hab_fetch",
        add_object_placement: list = [],
        change_object_state: list = [],
        change_object_placement: list = [],
        **kwargs,
    ):
        self.agent_radius = 0.3
        self.agent_height = 1.5

        sim_cfg = habitat_sim.SimulatorConfiguration()

        sim_cfg.scene_id = os.path.join(
            hab_data_dir, "scenes-articulated", sceneID + ".scene_instance.json"
        )
        sim_cfg.scene_dataset_config_file = os.path.join(
            hab_data_dir, "hssd-hab-articulated++.scene_dataset_config.json"
        )
        sim_cfg.enable_physics = True

        sim_cfg.allow_sliding = False

        sensor_specs = []

        color_sensor_spec = habitat_sim.CameraSensorSpec()
        color_sensor_spec.uuid = "color_sensor"
        color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_spec.resolution = [img_h, img_w]
        color_sensor_spec.position = [0.0, 0.0, 0.0]
        color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(color_sensor_spec)

        if useDepth:
            depth_sensor_spec = habitat_sim.CameraSensorSpec()
            depth_sensor_spec.uuid = "depth_sensor"
            depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor_spec.resolution = [img_h, img_w]
            depth_sensor_spec.position = [0.0, 0.0, 0.0]
            depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(depth_sensor_spec)

        if useSeg:
            semantic_sensor_spec = habitat_sim.CameraSensorSpec()
            semantic_sensor_spec.uuid = "semantic_sensor"
            semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
            semantic_sensor_spec.resolution = [img_h, img_w]
            semantic_sensor_spec.position = [0.0, 0.0, 0.0]
            semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(semantic_sensor_spec)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=moveMagnitude)
            ),
            "move_backward": habitat_sim.agent.ActionSpec(
                "move_backward", habitat_sim.agent.ActuationSpec(amount=moveMagnitude)
            ),
            "move_left": habitat_sim.agent.ActionSpec(
                "move_left", habitat_sim.agent.ActuationSpec(amount=moveMagnitude)
            ),
            "move_right": habitat_sim.agent.ActionSpec(
                "move_right", habitat_sim.agent.ActuationSpec(amount=moveMagnitude)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=rotateDegree)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=rotateDegree)
            ),
            "look_up": habitat_sim.ActionSpec(
                "look_up", habitat_sim.ActuationSpec(amount=rotateUpDownDegree)
            ),
            "look_down": habitat_sim.ActionSpec(
                "look_down", habitat_sim.ActuationSpec(amount=rotateUpDownDegree)
            ),
        }

        metadata_mediator = kwargs.get("metadata_mediator", None)
        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg], metadata_mediator)
        self.sim = habitat_sim.Simulator(cfg)
        self.scene_id = sceneID

        with open(f"{hab_data_dir}/semantics/hssd-hab_semantic_lexicon.json") as f:
            file = json.load(f)
            self.semantic_mapping = {x["id"]: x["name"] for x in file["classes"]}
        i = 1000
        self.rev_semantic_mapping_new_objs = {}
        for x in NEW_OBJ_CATEGORIES:
            if x not in self.semantic_mapping.values():
                self.semantic_mapping[i] = x
                self.rev_semantic_mapping_new_objs[x] = i
                i += 1
            else:
                for k, v in self.semantic_mapping.items():
                    if v == x:
                        self.rev_semantic_mapping_new_objs[x] = k
                        break

        self.agent_init_position = initial_position
        self.embodiment_urdf_path = os.path.join(
            os.path.dirname(hab_data_dir), f"hab_fetch/robots/{embodiment}.urdf"
        )
        self.hab_data_dir = hab_data_dir
        self.versioned_data_dir = Path(hab_data_dir).resolve().parent
        self.habitat_data_root = self.versioned_data_dir.parent

        self.add_object_placement = add_object_placement
        self.change_object_state = change_object_state
        self.change_object_placement = change_object_placement

        self._init_common(max_steps, max_failed_steps, useSeg)

    def _resolve_asset_handle(self, handle: str) -> str:
        uri_prefix = "habitat-data://"
        if handle.startswith(uri_prefix):
            relative = PurePosixPath(handle[len(uri_prefix) :])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid Habitat asset URI: {handle!r}")
            # Habitat's template registry keys preserve the lexical
            # ``versioned_data/../objects`` form from the scene-dataset config.
            # Validate the canonical target, but return the same lexical form
            # so it matches the registered template handle exactly.
            lexical_path = (self.versioned_data_dir / "..").joinpath(
                *relative.parts
            )
            resolved = lexical_path.resolve()
            try:
                resolved.relative_to(self.habitat_data_root)
            except ValueError as exc:
                raise ValueError(
                    f"Habitat asset URI escapes the data root: {handle!r}"
                ) from exc
            return str(lexical_path)

        # Accept absolute handles and paths containing ``/versioned_data/`` for
        # compatibility with externally generated datasets. Portable datasets
        # should use the ``habitat-data://`` form above.
        marker = "/versioned_data/"
        if marker not in handle:
            return handle
        relative_path = handle.split(marker, 1)[1]
        lexical_path = self.versioned_data_dir / relative_path
        resolved = lexical_path.resolve()
        try:
            resolved.relative_to(self.habitat_data_root)
        except ValueError as exc:
            raise ValueError(
                f"Habitat asset path escapes the configured data root: {handle!r}"
            ) from exc
        return str(lexical_path)

    def init_episode(self):
        self.sim.reset()

        rom = self.sim.get_rigid_object_manager()
        for placement in self.add_object_placement:
            placement["handle"] = self._resolve_asset_handle(placement["handle"])

            new_obj = rom.add_object_by_template_handle(placement["handle"])
            new_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
            transform_list = placement["transform"]  # 4x4 list
            mat_np = np.array(transform_list)  # shape (4,4)
            mat_mn = mn.Matrix4(mat_np)
            new_obj.root_scene_node.transformation = mat_mn

            if placement["undo"]:
                rom.remove_object_by_handle(new_obj.handle)
            else:
                assert new_obj.object_id == placement["object_id"], [
                    new_obj.object_id,
                    placement["object_id"],
                    placement["handle"],
                ]

            new_obj.semantic_id = self.rev_semantic_mapping_new_objs[
                placement["category"]
            ]

        aom = self.sim.get_articulated_object_manager()
        for state_change in self.change_object_state:
            state_change["handle"] = self._resolve_asset_handle(state_change["handle"])
            obj = aom.get_object_by_handle(state_change["handle"])
            obj.joint_positions = state_change["joints"]
        for placement_change in self.change_object_placement:
            placement_change["handle"] = self._resolve_asset_handle(
                placement_change["handle"]
            )

            obj = get_obj_from_handle(self.sim, placement_change["handle"])
            obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
            transform_list = placement_change["transform"]  # 4x4 list
            mat_np = np.array(transform_list)  # shape (4,4)
            mat_mn = mn.Matrix4(mat_np)
            obj.root_scene_node.transformation = mat_mn
            assert (
                placement_change["category"] == self.semantic_mapping[obj.semantic_id]
            )
        self.sim.step_physics(1.0)

        self.object_id_to_class = {}
        rom = self.sim.get_rigid_object_manager()
        for obj_handle in rom.get_object_handles():
            obj = rom.get_object_by_handle(obj_handle)
            assert obj.object_id not in self.object_id_to_class
            self.object_id_to_class[obj.object_id] = self.semantic_mapping[
                obj.semantic_id
            ]
            for v in obj.visual_scene_nodes:
                v.semantic_id = obj.object_id

        self.link_object_id_to_class = {}
        self.link_object_id_to_object_id = {}
        self.object_id_to_link_object_id = defaultdict(dict)
        aom = self.sim.get_articulated_object_manager()
        for obj_handle in aom.get_object_handles():
            obj = aom.get_object_by_handle(obj_handle)
            assert obj.object_id not in self.object_id_to_class
            self.object_id_to_class[obj.object_id] = self.semantic_mapping[
                obj.creation_attributes.semantic_id
            ]
            obj.root_scene_node.drawable_semantic_id = obj.object_id
            obj.root_scene_node.semantic_id = obj.object_id

            link_scene_nodes = []
            for link_obj_id in obj.link_object_ids:
                assert (
                    link_obj_id not in self.object_id_to_class
                    and link_obj_id not in self.link_object_id_to_class
                )
                self.link_object_id_to_class[link_obj_id] = (
                    f"{self.semantic_mapping[obj.creation_attributes.semantic_id]}..{obj.object_id}..{obj.link_object_ids[link_obj_id]}"
                )
                self.link_object_id_to_object_id[link_obj_id] = obj.object_id
                assert (
                    obj.link_object_ids[link_obj_id]
                    not in self.object_id_to_link_object_id[obj.object_id]
                )
                self.object_id_to_link_object_id[obj.object_id][
                    obj.link_object_ids[link_obj_id]
                ] = link_obj_id
                link_scene_node = obj.get_link_scene_node(
                    obj.link_object_ids[link_obj_id]
                )
                link_scene_nodes.append(link_scene_node)

            for x in obj.visual_scene_nodes:
                p = x.parent
                while p is not None:
                    if p in link_scene_nodes:
                        break
                    p = p.parent
                assert isinstance(p, habitat_sim.scene.SceneNode)
                x.semantic_id = p.object_semantic_id
                assert p.object_semantic_id in self.link_object_id_to_class

        self.action_inputs = {
            "object_id_to_class": self.object_id_to_class,
            "link_object_id_to_object_id": self.link_object_id_to_object_id,
            "object_id_to_link_object_id": self.object_id_to_link_object_id,
            "step_idx": 0,
        }

        self._init_episode_common()
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_radius = self.agent_radius
        navmesh_settings.agent_height = self.agent_height
        navmesh_settings.include_static_objects = True
        navmesh_settings.agent_max_climb = 0.1
        navmesh_settings.agent_max_slope = 15.0
        self.navmesh_settings = navmesh_settings
        navmesh_success = self.sim.recompute_navmesh(
            self.sim.pathfinder, navmesh_settings
        )
        assert navmesh_success

        self.largest_indoor_island_idx = get_largest_island_index(
            self.sim.pathfinder, self.sim, allow_outdoor=False
        )

        rom = self.sim.get_rigid_object_manager()
        aom = self.sim.get_articulated_object_manager()
        for obj_handle in rom.get_object_handles():
            obj = rom.get_object_by_handle(obj_handle)
            obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC

        self.robot = aom.add_articulated_object_from_urdf(
            filepath=self.embodiment_urdf_path,
            fixed_base=False,
        )
        self.robot.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        self.robot.joint_velocities = np.zeros(len(self.robot.joint_velocities))

        assert all(abs(x) < 200 for x in self.agent_init_position), (
            self.agent_init_position
        )
        self.robot.translation = self.agent_init_position

        for v in self.robot.visual_scene_nodes:
            v.semantic_id = 10000
        self.object_id_to_class[10000] = "robot_agent"

        self.robot_link_name_to_joint_ids = {}
        for link_id in range(self.robot.num_links):
            if (
                self.robot.get_link_joint_type(link_id)
                == habitat_sim.physics.JointType.Fixed
            ):
                continue
            link_name = self.robot.get_link_name(link_id)
            offset = self.robot.get_link_joint_pos_offset(link_id)
            num = self.robot.get_link_num_joint_pos(link_id)
            self.robot_link_name_to_joint_ids[link_name] = [
                x for x in range(offset, offset + num)
            ]

        for link_id in self.robot.get_link_ids():
            if self.robot.get_link_name(link_id) == "gripper_link":
                self.gripper_link_idx = link_id
                break

        self.rest_pose = np.zeros(len(self.robot.joint_positions))
        stow_arm_angles = {
            "shoulder_pan_link": 1.32,
            "shoulder_lift_link": 1.40,
            "upperarm_roll_link": -0.2,
            "elbow_flex_link": 1.72,
            "forearm_roll_link": 0.0,
            "wrist_flex_link": 1.66,
            "wrist_roll_link": 0.0,
            "head_pan_link": 0.0,
            "head_tilt_link": 0.0,
            "torso_lift_link": 0.0,
        }
        for name, angle in stow_arm_angles.items():
            self.rest_pose[self.robot_link_name_to_joint_ids[name][0]] = angle

        self.carry_pose = np.zeros(len(self.robot.joint_positions))

        carry_arm_angles = {
            "shoulder_pan_link": -0.5,
            "shoulder_lift_link": 1.0,
            "upperarm_roll_link": 0.0,
            "elbow_flex_link": 0.5,
            "forearm_roll_link": 0.0,
            "wrist_flex_link": 0.0,
            "wrist_roll_link": 0.0,
            "head_pan_link": 0.0,
            "head_tilt_link": 0.5,
            "torso_lift_link": 0.0,
        }
        for name, angle in carry_arm_angles.items():
            self.carry_pose[self.robot_link_name_to_joint_ids[name][0]] = angle

        self.robot.joint_positions = self.rest_pose

        self.held_object_id = None
        head_node = self.robot.get_link_scene_node(
            self.robot.get_link_id_from_name("head_camera_link")
        )
        agent = self.sim.agents[0]
        agent.scene_node.parent = head_node
        rotation_correction = mn.Quaternion.rotation(mn.Deg(90), mn.Vector3(1, 0, 0))
        rotation_correction = mn.Quaternion.rotation(mn.Deg(270), mn.Vector3(1, 0, 0))
        rotation_correction = mn.Quaternion.rotation(mn.Deg(90), mn.Vector3(0, 1, 0))
        rotation_correction1 = mn.Quaternion.rotation(mn.Deg(270), mn.Vector3(0, 1, 0))
        rotation_correction2 = mn.Quaternion.rotation(mn.Deg(270), mn.Vector3(0, 0, 1))
        agent.scene_node.rotation = rotation_correction1 * rotation_correction2
        agent.scene_node.translation = mn.Vector3(0, 0, 0)

        self.action_inputs.update(
            {
                "pan_idx": self.robot_link_name_to_joint_ids["head_pan_link"][0],
                "tilt_idx": self.robot_link_name_to_joint_ids["head_tilt_link"][0],
                "gripper_idx": self.gripper_link_idx,
                "rest_pose": self.rest_pose,
                "carry_pose": self.carry_pose,
                "largest_indoor_island_idx": self.largest_indoor_island_idx,
                "sync_object_dict": OrderedDict(),
            }
        )

        self.sim.step_physics(1.0)

        observations = self.sim.get_sensor_observations()
        rgb = observations["color_sensor"]
        additional_info = {
            "instance_seg_frame": observations.get("semantic_sensor", None),
            "object_id_to_class": self.object_id_to_class,
            "link_object_id_to_class": self.link_object_id_to_class,
            "link_object_id_to_object_id": self.link_object_id_to_object_id,
            "nearby_object_and_link_ids": self._build_nearby_object_id_set(
                distance_threshold=3.0
            ),
        }
        return rgb, additional_info

    def _convert_ref_to_hab_object_id(
        self,
        ref_value: Union[str, List[float]],
        scene_graph: Optional[UnifiedSceneGraph],
    ):
        """
        Convert SG node id (e.g. category_idx) back to Hab objectId when needed.
        """
        if "_part_" in ref_value:
            object_id = ref_value.split("_part")[0].split("_")[-1]
            link_idx = ref_value.split("_part_")[1]
            object_category = ref_value.split(f"_{object_id}_part")[0]
            semantic = f"{object_category}..{object_id}..{link_idx}"
            for k, v in self.link_object_id_to_class.items():
                if v == semantic:
                    return str(k)
            assert False, [ref_value, semantic, self.link_object_id_to_class]
        elif ref_value in scene_graph.cache["sim_region_id_to_SG_node_ID"].values():
            return ref_value
        elif isinstance(ref_value, list):
            assert isinstance(ref_value[0], float), f"Invalid bbox: {ref_value}"
            assert len(ref_value) == 4, f"Invalid bbox: {ref_value}"
            return ref_value
        else:
            object_id = ref_value.split("_")[-1]
            assert int(object_id) in self.object_id_to_class, [
                ref_value,
                self.object_id_to_class,
            ]
            assert (
                f"{self.object_id_to_class[int(object_id)]}_{object_id}" == ref_value
            ), [ref_value, self.object_id_to_class[int(object_id)], object_id]
            return str(object_id)

    def _build_nearby_object_id_set(self, distance_threshold: float = 3.0) -> set[int]:
        agent_pos = np.array(self.sim.agents[0].state.position, dtype=np.float64)
        nearby_ids = set()

        for obj_id in self.object_id_to_class:
            if obj_id in [0, 10000]:
                continue
            obj = get_obj_from_id(self.sim, obj_id)
            obj_pos = np.array(obj.translation, dtype=np.float64)
            if np.linalg.norm(obj_pos - agent_pos) < distance_threshold:
                nearby_ids.add(obj_id)

        for link_obj_id, parent_obj_id in self.link_object_id_to_object_id.items():
            parent_obj = get_obj_from_id(self.sim, parent_obj_id)
            link_id = parent_obj.link_object_ids[link_obj_id]
            link_pos = np.array(
                parent_obj.get_link_scene_node(link_id).absolute_translation,
                dtype=np.float64,
            )
            if np.linalg.norm(link_pos - agent_pos) < distance_threshold:
                nearby_ids.add(link_obj_id)

        return nearby_ids

    def _normalize_action_args_refs(
        self,
        action_args: Optional[dict[str, Any]],
        scene_graph: Optional[UnifiedSceneGraph],
    ) -> dict[str, Any]:
        """
        Normalize action args for Hab:
        - target_position_pixel -> pixel
        - object_ref/tool_ref strings -> Hab objectId; bbox lists pass through
        """
        normalized_action_args = dict(action_args or {})
        if "target_position_pixel" in normalized_action_args:
            normalized_action_args["pixel"] = normalized_action_args.pop(
                "target_position_pixel"
            )
        for key, value in normalized_action_args.items():
            if key.endswith("_ref"):
                normalized_action_args[key] = self._convert_ref_to_hab_object_id(
                    value, scene_graph
                )
        return normalized_action_args

    def step(self, action_cls: str, action_args: dict, usg: UnifiedSceneGraph):

        rom = self.sim.get_rigid_object_manager()

        assert self.sim.pathfinder.is_loaded

        self.num_steps += 1
        done = self.num_steps >= self.max_steps
        args_valid = True
        try:
            action_args = self._normalize_action_args_refs(action_args, usg)
        except Exception:
            args_valid = False
            action_success, feedback = False, "Invalid action arguments"
            self.num_steps -= 1
            self.failed_steps -= 1
        try:
            if not args_valid:
                pass
            elif action_cls == "moveAhead":
                action_success, feedback = moveAhead(
                    self.sim, self.robot, self.action_inputs
                )
            elif action_cls == "moveBack":
                action_success, feedback = moveBack(
                    self.sim, self.robot, self.action_inputs
                )
            elif action_cls == "moveLeft":
                action_success, feedback = moveLeft(
                    self.sim, self.robot, self.action_inputs
                )
            elif action_cls == "moveRight":
                action_success, feedback = moveRight(
                    self.sim, self.robot, self.action_inputs
                )
            elif action_cls == "turnLeft":
                action_success, feedback = turnLeft(
                    self.sim, self.robot, self.action_inputs
                )
            elif action_cls == "turnRight":
                action_success, feedback = turnRight(
                    self.sim, self.robot, self.action_inputs
                )
            elif action_cls == "turnToLook":
                action_success, feedback = turnToLook(
                    self.sim, self.robot, self.action_inputs
                )
            elif action_cls == "lookLeft":
                action_success, feedback = lookLeft(
                    self.sim, self.robot, self.action_inputs
                )
            elif action_cls == "lookRight":
                action_success, feedback = lookRight(
                    self.sim, self.robot, self.action_inputs
                )
            elif action_cls == "lookUp":
                action_success, feedback = lookUp(
                    self.sim, self.robot, self.action_inputs
                )
            elif action_cls == "lookDown":
                action_success, feedback = lookDown(
                    self.sim, self.robot, self.action_inputs
                )
            elif action_cls == "resetHead":
                action_success, feedback = resetHead(
                    self.sim, self.robot, self.action_inputs
                )

            elif action_cls == "turnTo_obj":
                action_success, feedback = turnTo_obj(
                    self.sim, self.robot, usg, self.action_inputs, **action_args
                )
            elif action_cls == "turnTo_point":
                action_success, feedback = turnTo_point(
                    self.sim, self.robot, usg, self.action_inputs, **action_args
                )
            elif action_cls == "goTo_loc":
                action_success, feedback = goTo_loc(
                    self.sim, self.robot, usg, self.action_inputs, **action_args
                )
            elif action_cls == "goTo_obj":
                action_success, feedback = goTo_obj(
                    self.sim, self.robot, usg, self.action_inputs, **action_args
                )
            elif action_cls == "pick_obj":
                action_success, feedback = pick_obj(
                    self.sim, self.robot, usg, self.action_inputs, **action_args
                )
            elif action_cls == "placeIn_recep":
                action_success, feedback = placeIn_recep(
                    self.sim, self.robot, usg, self.action_inputs, **action_args
                )
            elif action_cls == "placeOn_recep":
                action_success, feedback = placeOn_recep(
                    self.sim, self.robot, usg, self.action_inputs, **action_args
                )
            elif action_cls == "placeTo_recep":
                action_success, feedback = placeTo_recep(
                    self.sim, self.robot, usg, self.action_inputs, **action_args
                )
            elif action_cls == "placeOn_roomFloor":
                action_success, feedback = placeOn_roomFloor(
                    self.sim, self.robot, usg, self.action_inputs, **action_args
                )
            elif action_cls == "placeTo_point":
                action_success, feedback = placeTo_point(
                    self.sim, self.robot, usg, self.action_inputs, **action_args
                )
            elif action_cls == "open_recep":
                action_success, feedback = open_recep(
                    self.sim, self.robot, usg, self.action_inputs, **action_args
                )
                navmesh_success = self.sim.recompute_navmesh(
                    self.sim.pathfinder, self.navmesh_settings
                )
                self.largest_indoor_island_idx = get_largest_island_index(
                    self.sim.pathfinder, self.sim, allow_outdoor=False
                )
                self.action_inputs["largest_indoor_island_idx"] = (
                    self.largest_indoor_island_idx
                )
                assert navmesh_success

                cur = np.array(self.robot.translation, dtype=np.float64)
                snapped = self.sim.pathfinder.snap_point(cur)
                if (
                    not np.isnan(snapped).any()
                    and np.linalg.norm(cur - np.array(snapped)) <= self.agent_radius
                ):
                    self.robot.translation = mn.Vector3(snapped)
            elif action_cls == "close_recep":
                action_success, feedback = close_recep(
                    self.sim, self.robot, usg, self.action_inputs, **action_args
                )
                navmesh_success = self.sim.recompute_navmesh(
                    self.sim.pathfinder, self.navmesh_settings
                )
                self.largest_indoor_island_idx = get_largest_island_index(
                    self.sim.pathfinder, self.sim, allow_outdoor=False
                )
                self.action_inputs["largest_indoor_island_idx"] = (
                    self.largest_indoor_island_idx
                )
                assert navmesh_success
            elif action_cls == "done":
                done = True
                action_success, feedback = True, "done"
            else:
                action_success, feedback = False, f"Invalid action"
                self.num_steps -= 1
                self.failed_steps -= 1
        except ValueError:
            action_success, feedback = False, "Invalid action arguments"
            self.num_steps -= 1
            self.failed_steps -= 1

        sync_object_bindings(self.sim, self.action_inputs["sync_object_dict"])
        self.sim.step_physics(1.0)

        self.failed_steps += 0 if action_success else 1
        done = done or (self.failed_steps >= self.max_failed_steps)

        observations = self.sim.get_sensor_observations()
        rgb = observations["color_sensor"]

        additional_info = {
            "instance_seg_frame": observations.get("semantic_sensor", None),
            "object_id_to_class": self.object_id_to_class,
            "link_object_id_to_class": self.link_object_id_to_class,
            "link_object_id_to_object_id": self.link_object_id_to_object_id,
            "nearby_object_and_link_ids": self._build_nearby_object_id_set(
                distance_threshold=3.0
            ),
        }

        self.action_inputs["step_idx"] += 1

        return rgb, action_success, feedback, done, additional_info

    def close(
        self,
    ):
        self.sim.close()
