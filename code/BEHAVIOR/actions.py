import sys
import os
import numpy as np
from BEHAVIOR.ModifiedSemanticActionPrimitive import (
    ModifiedSemanticActionPrimitiveSet,
    ModifiedSemanticActionPrimitives,
)


_current_dir = os.path.dirname(os.path.abspath(__file__))
_omnigibson_path = os.path.join(_current_dir, "BEHAVIOR-1K", "OmniGibson")
if os.path.exists(_omnigibson_path) and _omnigibson_path not in sys.path:
    sys.path.insert(0, _omnigibson_path)

import omnigibson as og
from BEHAVIOR.env_utils import *
from omnigibson.robots import BaseRobot
import torch as th
from omnigibson.transition_rules import *
from typing import List, Tuple, Dict
from omnigibson.action_primitives.action_primitive_set_base import (
    ActionPrimitiveErrorGroup,
)
from omnigibson.utils.transform_utils import (
    quat2euler,
    relative_pose_transform,
)
from omnigibson.utils.geometry_utils import wrap_angle
from omnigibson.objects import StatefulObject
from omnigibson.action_primitives.action_primitive_set_base import (
    ActionPrimitiveErrorGroup,
)
from omnigibson.sensors import VisionSensor
from scipy.spatial.transform import Rotation as R


class ExecuteActions:
    def __init__(self, env_wrapper, robot: BaseRobot, verbose: bool = False):
        self.robot = robot
        camera_sensor = None
        for sensor_name, sensor in self.robot.sensors.items():
            if isinstance(sensor, VisionSensor) and "eyes" in sensor_name.lower():
                camera_sensor = sensor
                break
        if camera_sensor is None:
            for sensor in self.robot.sensors.values():
                if isinstance(sensor, VisionSensor):
                    camera_sensor = sensor
                    break
        if camera_sensor is None:
            raise ValueError(
                f"No VisionSensor found on robot {self.robot.name}. Available sensors: {list(self.robot.sensors.keys())}"
            )
        self.camera = camera_sensor
        self.no_op = {f"{self.robot.name}": np.zeros(self.robot.action_dim)}
        self.primitives = ModifiedSemanticActionPrimitives(env_wrapper, robot)
        self.PS = ModifiedSemanticActionPrimitiveSet
        self.verbose = verbose
        self.env_wrapper = env_wrapper

    def get_all_room_names(self) -> List[str]:
        if (
            hasattr(self.env_wrapper.env.scene, "_seg_map")
            and self.env_wrapper.env.scene._seg_map.room_ins_name_to_ins_id is not None
        ):
            return list(
                self.env_wrapper.env.scene._seg_map.room_ins_name_to_ins_id.keys()
            )
        return []

    def _run_generator(self, primitive_type, *args) -> Tuple[bool, str, Dict]:
        try:
            gen = self.primitives.apply_ref(primitive_type, *args)
            for action in gen:
                if action is not None:
                    self.env_wrapper.env.step(action)
            primitive_desc = getattr(primitive_type, "__doc__", None)
            msg = f"Successfully {primitive_desc}"
            return True, msg, {}
        except ActionPrimitiveErrorGroup as e:
            if not e.exceptions:
                return False, "", {}
            error = e.exceptions[-1]
            message = str(error)
            prefix = f"{error.reason.name}: "
            if message.startswith(prefix):
                message = message[len(prefix) :]
            message = message.rsplit(". Additional info:", 1)[0]
            return False, message, error.metadata
        except Exception:
            import traceback

            traceback.print_exc()
            return False, "", {}

    def _execute(self, primitive_type, *args) -> Tuple[bool, str]:
        success, msg, meta = self._run_generator(primitive_type, *args)
        return success, msg

    def pick_obj(self, object_ref):
        return self._execute(self.PS.GRASP, object_ref)

    def drop(self):
        return self._execute(self.PS.RELEASE)

    def place_on_top(self, object_ref):
        return self._execute(self.PS.PLACE_ON_TOP, object_ref)

    def place_inside(self, object_ref):
        return self._execute(self.PS.PLACE_INSIDE, object_ref)

    def open_obj(self, object_ref):
        return self._execute(self.PS.OPEN, object_ref)

    def close_obj(self, object_ref):
        return self._execute(self.PS.CLOSE, object_ref)

    def turnOn_obj(self, object_ref):
        return self._execute(self.PS.TOGGLE_ON, object_ref)

    def turnOff_obj(self, object_ref):
        return self._execute(self.PS.TOGGLE_OFF, object_ref)

    def soak_with_tool(self, tool_ref):
        return self._execute(self.PS.SOAK, tool_ref)

    def clean_obj(self, object_ref):
        return self._execute(self.PS.CLEAN_OBJ, object_ref)

    def cut(self, object_ref):
        return self._execute(self.PS.CUT, object_ref)

    def _target_xy_navigable_on_trav_map(self, pos_xy: np.ndarray) -> bool:
        if self.env_wrapper is None:
            return True
        scene = self.env_wrapper.env.scene
        if not hasattr(scene, "trav_map") or scene.trav_map is None:
            return True
        trav_map = scene.trav_map
        fm = getattr(trav_map, "floor_map", None)
        if fm is None or len(fm) == 0:
            return True
        floor = 0
        trav_map_data = th.clone(trav_map.floor_map[floor])
        trav_map_data = trav_map._erode_trav_map(trav_map_data, robot=self.robot)
        candidate_pos_map = trav_map.world_to_map(pos_xy)
        ms = trav_map.map_size
        if ms is None:
            return True
        yi = int(candidate_pos_map[0].item())
        xi = int(candidate_pos_map[1].item())
        if yi < 0 or yi >= ms or xi < 0 or xi >= ms:
            return False
        return int(trav_map_data[yi, xi].item()) == 255

    def _move_target_same_room_as_robot(self, target_xy: np.ndarray) -> bool:
        if self.env_wrapper is None:
            return True
        scene = self.env_wrapper.env.scene
        if not hasattr(scene, "_seg_map") or scene._seg_map is None:
            return True
        rp = self.robot.get_position()
        if isinstance(rp, th.Tensor):
            robot_2d = rp[:2].detach().cpu().numpy()
        else:
            robot_2d = np.asarray(rp[:2], dtype=np.float64)
        robot_room = scene._seg_map.get_room_instance_by_point(robot_2d)
        if robot_room is None:
            return True
        target_room = scene._seg_map.get_room_instance_by_point(target_xy)
        return target_room == robot_room

    def _move_relative(self, distance, angle_offset=0, direction="Ahead"):

        pos, quat = self.robot.get_position_orientation()

        euler = quat2euler(quat)
        if not isinstance(euler, th.Tensor):
            euler = th.tensor(euler)

        yaw = euler[2]

        if not isinstance(angle_offset, th.Tensor):
            angle_offset_tensor = th.tensor(
                angle_offset, dtype=yaw.dtype, device=yaw.device
            )
        else:
            angle_offset_tensor = angle_offset
        target_yaw = yaw + angle_offset_tensor

        new_x = pos[0] + distance * th.cos(target_yaw)
        new_y = pos[1] + distance * th.sin(target_yaw)

        pose_2d = [new_x.item(), new_y.item(), yaw.item()]

        target_xy = np.array([pose_2d[0], pose_2d[1]], dtype=np.float64)
        if not self._target_xy_navigable_on_trav_map(target_xy):
            return False, "The movement is blocked."
        if not self._move_target_same_room_as_robot(target_xy):
            return False, "The movement is blocked."

        success, _, _ = self._run_generator(self.PS.NAVIGATE_TO, pose_2d)
        if self.env_wrapper is not None:
            for _ in range(6):
                self.env_wrapper.env.step(self.env_wrapper.no_op)
            robot_pos, robot_orn = self.robot.get_position_orientation()
            if (
                th.isnan(robot_pos).any()
                or th.isinf(robot_pos).any()
                or th.isnan(robot_orn).any()
                or th.isinf(robot_orn).any()
            ):
                return False, ""
        return success, ""

    def moveAhead(self):
        return self._move_relative(1, angle_offset=0, direction="Ahead")

    def moveBack(self):
        return self._move_relative(1, angle_offset=np.pi, direction="Back")

    def moveLeft(self):
        return self._move_relative(1, angle_offset=np.pi / 2, direction="Left")

    def moveRight(self):
        return self._move_relative(1, angle_offset=-np.pi / 2, direction="Right")

    def goTo_obj(self, object_ref):
        if isinstance(object_ref, str):
            return self.moveToRoom(object_ref)
        success, _, _ = self._run_generator(self.PS.NAVIGATE_TO, object_ref)
        return success, ""

    def _rotate_relative(
        self, yaw_delta: float, direction_name: str
    ) -> Tuple[bool, str]:
        try:
            pos, quat = self.robot.get_position_orientation()

            if isinstance(quat, th.Tensor):
                current_quat = quat.cpu().numpy()
            elif not isinstance(quat, np.ndarray):
                current_quat = np.array(quat)
            else:
                current_quat = quat

            yaw_rotation = R.from_euler("Z", yaw_delta)
            yaw_quat = yaw_rotation.as_quat()
            new_quat = quaternion_multiply(yaw_quat, current_quat)

            if np.any(np.isnan(new_quat)) or np.any(np.isinf(new_quat)):
                new_quat = np.array([0, 0, 0, 1.0])
            norm = np.linalg.norm(new_quat)
            if norm < 1e-6:
                new_quat = np.array([0, 0, 0, 1.0])
            else:
                new_quat = new_quat / norm
                if new_quat[3] < 0:
                    new_quat = -new_quat

            if isinstance(pos, th.Tensor):
                pos = pos.cpu().numpy()
            elif not isinstance(pos, np.ndarray):
                pos = np.array(pos)
            if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
                pos = self.robot.get_position()
                if isinstance(pos, th.Tensor):
                    pos = pos.cpu().numpy()
                elif not isinstance(pos, np.ndarray):
                    pos = np.array(pos)

            self.robot.set_position_orientation(pos, new_quat)

            if hasattr(self.robot, "keep_still"):
                self.robot.keep_still()
            return True, f"Successfully turned {direction_name}"
        except Exception as e:
            return False, f"Failed to turn {direction_name}"

    def turnLeft(self) -> Tuple[bool, str]:
        return self._rotate_relative(np.pi / 2, "Left")

    def turnRight(self) -> Tuple[bool, str]:
        return self._rotate_relative(-np.pi / 2, "Right")

    def moveToRoom(self, room_name: str) -> Tuple[bool, str]:
        seg_map = self.env_wrapper.env.scene._seg_map
        if room_name not in seg_map.room_ins_name_to_ins_id:
            available_rooms = list(seg_map.room_ins_name_to_ins_id.keys())
            return (
                False,
                f"Invalid room name: {room_name}! Available rooms: {available_rooms}",
            )

        floor, robot_pos = seg_map.get_random_point_by_room_instance(room_name)
        if robot_pos is None:
            return False, f"Failed to get random point in room: {room_name}!"

        if isinstance(robot_pos, th.Tensor):
            robot_pos = robot_pos.cpu().numpy()
        robot_pos[2] = 0.004

        grasped_obj = self.env_wrapper.grasped_obj
        task_objs = self.env_wrapper.task_objs
        robot_orientation = self.env_wrapper.robot_orientation
        ret, msg = move(
            self.robot,
            None,
            robot_pos,
            grasped_obj,
            task_objs,
            orientation=robot_orientation,
        )
        return ret, msg

    def turnTo_obj(self, object_ref) -> Tuple[bool, str]:
        try:
            robot_pos, robot_quat = self.robot.get_position_orientation()
            obj_pos, obj_quat = object_ref.get_position_orientation()
            if not isinstance(robot_quat, th.Tensor):
                robot_quat = th.tensor(robot_quat)
            if not isinstance(robot_pos, th.Tensor):
                robot_pos = th.tensor(robot_pos)
            if not isinstance(obj_pos, th.Tensor):
                obj_pos = th.tensor(obj_pos)
            if not isinstance(obj_quat, th.Tensor):
                obj_quat = th.tensor(obj_quat)
            obj_in_robot = relative_pose_transform(
                obj_pos, obj_quat, robot_pos, robot_quat
            )
            obj_pos_in_robot = obj_in_robot[0]

            target_yaw = th.atan2(obj_pos_in_robot[1], obj_pos_in_robot[0])
            target_yaw_float = (
                target_yaw.item()
                if isinstance(target_yaw, th.Tensor)
                else float(target_yaw)
            )

            yaw_diff_float = wrap_angle(target_yaw_float)
            obj_name = getattr(object_ref, "name", "the object")
            if abs(yaw_diff_float) >= 0.01:
                ret, msg = self._rotate_relative(yaw_diff_float, f"to face {obj_name}")
            self.env_wrapper.get_obs()
            for _ in range(3):
                og.sim.step()
            camera_pos_world, camera_quat_world = self.camera.get_position_orientation()
            if isinstance(camera_pos_world, th.Tensor):
                camera_pos_world = camera_pos_world.cpu().numpy()
            if isinstance(obj_pos, th.Tensor):
                obj_pos_np = obj_pos.cpu().numpy()
            else:
                obj_pos_np = np.array(obj_pos)
            if not isinstance(camera_pos_world, np.ndarray):
                camera_pos_world = np.array(camera_pos_world)
            vec_to_obj = obj_pos_np - camera_pos_world
            dist_horizontal = np.sqrt(vec_to_obj[0] ** 2 + vec_to_obj[1] ** 2)
            dist_vertical = vec_to_obj[2]
            target_pitch = np.arctan2(dist_vertical, dist_horizontal)
            pitch_diff = target_pitch - self.env_wrapper.camera_pitch
            pitch_diff = np.clip(pitch_diff, -np.pi / 2, np.pi / 2)
            if abs(pitch_diff) >= 0.01:
                if pitch_diff > 0:
                    return self.lookUp(pitch_diff)
                else:
                    return self.lookDown(abs(pitch_diff))
            else:
                return True, f"Successfully faced {obj_name}"
        except Exception as e:
            return False, f"Failed to turn to object: {str(e)}"

    def fill_with(self, tool_ref: StatefulObject):
        return self._execute(self.PS.FILL_WITH, tool_ref)

    def pour_into(self, object_ref: StatefulObject):
        return self._execute(self.PS.POUR_INTO, object_ref)

    def _pixel_to_world_ray_target(self, pixel: List[float]) -> np.ndarray:
        target = pixel2D_to_world_point(self.camera, pixel)
        if target is None:
            raise ValueError(
                f"Cannot unproject pixel {pixel} to a 3D point "
                f"(invalid or missing depth_linear at pixel)."
            )
        return target

    def _pitch_diff_to_world_point(self, target_pos_np: np.ndarray) -> float:
        camera_pos_world, _ = self.camera.get_position_orientation()
        if isinstance(camera_pos_world, th.Tensor):
            camera_pos_world = camera_pos_world.cpu().numpy()
        if not isinstance(camera_pos_world, np.ndarray):
            camera_pos_world = np.array(camera_pos_world)
        target_pos_np = np.asarray(target_pos_np, dtype=np.float64)
        vec_to_target = target_pos_np - camera_pos_world
        dist_horizontal = np.sqrt(vec_to_target[0] ** 2 + vec_to_target[1] ** 2)
        dist_vertical = vec_to_target[2]
        target_pitch = np.arctan2(dist_vertical, dist_horizontal)
        pitch_diff = target_pitch - self.env_wrapper.camera_pitch
        return float(np.clip(pitch_diff, -np.pi / 2, np.pi / 2))

    def turnTo_point(self, target_position_pixel: List[float]) -> Tuple[bool, str]:
        if (
            not isinstance(target_position_pixel, list)
            or len(target_position_pixel) != 2
        ):
            return (
                False,
                f"Invalid point format! Expected [x, y], got {target_position_pixel}",
            )
        try:
            target_pos = self._pixel_to_world_ray_target(target_position_pixel)
            robot_pos, robot_quat = self.robot.get_position_orientation()
            if not isinstance(robot_quat, th.Tensor):
                robot_quat = th.tensor(robot_quat)
            if not isinstance(robot_pos, th.Tensor):
                robot_pos = th.tensor(robot_pos)
            if not isinstance(target_pos, np.ndarray):
                target_pos = np.asarray(target_pos, dtype=np.float64)
            target_quat = th.tensor([0.0, 0.0, 0.0, 1.0])
            target_pos_t = th.tensor(target_pos, dtype=th.float32)
            target_in_robot = relative_pose_transform(
                target_pos_t, target_quat, robot_pos, robot_quat
            )
            target_pos_in_robot = target_in_robot[0]
            target_yaw = th.atan2(target_pos_in_robot[1], target_pos_in_robot[0])
            yaw_diff_float = wrap_angle(
                target_yaw.item()
                if isinstance(target_yaw, th.Tensor)
                else float(target_yaw)
            )
            if abs(yaw_diff_float) >= 0.01:
                ret, msg = self._rotate_relative(
                    yaw_diff_float, f"to face point {target_position_pixel}"
                )
                if not ret:
                    return ret, msg
            self.env_wrapper.get_obs()
            for _ in range(3):
                og.sim.step()
            target_pos = self._pixel_to_world_ray_target(target_position_pixel)
            pitch_diff = self._pitch_diff_to_world_point(target_pos)
            if abs(pitch_diff) >= 0.01:
                if pitch_diff > 0:
                    return self.lookUp(pitch_diff)
                return self.lookDown(abs(pitch_diff))
            return True, f"Successfully faced point {target_position_pixel}"
        except Exception as e:
            return False, f"Failed to turn to point: {str(e)}"

    def placeTo_point(self, target_position_pixel: List[float]) -> Tuple[bool, str]:
        if (
            not isinstance(target_position_pixel, list)
            or len(target_position_pixel) != 2
        ):
            return (
                False,
                f"Invalid point format! Expected [x, y], got {target_position_pixel}",
            )
        if len(self.env_wrapper.grasped_obj) == 0:
            return False, "No object grasped! You need to grasp an object first."
        recep_obj = self.env_wrapper.pixel2D_to_obj(target_position_pixel)
        if not recep_obj:
            return False, "No object found at the target position"
        return self.place_on_top(recep_obj)

    def spread(self, object_ref: StatefulObject):
        return self._execute(self.PS.SPREAD, object_ref)

    def lookUp(self, angle: float = 0.3) -> Tuple[bool, str]:
        return self._rotate_camera_pitch(angle, "up")

    def lookDown(self, angle: float = 0.3) -> Tuple[bool, str]:
        return self._rotate_camera_pitch(-angle, "down")

    def _rotate_camera_pitch(
        self, pitch_delta: float, direction: str
    ) -> Tuple[bool, str]:
        try:
            camera_pos, camera_quat = self.camera.get_position_orientation(
                frame="parent"
            )
            if isinstance(camera_quat, th.Tensor):
                camera_quat_np = camera_quat.cpu().numpy()
            elif not isinstance(camera_quat, np.ndarray):
                camera_quat_np = np.array(camera_quat)
            else:
                camera_quat_np = camera_quat

            rotation = R.from_euler("X", pitch_delta)
            rotation_quat = rotation.as_quat()
            new_quat_np = quaternion_multiply(rotation_quat, camera_quat_np)
            new_quat_np = validate_and_normalize_quaternion(new_quat_np)
            new_quat = th.tensor(new_quat_np, dtype=th.float32)
            self.camera.set_position_orientation(
                position=camera_pos, orientation=new_quat, frame="parent"
            )
            for _ in range(3):
                og.sim.render()
            if self.env_wrapper is not None:
                self.env_wrapper.camera_pitch = (
                    self.env_wrapper.camera_pitch + pitch_delta
                )
            msg = f"Successfully looked {direction} by {abs(pitch_delta):.3f} radians ({np.degrees(abs(pitch_delta)):.1f}°)"
            return True, msg
        except Exception as e:
            return False, f"Failed to look {direction}: {str(e)}"

    def find_objCls(self, object_cls: str, by_dist: bool = True):
        task_objs = self.env_wrapper.task_objs
        satisfied_objs = []
        robot_pos = self.robot.get_position()
        for obj in task_objs:
            if hasattr(obj, "category") and obj.category == object_cls:
                satisfied_objs.append(obj)
        if by_dist:
            candidates = sorted(
                satisfied_objs, key=lambda x: cal_dis(x.get_position(), robot_pos)
            )
        else:
            candidates = satisfied_objs.copy()
            np.random.shuffle(candidates)
        for obj in candidates:
            ret, _ = self.goTo_obj(obj)
            ret, _ = self.turnTo_obj(obj)
            return ret, f"Successfully find object category {object_cls}!"
        return (
            False,
            f"No object found for {object_cls} (been to candidates {[obj.name for obj in candidates]}) but did not see the object",
        )

    def cook(self, tool_ref):
        return self._execute(self.PS.COOK, tool_ref)

    def wash(self, tool_ref):
        return self._execute(self.PS.WASH, tool_ref)

    def burn(self, tool_ref):
        return self._execute(self.PS.BURN, tool_ref)

    def cool(self, tool_ref):
        return self._execute(self.PS.COOL, tool_ref)
