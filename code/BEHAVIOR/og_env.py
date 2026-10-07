import ast
import numpy as np
import torch as th
import os
import sys
from pathlib import Path
from typing import Any, Iterable
from BEHAVIOR.env_utils import *
from BEHAVIOR.actions import *
from BEHAVIOR.modified_task import TemplateTask
import omnigibson as og
from omnigibson.macros import gm
import yaml
from omnigibson.objects import StatefulObject
from omnigibson.utils.transform_utils import euler2quat
from general_env import Environment

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = True
gm.RENDER_VIEWER_CAMERA = False

CAMERA_ANGLE = 0.3
_MANIP_REACH_XY_M = 2.0
_RENDER_AFTER_SUCCESS_PHYSICS = 3
BOX_TO_ID_IOU_THRESH = 0.1


_OBJECT_UNARY_ACTIONS = frozenset(
    {
        "pick_obj",
        "goTo_obj",
        "placeIn_recep",
        "placeOn_recep",
        "open_recep",
        "open",
        "close_recep",
        "close",
        "turnOn_obj",
        "turnOff_obj",
        "clean_obj",
        "slice_obj",
        "pourTo_recep",
        "spreadTo_obj",
        "turnTo_obj",
    }
)

_TOOL_UNARY_ACTIONS = frozenset(
    {
        "soakWith_tool",
        "fillWith_tool",
        "cookWith_tool",
        "washWith_tool",
        "burnWith_tool",
        "coolWith_tool",
    }
)

_POINT_UNARY_ACTIONS = frozenset({"turnTo_point", "placeTo_point"})


def _flush_sim_render(n: int) -> None:
    for _ in range(max(0, n)):
        og.sim.render()


def _axis_iou_inclusive_xyxy(a, b):
    """IoU bbox"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix1 > ix2 or iy1 > iy2:
        inter = 0
    else:
        inter = (ix2 - ix1 + 1) * (iy2 - iy1 + 1)
    aa = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    bb = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    u = aa + bb - inter
    return inter / u if u > 0 else 0.0


class BehaviorEnv(Environment):
    def __init__(
        self, simulator_name, args, max_steps, max_failed_steps, useSeg, **kwargs
    ):
        self._init_common(
            max_steps=max_steps, max_failed_steps=max_failed_steps, useSeg=useSeg
        )
        self.args = args

        def _coerce_positive_int(v):
            if v is None:
                return None
            iv = int(v)
            if iv <= 0:
                raise ValueError(f"camera resolution must be positive, got {v}")
            return iv

        self.camera_height = _coerce_positive_int(args.get("camera_height")) or 512
        self.camera_width = _coerce_positive_int(args.get("camera_width")) or 512
        config_path = Path(__file__).with_name("behavior_env.yaml")
        with config_path.open(encoding="utf-8") as config_file:
            cfg = yaml.load(config_file, Loader=yaml.FullLoader)
        cfg["scene"]["scene_model"] = args["scene"]
        cfg["scene"]["scene_file"] = args["scene_file"]
        if self.camera_height is not None or self.camera_width is not None:
            for robot_cfg in cfg.get("robots", []):
                sensor_cfg = robot_cfg.setdefault("sensor_config", {})
                vision_cfg = sensor_cfg.setdefault("VisionSensor", {})
                sensor_kwargs = vision_cfg.setdefault("sensor_kwargs", {})
                if self.camera_height is not None:
                    sensor_kwargs["image_height"] = self.camera_height
                if self.camera_width is not None:
                    sensor_kwargs["image_width"] = self.camera_width
        for robot_cfg in cfg.get("robots", []):
            robot_cfg["visual_only"] = True
        self.env = og.Environment(configs=cfg)
        self.func_list = [
            "turnLeft",
            "turnRight",
            "pick_obj",
            "drop",
            "moveAhead",
            "moveLeft",
            "moveRight",
            "moveBack",
            "goTo_obj",
            "placeOn_recep",
            "placeIn_recep",
            "soakWith_tool",
            "clean_obj",
            "slice_obj",
            "open_recep",
            "close_recep",
            "turnOn_obj",
            "turnOff_obj",
            "turnTo_obj",
            "fillWith_tool",
            "pourTo_recep",
            "spreadTo_obj",
            "turnTo_point",
            "placeTo_point",
            "lookUp",
            "lookDown",
            "done",
            "cookWith_tool",
            "washWith_tool",
            "burnWith_tool",
            "coolWith_tool",
        ]
        self.one_arg_funcs = [
            "pick_obj",
            "goTo_obj",
            "open_recep",
            "close_recep",
            "turnOn_obj",
            "turnOff_obj",
            "soakWith_tool",
            "slice_obj",
            "turnTo_obj",
            "placeIn_recep",
            "placeOn_recep",
            "cookWith_tool",
            "washWith_tool",
            "burnWith_tool",
            "pourTo_recep",
            "clean_obj",
            "fillWith_tool",
            "spreadTo_obj",
            "coolWith_tool",
        ]
        self.zero_arg_funcs = [
            "turnLeft",
            "turnRight",
            "drop",
            "moveAhead",
            "moveLeft",
            "moveRight",
            "moveBack",
            "lookUp",
            "lookDown",
            "done",
        ]
        self.point_arg_funcs = ["turnTo_point", "placeTo_point"]
        # Episode-initialized fields (populated in init_episode)
        self.robot = None
        self.action_executer = None
        self.camera = None
        self.no_op = None
        self.robot_orientation = None
        self.robot_camera_height = 1.05
        self.camera_pitch = 0.0
        self.prim_path_to_obj = {}
        self.task_objs = []
        self.task_obj_to_name = {}
        self.moveable_objs = set()
        self.rooms = []
        self.current_room = None
        self.grasped_obj = []
        self.seen_objs = set()
        self.visible_objs = set()
        self.done = False
        self._dropped_obj_for_refill = None

    def init_episode(self):
        self._init_episode_common()
        self.done = False
        self.grasped_obj = []
        self.seen_objs = set()
        self.visible_objs = set()
        self.done = False
        self._dropped_obj_for_refill = None

        self.env.templateTask = TemplateTask(scene=self.env.scene)
        self.robot = self.env.robots[0]
        if hasattr(self.robot, "keep_still"):
            self.robot.keep_still()
        if hasattr(self.robot, "sleep"):
            self.robot.sleep()
        self.action_executer = ExecuteActions(self, self.robot, verbose=False)
        self.camera = self.action_executer.camera
        if self.camera_height is not None:
            self.camera.image_height = self.camera_height
        if self.camera_width is not None:
            self.camera.image_width = self.camera_width
        self.camera.focal_length = 4.8
        if "bbox_2d_tight" not in self.camera.modalities:
            self.camera.add_modality("bbox_2d_tight")
            self.camera._post_load()
        self.no_op = {f"{self.robot.name}": np.zeros(self.robot.action_dim)}
        self.rooms = list(self.env.scene._seg_map.room_ins_id_to_ins_name.values())
        self.prim_path_to_obj = {obj.prim_path: obj for obj in self.env.scene.objects}
        self.task_objs = []
        self.task_obj_to_name = {}
        self.moveable_objs = set()
        fixed_objs = list(self.env.scene.fixed_objects.values())
        for key, value in self.env.templateTask.object_scope.items():
            use_obj = False
            if "agent" not in key:
                use_obj = True
            if not use_obj:
                continue
            if value is None or getattr(value, "unwrapped", None) is None:
                continue
            obj = value.unwrapped
            self.task_objs.append(obj)
            category = key.split(".")[0].split("_")[-1]
            self.task_obj_to_name[obj] = f"{len(self.task_objs)}.{category}"
            if obj not in fixed_objs:
                self.moveable_objs.add(obj)
        try:
            cam_pos = self.camera.get_position()
            cam_pos[2] = self.robot_camera_height
            self.camera.set_position(cam_pos)
        except Exception:
            pass
        try:
            _, quat = self.robot.get_position_orientation()
            self.robot_orientation = validate_and_normalize_quaternion(quat)
        except Exception:
            self.robot_orientation = None
        self.current_room = self.env.scene._seg_map.get_room_instance_by_point(
            self.robot.get_position()[:2]
        )

        og.sim.step()
        for _ in range(3):
            og.sim.render()
        for _ in range(16):
            try:
                _, _, _, _, info = self.env.step(self.no_op)
                self.done = info["done"]["success"]
            except AssertionError as e:
                if "child_values has NoneTypes" in str(e):
                    continue

        self._reset_camera_level_and_pitch()
        obs_dict = self.get_obs()
        additional_info = {
            "depth": obs_dict.get("depth", None),
            "instance_seg_frame": obs_dict.get("instance_seg_frame", None),
            "instance_bboxs": obs_dict.get("instance_bboxs", {}),
            "visible_objects": obs_dict.get("visible_objects", []),
        }
        return obs_dict["rgb"], additional_info

    def _reset_camera_level_and_pitch(self) -> None:
        if self.camera is None:
            return
        try:
            cam_pos, _ = self.camera.get_position_orientation(frame="parent")
            level_quat = euler2quat(th.tensor([0.0, 0.0, 0.0]))
            self.camera.set_position_orientation(
                position=cam_pos, orientation=level_quat, frame="parent"
            )
        except Exception:
            pass
        self.camera_pitch = 0.0
        for _ in range(3):
            og.sim.render()

    @staticmethod
    def _seg_instance_to_numpy(seg_img):
        if isinstance(seg_img, th.Tensor):
            return seg_img.detach().cpu().numpy()
        if hasattr(seg_img, "cpu"):
            return seg_img.cpu().numpy()
        return np.asarray(seg_img)

    def _task_instance_seg_id_to_range(self, mapping, seg_img):
        seg_instance = BehaviorEnv._seg_instance_to_numpy(seg_img)
        count = np.unique(seg_instance, return_counts=True)
        id_to_count = {}
        for _ in range(len(count[0])):
            id, cnt = count[0][_].item(), count[1][_].item()
            id_to_count[id] = cnt
        id_to_range = {}
        for id, cnt in id_to_count.items():
            if id == 0 or id == 1 or cnt < 2:
                continue
            if id - 1 not in mapping or mapping[id - 1][1] not in self.prim_path_to_obj:
                continue
            obj = self.prim_path_to_obj[mapping[id - 1][1]]
            if obj not in self.task_objs:
                continue
            coords = np.where(seg_instance == id)
            x1, y1, x2, y2 = (
                min(coords[1]),
                min(coords[0]),
                max(coords[1]),
                max(coords[0]),
            )
            id_to_range[id] = [x1, y1, x2, y2]
        return id_to_range

    def _build_instance_bboxes(self, seg_img, mapping):
        id_to_range = self._task_instance_seg_id_to_range(mapping, seg_img)
        visible_objects_info = {}
        for id, _range in id_to_range.items():
            if id - 1 not in mapping or mapping[id - 1][1] not in self.prim_path_to_obj:
                continue
            obj = self.prim_path_to_obj[mapping[id - 1][1]]
            name = getattr(obj, "name", None)
            if name is None:
                continue
            visible_objects_info[name] = [
                float(_range[0]),
                float(_range[1]),
                float(_range[2]),
                float(_range[3]),
            ]
        return visible_objects_info

    def get_obs(self):
        objs = set()

        og.sim.step()
        for _ in range(3):
            og.sim.render()
        for __ in range(6):
            _, _, _, _, info = self.env.step(self.no_op)
            self.done = info["done"]["success"]
        observation = self.camera.get_obs()[0]
        rgb = observation["rgb"].detach().cpu().numpy()
        depth = observation.get("depth", None)
        _seg = observation.get("seg_semantic", None)
        if _seg is not None and hasattr(_seg, "detach"):
            instance_seg_frame = _seg.detach().cpu().numpy()
        else:
            instance_seg_frame = _seg
        seg_instance = get_seg_instance(self.camera)
        seg_img = seg_instance[0]
        mapping = seg_instance[1]
        objs = self.get_visible_objects(self.camera)

        self.visible_objs = objs
        self.seen_objs.update(objs)

        grasped_obj = (
            f"{self.task_obj_to_name[self.grasped_obj[0]]}"
            if self.grasped_obj != []
            else None
        )
        self.current_room = self.env.scene._seg_map.get_room_instance_by_point(
            self.robot.get_position()[:2]
        )

        visible_objects_info = self._build_instance_bboxes(seg_img, mapping)
        obs_dict = {
            "rgb": rgb,
            "depth": depth,
            "instance_seg_frame": instance_seg_frame,
            "grasped_obj": grasped_obj,
            "rooms": self.rooms,
            "current_room": self.current_room,
            "done": self.done,
            "num_step": self.num_steps,
            "instance_bboxs": visible_objects_info,
            "visible_objects": self.visible_objs,
        }
        return obs_dict

    def _additional_info_from_obs(self, obs_dict):
        return {
            "depth": obs_dict.get("depth", None),
            "instance_seg_frame": obs_dict.get("instance_seg_frame", None),
            "instance_bboxs": obs_dict.get("instance_bboxs", {}),
            "visible_objects": obs_dict.get("visible_objects", []),
        }

    def _fail_step(self, msg):
        msg = normalize_behavior_feedback(msg)
        o = self.get_obs()
        self.failed_steps += 1
        done = (
            self.num_steps >= self.max_steps
            or self.failed_steps >= self.max_failed_steps
        )
        return o["rgb"], False, msg, done, self._additional_info_from_obs(o)

    @staticmethod
    def _vec_to_np_f(vec):
        if hasattr(vec, "cpu"):
            vec = vec.cpu().numpy()
        return np.asarray(vec, dtype=float)

    def _obj_within_manip_reach_xy(self, obj) -> bool:
        try:
            r = self._vec_to_np_f(self.robot.get_position())[:2]
            o = self._vec_to_np_f(obj.get_position())[:2]
            return float(cal_dis(r, o)) <= _MANIP_REACH_XY_M
        except Exception:
            return True

    @staticmethod
    def _action_arg_value_filled(v) -> bool:
        if v is None:
            return False
        if isinstance(v, str) and not v.strip():
            return False
        if isinstance(v, (list, tuple)) and len(v) == 0:
            return False
        return True

    def _validate_action_args_keys(
        self, action_cls: str, action_args: dict | None
    ) -> str | None:
        if action_args is None:
            action_args = {}
        if not isinstance(action_args, dict):
            return "Invalid action format: action_args must be a dict."
        keys = set(action_args.keys())
        if action_cls in self.zero_arg_funcs or action_cls in ("pass",):
            if keys:
                return f"Action {action_cls!r} takes no arguments; got keys: {sorted(keys)}"
            return None
        if len(keys) != 1:
            return f"Action {action_cls!r} takes too many arguments; got keys: {sorted(keys)}"
        if action_cls in _OBJECT_UNARY_ACTIONS:
            if keys != {"object_ref"}:
                return (
                    f"Action {action_cls!r} requires exactly key 'object_ref'"
                    f"got keys: {sorted(keys)}"
                )
            if not self._action_arg_value_filled(action_args["object_ref"]):
                return f"Action {action_cls!r} requires a non-empty object_ref."
            return None
        if action_cls in _TOOL_UNARY_ACTIONS:
            if keys != {"tool_ref"}:
                return (
                    f"Action {action_cls!r} requires exactly key 'tool_ref'"
                    f"got keys: {sorted(keys)}"
                )
            if not self._action_arg_value_filled(action_args["tool_ref"]):
                return f"Action {action_cls!r} requires a non-empty tool_ref."
            return None
        if action_cls in _POINT_UNARY_ACTIONS:
            if keys != {"target_position_pixel"}:
                return (
                    f"Action {action_cls!r} requires exactly key 'target_position_pixel' "
                    f"got keys: {sorted(keys)}"
                )
            if not self._action_arg_value_filled(action_args["target_position_pixel"]):
                return (
                    f"Action {action_cls!r} requires a non-empty target_position_pixel."
                )
            return None
        return None

    def _visibility_precheck_one_arg(
        self, action_cls: str, action_args: dict
    ) -> str | None:
        if (
            not action_args
            or action_cls in ("goTo_obj", "turnTo_obj")
            or action_cls in self.point_arg_funcs
        ):
            return None
        if action_cls in _OBJECT_UNARY_ACTIONS:
            ref = action_args.get("object_ref")
        elif action_cls in _TOOL_UNARY_ACTIONS:
            ref = action_args.get("tool_ref")
        else:
            return None
        if not self._action_arg_value_filled(ref):
            return TARGET_NOT_VISIBLE_OR_FAR_MSG
        if isinstance(ref, str):
            return None
        vis = self.get_visible_objects(self.camera)
        if ref not in vis and ref not in self.grasped_obj:
            return TARGET_NOT_VISIBLE_OR_FAR_MSG
        if ref not in self.grasped_obj and not self._obj_within_manip_reach_xy(ref):
            return TARGET_NOT_VISIBLE_OR_FAR_MSG
        return None

    def step(self, action_cls: str, action_args: dict, scene_graph=None):
        self.num_steps += 1
        done = self.num_steps >= self.max_steps
        ret, msg = None, None
        if action_args is None:
            action_args = {}
        _key_err = self._validate_action_args_keys(action_cls, action_args)
        if _key_err is not None:
            return self._fail_step(_key_err)
        if action_cls == "turnLeft":
            ret, msg = self.action_executer.turnLeft()
        elif action_cls == "turnRight":
            ret, msg = self.action_executer.turnRight()
        elif action_cls == "lookUp":
            ret, msg = self.action_executer.lookUp(angle=CAMERA_ANGLE)
        elif action_cls == "lookDown":
            ret, msg = self.action_executer.lookDown(angle=CAMERA_ANGLE)
        elif action_cls == "moveAhead":
            _, original_orientation = self.robot.get_position_orientation()
            original_orientation = validate_and_normalize_quaternion(
                original_orientation
            )
            ret, msg = self.action_executer.moveAhead()
            if ret:
                if len(self.grasped_obj) > 0:
                    update_obj(self.robot, self.grasped_obj[0], self.task_objs)
                robot_pos = self.robot.get_position()
                robot_pos[2] = 0.004
                original_orientation = validate_and_normalize_quaternion(
                    original_orientation
                )
                self.robot.set_position_orientation(
                    position=robot_pos, orientation=original_orientation
                )
                self.robot_orientation = original_orientation
                if (
                    self.env.scene._seg_map.get_room_instance_by_point(
                        self.robot.get_position()[:2]
                    )
                    != self.current_room
                ):
                    ret, msg = False, "Collision happened when moving ahead"
        elif action_cls == "moveBack":
            _, original_orientation = self.robot.get_position_orientation()
            original_orientation = validate_and_normalize_quaternion(
                original_orientation
            )
            ret, msg = self.action_executer.moveBack()
            if ret:
                if len(self.grasped_obj) > 0:
                    update_obj(self.robot, self.grasped_obj[0], self.task_objs)
                robot_pos = self.robot.get_position()
                robot_pos[2] = 0.004
                original_orientation = validate_and_normalize_quaternion(
                    original_orientation
                )
                self.robot.set_position_orientation(
                    position=robot_pos, orientation=original_orientation
                )
                self.robot_orientation = original_orientation
                if (
                    self.env.scene._seg_map.get_room_instance_by_point(
                        self.robot.get_position()[:2]
                    )
                    != self.current_room
                ):
                    ret, msg = False, "Collision happened when moving back"
        elif action_cls == "moveLeft":
            _, original_orientation = self.robot.get_position_orientation()
            original_orientation = validate_and_normalize_quaternion(
                original_orientation
            )
            ret, msg = self.action_executer.moveLeft()
            if ret:
                if len(self.grasped_obj) > 0:
                    update_obj(self.robot, self.grasped_obj[0], self.task_objs)
                robot_pos = self.robot.get_position()
                robot_pos[2] = 0.004
                original_orientation = validate_and_normalize_quaternion(
                    original_orientation
                )
                self.robot.set_position_orientation(
                    position=robot_pos, orientation=original_orientation
                )
                self.robot_orientation = original_orientation
                if (
                    self.env.scene._seg_map.get_room_instance_by_point(
                        self.robot.get_position()[:2]
                    )
                    != self.current_room
                ):
                    ret, msg = False, "Collision happened when moving left"
        elif action_cls == "moveRight":
            _, original_orientation = self.robot.get_position_orientation()
            original_orientation = validate_and_normalize_quaternion(
                original_orientation
            )
            ret, msg = self.action_executer.moveRight()
            if ret:
                if len(self.grasped_obj) > 0:
                    update_obj(self.robot, self.grasped_obj[0], self.task_objs)
                robot_pos = self.robot.get_position()
                robot_pos[2] = 0.004
                original_orientation = validate_and_normalize_quaternion(
                    original_orientation
                )
                self.robot.set_position_orientation(
                    position=robot_pos, orientation=original_orientation
                )
                self.robot_orientation = original_orientation
                if (
                    self.env.scene._seg_map.get_room_instance_by_point(
                        self.robot.get_position()[:2]
                    )
                    != self.current_room
                ):
                    ret, msg = False, "Collision happened when moving right"
        elif action_cls == "drop":
            robot_pos, original_orientation = self.robot.get_position_orientation()
            original_orientation = validate_and_normalize_quaternion(
                original_orientation
            )
            dropped_obj = self.grasped_obj[0] if len(self.grasped_obj) > 0 else None
            ret, msg = self.action_executer.drop()
            if ret:
                if len(self.grasped_obj) > 0:
                    self.grasped_obj = []

                self._dropped_obj_for_refill = dropped_obj
                if isinstance(robot_pos, th.Tensor):
                    robot_pos = robot_pos.cpu().numpy()
                elif not isinstance(robot_pos, np.ndarray):
                    robot_pos = np.array(robot_pos)
                robot_pos = robot_pos.copy()
                original_orientation = validate_and_normalize_quaternion(
                    original_orientation
                )
                self.robot.set_position_orientation(
                    position=robot_pos, orientation=original_orientation
                )
                self.robot_orientation = original_orientation
            else:
                self._dropped_obj_for_refill = None
        elif action_cls == "done":
            done = True
            self.done = True
            ret, msg = True, "The task is done!"
        elif action_cls in self.one_arg_funcs:
            if not action_args or not isinstance(action_args, dict):
                return self._fail_step("Invalid action format!")
            for key, value in action_args.items():
                obj_or_not, value_obj = self.resolve_object(value)
                if not obj_or_not:
                    return self._fail_step(value_obj)
                action_args[key] = value_obj
                if isinstance(value_obj, str) and action_cls != "goTo_obj":
                    return self._fail_step("Room name is not object")

            _vis_msg = self._visibility_precheck_one_arg(action_cls, action_args)
            if _vis_msg is not None:
                return self._fail_step(_vis_msg)

            if action_cls == "pick_obj":
                robot_pos, original_orientation = self.robot.get_position_orientation()
                original_orientation = validate_and_normalize_quaternion(
                    original_orientation
                )
                ret, msg = self.action_executer.pick_obj(**action_args)
                if ret:
                    if action_args["object_ref"] not in self.grasped_obj:
                        self.grasped_obj = (
                            [action_args["object_ref"]]
                            if action_args["object_ref"] in self.task_objs
                            else []
                        )
                    if isinstance(robot_pos, th.Tensor):
                        robot_pos = robot_pos.cpu().numpy()
                    elif not isinstance(robot_pos, np.ndarray):
                        robot_pos = np.array(robot_pos)
                    robot_pos = robot_pos.copy()
                    original_orientation = validate_and_normalize_quaternion(
                        original_orientation
                    )
                    self.robot.set_position_orientation(
                        position=robot_pos, orientation=original_orientation
                    )
                    self.robot_orientation = original_orientation
                    if hasattr(self.robot, "keep_still"):
                        self.robot.keep_still()
            elif action_cls == "goTo_obj":
                ret, msg = self.action_executer.goTo_obj(**action_args)
                ref = action_args["object_ref"]
                if ret and not isinstance(ref, str):
                    ret, msg = self.action_executer.turnTo_obj(**action_args)
                if ret:
                    if hasattr(self.robot, "keep_still"):
                        self.robot.keep_still()
            elif action_cls == "placeIn_recep":
                dropped_obj = self.grasped_obj[0] if len(self.grasped_obj) > 0 else None
                ret, msg = self.action_executer.place_inside(**action_args)
                if ret:
                    self.grasped_obj = []
                    self._dropped_obj_for_refill = dropped_obj
                else:
                    self._dropped_obj_for_refill = None
            elif action_cls == "placeOn_recep":
                dropped_obj = self.grasped_obj[0] if len(self.grasped_obj) > 0 else None
                ret, msg = self.action_executer.place_on_top(**action_args)
                if ret:
                    self._dropped_obj_for_refill = dropped_obj
                    self.grasped_obj = []
                else:
                    self._dropped_obj_for_refill = None
            elif action_cls == "open_recep" or action_cls == "open":
                ret, msg = self.action_executer.open_obj(**action_args)
            elif action_cls == "close_recep" or action_cls == "close":
                ret, msg = self.action_executer.close_obj(**action_args)
            elif action_cls == "turnOn_obj":
                ret, msg = self.action_executer.turnOn_obj(**action_args)
            elif action_cls == "turnOff_obj":
                ret, msg = self.action_executer.turnOff_obj(**action_args)
            elif action_cls == "soakWith_tool":
                ret, msg = self.action_executer.soak_with_tool(**action_args)
            elif action_cls == "clean_obj":
                ret, msg = self.action_executer.clean_obj(**action_args)
            elif action_cls == "slice_obj":
                prim_paths_before = frozenset(
                    o.prim_path for o in self.env.scene.objects
                )
                ret, msg = self.action_executer.cut(**action_args)
                if ret:
                    self._sync_objects_after_slice(prim_paths_before)
            elif action_cls == "turnTo_obj":
                ret, msg = self.action_executer.turnTo_obj(**action_args)
            elif action_cls == "fillWith_tool":
                ret, msg = self.action_executer.fill_with(**action_args)
            elif action_cls == "pourTo_recep":
                ret, msg = self.action_executer.pour_into(**action_args)
            elif action_cls == "spreadTo_obj":
                ret, msg = self.action_executer.spread(**action_args)
            elif action_cls == "cookWith_tool":
                ret, msg = self.action_executer.cook(**action_args)
            elif action_cls == "washWith_tool":
                ret, msg = self.action_executer.wash(**action_args)
            elif action_cls == "burnWith_tool":
                ret, msg = self.action_executer.burn(**action_args)
            elif action_cls == "coolWith_tool":
                ret, msg = self.action_executer.cool(**action_args)
            else:
                return self._fail_step(
                    f"Invalid action! You should call the function in the Predefined Action List!"
                )

        elif action_cls in self.point_arg_funcs:
            if not action_args or not isinstance(action_args, dict):
                return self._fail_step("Invalid action format!")
            for key, value in action_args.items():
                if isinstance(value, str):
                    try:
                        action_args[key] = ast.literal_eval(value)
                    except (SyntaxError, ValueError):
                        return self._fail_step(
                            f"Invalid action format: {key!r} must be a literal."
                        )
            if action_cls == "turnTo_point":
                ret, msg = self.action_executer.turnTo_point(**action_args)
            elif action_cls == "placeTo_point":
                dropped_obj = self.grasped_obj[0] if len(self.grasped_obj) > 0 else None
                ret, msg = self.action_executer.placeTo_point(**action_args)
                if ret:
                    self._dropped_obj_for_refill = dropped_obj
                    self.grasped_obj = []
                else:
                    self._dropped_obj_for_refill = None
        elif action_cls == "pass":
            ret, msg = True, ""
        else:
            return self._fail_step("Wrong action format!")
        if ret:
            robot_pos = self.robot.get_position()
            robot_pos[2] = 0.004
            robot_orientation = self.robot.get_orientation()
            robot_orientation = validate_and_normalize_quaternion(robot_orientation)
            self.robot.set_position_orientation(
                position=robot_pos, orientation=robot_orientation
            )
            self.robot_orientation = robot_orientation
            if action_cls in [
                "goTo_obj",
                "moveAhead",
                "moveBack",
                "moveLeft",
                "moveRight",
                "pick_obj",
            ]:
                if len(self.grasped_obj) > 0 and isinstance(
                    self.grasped_obj[0], StatefulObject
                ):
                    self.action_executer.primitives._refill_container(
                        self.grasped_obj[0]
                    )
            elif action_cls in [
                "drop",
                "placeIn_recep",
                "placeOn_recep",
                "placeTo_point",
            ]:
                if self._dropped_obj_for_refill is not None and isinstance(
                    self._dropped_obj_for_refill, StatefulObject
                ):
                    self.action_executer.primitives._refill_container(
                        self._dropped_obj_for_refill
                    )
                self._dropped_obj_for_refill = None
            og.sim.step()
            og.sim.step_physics()
            _flush_sim_render(_RENDER_AFTER_SUCCESS_PHYSICS)
            msg = ""
        else:
            msg = normalize_behavior_feedback(msg)
            self.failed_steps += 1
        obs_dict = self.get_obs()
        done = done or self.failed_steps >= self.max_failed_steps
        return obs_dict["rgb"], ret, msg, done, self._additional_info_from_obs(obs_dict)

    def _sync_objects_after_slice(self, prim_paths_before: frozenset) -> None:
        self.prim_path_to_obj = {obj.prim_path: obj for obj in self.env.scene.objects}
        try:
            fixed_objs = frozenset(self.env.scene.fixed_objects.values())
        except Exception:
            fixed_objs = frozenset()
        task_set = set(self.task_objs)
        for obj in self.env.scene.objects:
            if obj.prim_path in prim_paths_before:
                continue
            if obj in task_set:
                continue
            self.task_objs.append(obj)
            task_set.add(obj)
            category = getattr(obj, "category", "object")
            self.task_obj_to_name[obj] = f"{len(self.task_objs)}.{category}"
            if obj not in fixed_objs:
                self.moveable_objs.add(obj)

    def expand_dynamic_object_references(
        self,
        object_reference: dict[str, list[Any]],
        current_object_ids: Iterable[Any],
        goal_categories: set[str] | None = None,
    ) -> dict[str, list[Any]]:
        """Bind sliced nodes to the referenced whole-object lineage.

        OmniGibson's SlicingRule removes the source and creates objects named
        ``half_{source.name}_{i}``.  These names are already BEHAVIOR USG IDs.
        """
        expanded = super().expand_dynamic_object_references(
            object_reference, current_object_ids, goal_categories
        )
        if goal_categories is None:
            sliced_categories = {
                f"{category}Sliced" for category in object_reference
            }
        else:
            sliced_categories = {
                category for category in goal_categories if category.endswith("Sliced")
            }

        current_ids = tuple(current_object_ids)
        for sliced_category in sliced_categories:
            whole_category = sliced_category[: -len("Sliced")]
            if whole_category not in object_reference:
                continue

            derived_ids = list(expanded.get(sliced_category, []))
            seen_ids = set(derived_ids)
            for whole_id in object_reference[whole_category]:
                descendant_prefix = f"half_{whole_id}_"
                for object_id in current_ids:
                    if (
                        isinstance(object_id, str)
                        and object_id.startswith(descendant_prefix)
                        and object_id not in seen_ids
                    ):
                        derived_ids.append(object_id)
                        seen_ids.add(object_id)
            expanded[sliced_category] = derived_ids
        return expanded

    def build_scene_context(self, entities: Iterable[Any]) -> dict[str, Any]:
        entities = tuple(entities)
        context = super().build_scene_context(entities)
        wall_ids = {
            entity.ID for entity in entities if entity.category == "walls"
        }
        context["usg_big_objects_list"] = [
            object_id
            for object_id in context["usg_big_objects_list"]
            if object_id not in wall_ids
        ]
        return context

    def get_visible_objects(self, camera):
        instance_image, instance_id_to_prim = get_seg_instance(camera)
        task_objects = set(self.task_objs)
        visible_objects = set()

        for instance_id in np.unique(instance_image):
            instance_id = int(instance_id)
            if instance_id <= 1:
                continue

            mapping_entry = instance_id_to_prim.get(instance_id - 1)
            if mapping_entry is None:
                continue

            obj = self.prim_path_to_obj.get(mapping_entry[1])
            if obj in task_objects:
                visible_objects.add(obj)

        return visible_objects

    def resolve_object(self, object_ref):

        if isinstance(object_ref, str):
            if object_ref in self.action_executer.get_all_room_names():
                return True, object_ref
            candidates = [
                o
                for o in self.env.scene.objects
                if getattr(o, "name", None) == object_ref
            ]
            if len(candidates) == 0:
                return False, f"Object '{object_ref}' not found in the scene!"
            if len(candidates) > 1:
                return False, f"Object name '{object_ref}' is ambiguous in the scene!"
            obj = candidates[0]
            return True, obj
        elif isinstance(object_ref, list) and len(object_ref) > 0:
            if not all(isinstance(x, (int, float)) for x in object_ref):
                return (
                    False,
                    f"List object reference must contain only numbers: {object_ref}",
                )

            object_ref = [float(x) for x in object_ref]
            if len(object_ref) == 2:
                obj = self.pixel2D_to_obj(object_ref)
                if obj is not None:
                    return True, obj
                else:
                    return False, "The target pixel does not correspond to the object"
            elif len(object_ref) == 4:
                obj, bbox_err = self.bbox2D_to_obj(object_ref)
                if obj is not None:
                    return True, obj
                return False, bbox_err
            else:
                return (
                    False,
                    f"List object reference must have length 2 (pixel2D) or 4 (bbox2D), got {len(object_ref)}: {object_ref}",
                )
        else:
            return (
                False,
                f"Unknown object reference type: {type(object_ref)}, value: {object_ref}",
            )

    def pixel2D_to_obj(self, object_ref):
        img, mapping_list = get_seg_instance(self.camera)
        img_height, img_width = img.shape
        _, _, ui, vi = norm_pixel_to_coords(object_ref, img_width, img_height)
        seg_id = int(img[vi, ui])

        if seg_id == 0 or seg_id == 1:
            return None
        if seg_id - 1 not in mapping_list:
            return None
        prim_path = mapping_list[seg_id - 1][1]
        return self.prim_path_to_obj.get(prim_path, None)

    def bbox2D_to_obj(self, object_ref):
        img, mapping_list = get_seg_instance(self.camera)
        h, w = img.shape[:2]
        pred_box = norm_bbox_to_pixel_xyxy(object_ref, w, h)
        best_iou, best_path = -1.0, None
        for seg_id in np.unique(img):
            sid = int(seg_id)
            if sid in (0, 1):
                continue
            if sid - 1 not in mapping_list:
                continue
            prim_path = mapping_list[sid - 1][1]
            ys, xs = np.where(img == sid)
            if xs.size == 0:
                continue
            inst_box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            iou = _axis_iou_inclusive_xyxy(pred_box, inst_box)
            if iou > best_iou:
                best_iou, best_path = iou, prim_path
        if best_path is None:
            return (
                None,
                "No mapped object in predicted 2D bbox (segmentation mapping missing).",
            )
        if best_iou <= BOX_TO_ID_IOU_THRESH:
            return (
                None,
                f"Best IoU={best_iou:.3f} below threshold {BOX_TO_ID_IOU_THRESH}.",
            )
        obj = self.prim_path_to_obj.get(best_path)
        if obj is None:
            return (
                None,
                f"Best IoU={best_iou:.3f} but prim_path is not a known object in this task.",
            )
        return obj, ""

    def close(self):
        og.shutdown()
