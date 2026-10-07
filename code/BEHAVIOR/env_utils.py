# Portions of this file are adapted from VisualAgentBench
# (https://github.com/THUDM/VisualAgentBench), in particular from
# src/server/tasks/omnigibson/vab_omnigibson_src/utils/env_utils.py and
# src/server/tasks/omnigibson/vab_omnigibson_src/utils/actions.py.
# Copyright (c) the VisualAgentBench authors. Licensed under the Apache License, Version 2.0.
# This file has been modified from the original.

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import sys
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_current_dir, "..")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_current_dir = os.path.dirname(os.path.abspath(__file__))
_omnigibson_path = os.path.join(_current_dir, "BEHAVIOR-1K", "OmniGibson")
if os.path.exists(_omnigibson_path) and _omnigibson_path not in sys.path:
    sys.path.insert(0, _omnigibson_path)
import omnigibson as og
import torch as th
from omnigibson.utils.transform_utils import quat2euler, euler2mat, quat2mat
from omnigibson.utils.constants import PrimType
from task.commonsense_knowledge.behavior import IN_RECEPTACLE, ON_RECEPTACLE
from omnigibson.utils.bddl_utils import OBJECT_TAXONOMY


def _zero_object_velocities_if_supported(obj) -> None:
    """Kinematic objects may use RigidKinematicPrim without set_linear_velocity."""
    try:
        obj.set_linear_velocity(velocity=th.zeros(3))
        obj.set_angular_velocity(velocity=th.zeros(3))
    except AttributeError:
        pass


def cal_dis(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)


def distance_to_cuboid(point, cuboid):
    min_corner, max_corner = cuboid
    closest_point = np.clip(point, min_corner, max_corner)
    return np.linalg.norm(point - closest_point)


def quaternion_multiply(q1, q2):
    # calculate the multiply of two quaternion
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([x, y, z, w])


def trans_camera(q):
    random_yaw = np.pi / 2
    yaw_orn = R.from_euler("Z", random_yaw)
    new_camera_orn = quaternion_multiply(yaw_orn.as_quat(), q)
    return new_camera_orn


def transform_camera_euler(q, roll=0, pitch=0, yaw=0, degrees=False):
    rotation = R.from_euler("XYZ", [roll, pitch, yaw], degrees=degrees)

    transform_quat = rotation.as_quat()
    new_camera_orn = quaternion_multiply(transform_quat, q)

    return new_camera_orn


def get_valid_orientation(obj):
    try:
        _, orientation = obj.get_position_orientation()
        if isinstance(orientation, th.Tensor):
            orientation = orientation.cpu().numpy()
        elif not isinstance(orientation, np.ndarray):
            orientation = np.array(orientation)
        if np.any(np.isnan(orientation)) or np.any(np.isinf(orientation)):
            return np.array([0, 0, 0, 1.0])
        norm = np.linalg.norm(orientation)
        if norm < 1e-6:
            return np.array([0, 0, 0, 1.0])
        orientation = orientation / norm

        if orientation[3] < 0:
            orientation = -orientation
        return orientation
    except:
        return np.array([0, 0, 0, 1.0])


def update_obj(robot, obj, all_objs):
    # update objects position according to robot position
    obj_pos = obj.get_position()
    if isinstance(obj_pos, th.Tensor):
        obj_pos = obj_pos.cpu().numpy()
    elif not isinstance(obj_pos, np.ndarray):
        obj_pos = np.array(obj_pos)

    obj_orientation = get_valid_orientation(obj)

    inside_obj_pos = []
    on_top_obj_pos = []
    obj_synset = OBJECT_TAXONOMY.get_synset_from_category(obj.category)
    for o in all_objs:
        if o == obj or not hasattr(o, "states"):
            continue

        if hasattr(obj, "prim_type") and obj.prim_type == PrimType.CLOTH:
            pass
        else:
            if (
                obj_synset in list(IN_RECEPTACLE)
                and og.object_states.Inside in o.states
                and o.states[og.object_states.Inside].get_value(obj)
            ):
                o_pos = o.get_position()

                if isinstance(o_pos, th.Tensor):
                    o_pos = o_pos.cpu().numpy()
                elif not isinstance(o_pos, np.ndarray):
                    o_pos = np.array(o_pos)
                inside_obj_pos.append((o, o_pos - obj_pos))
            if (
                obj_synset in list(ON_RECEPTACLE)
                and og.object_states.OnTop in o.states
                and o.states[og.object_states.OnTop].get_value(obj)
            ):
                o_pos = o.get_position()

                if isinstance(o_pos, th.Tensor):
                    o_pos = o_pos.cpu().numpy()
                elif not isinstance(o_pos, np.ndarray):
                    o_pos = np.array(o_pos)
                on_top_obj_pos.append((o, o_pos - obj_pos))

    robot_pos = robot.get_position()
    robot_pos[2] += robot.aabb_center[2] - 0.2
    obj.set_position_orientation(position=robot_pos, orientation=obj_orientation)
    _zero_object_velocities_if_supported(obj)

    for o, pos in inside_obj_pos:
        if not hasattr(o, "states"):
            continue
        o_orientation = get_valid_orientation(o)
        o.set_position_orientation(position=robot_pos + pos, orientation=o_orientation)

        if hasattr(obj, "prim_type") and obj.prim_type == PrimType.CLOTH:
            continue
        if not o.states[og.object_states.Inside].get_value(obj):
            o.states[og.object_states.Inside].set_value(obj, True)

    for o, pos in on_top_obj_pos:
        if not hasattr(o, "states"):
            continue
        o_orientation = get_valid_orientation(o)
        o.set_position_orientation(position=robot_pos + pos, orientation=o_orientation)

        if hasattr(obj, "prim_type") and obj.prim_type == PrimType.CLOTH:
            continue
        if not o.states[og.object_states.OnTop].get_value(obj):
            o.states[og.object_states.OnTop].set_value(obj, True)
    if hasattr(robot, "keep_still"):
        robot.keep_still()
    for _ in range(3):
        og.sim.step()


def grasp(robot, grasped_obj, obj, all_objs, delta=2.0):
    if len(grasped_obj) > 0:
        return False, "The agent is already holding an object."

    if hasattr(obj, "mass") and obj.mass > 128:
        return False, "The target object is not graspable."

    robot_pos = robot.get_position()
    obj_pos = obj.get_position()
    dis = distance_to_cuboid(robot_pos[:2], (obj.aabb[0][:2], obj.aabb[1][:2]))

    if dis > delta:
        return False, "The target is too far away."

    robot_pos = robot.get_position()
    if isinstance(robot_pos, th.Tensor):
        robot_pos = robot_pos.cpu().numpy()
    elif not isinstance(robot_pos, np.ndarray):
        robot_pos = np.array(robot_pos)
    robot_pos = robot_pos.copy()
    robot_pos[2] += robot.aabb_center[2] - 0.2

    obj_orientation = get_valid_orientation(obj)

    if isinstance(obj_pos, th.Tensor):
        obj_pos = obj_pos.cpu().numpy()
    elif not isinstance(obj_pos, np.ndarray):
        obj_pos = np.array(obj_pos)
    inside_obj_pos = []
    on_top_obj_pos = []
    obj_synset = OBJECT_TAXONOMY.get_synset_from_category(obj.category)
    for o in all_objs:
        if o == obj or not hasattr(o, "states"):
            continue

        if hasattr(obj, "prim_type") and obj.prim_type == PrimType.CLOTH:
            pass
        else:
            if (
                obj_synset in list(IN_RECEPTACLE)
                and og.object_states.Inside in o.states
                and o.states[og.object_states.Inside].get_value(obj)
            ):
                o_pos = o.get_position()

                if isinstance(o_pos, th.Tensor):
                    o_pos = o_pos.cpu().numpy()
                elif not isinstance(o_pos, np.ndarray):
                    o_pos = np.array(o_pos)
                inside_obj_pos.append((o, o_pos - obj_pos))
            if (
                obj_synset in list(ON_RECEPTACLE)
                and og.object_states.OnTop in o.states
                and o.states[og.object_states.OnTop].get_value(obj)
            ):
                o_pos = o.get_position()

                if isinstance(o_pos, th.Tensor):
                    o_pos = o_pos.cpu().numpy()
                elif not isinstance(o_pos, np.ndarray):
                    o_pos = np.array(o_pos)
                on_top_obj_pos.append((o, o_pos - obj_pos))

    obj.set_position_orientation(position=robot_pos, orientation=obj_orientation)

    flag_inside = True
    for o, pos in inside_obj_pos:
        if not hasattr(o, "states"):
            continue
        o_orientation = get_valid_orientation(o)

        if isinstance(pos, th.Tensor):
            pos = pos.cpu().numpy()
        elif not isinstance(pos, np.ndarray):
            pos = np.array(pos)

        pos = np.array(pos).flatten()[:3]
        o.set_position_orientation(position=robot_pos + pos, orientation=o_orientation)

        if hasattr(obj, "prim_type") and obj.prim_type == PrimType.CLOTH:
            continue
        if not o.states[og.object_states.Inside].get_value(obj):
            o.states[og.object_states.Inside]._set_value(obj, True)
            if not o.states[og.object_states.Inside].get_value(obj):
                flag_inside = False

    flag_ontop = True
    for o, pos in on_top_obj_pos:
        if not hasattr(o, "states"):
            continue
        o_orientation = get_valid_orientation(o)

        if isinstance(pos, th.Tensor):
            pos = pos.cpu().numpy()
        elif not isinstance(pos, np.ndarray):
            pos = np.array(pos)

        pos = np.array(pos).flatten()[:3]
        o.set_position_orientation(position=robot_pos + pos, orientation=o_orientation)

        if hasattr(obj, "prim_type") and obj.prim_type == PrimType.CLOTH:
            continue
        if not o.states[og.object_states.OnTop].get_value(obj):
            o.states[og.object_states.OnTop]._set_value(obj, True)
            if not o.states[og.object_states.OnTop].get_value(obj):
                flag_ontop = False

    grasped_obj.append(obj)
    _zero_object_velocities_if_supported(obj)
    if hasattr(robot, "keep_still"):
        robot.keep_still()
    msg = "Grasped successfully!"
    if not flag_ontop:
        msg += " But something on top of the object is left there!"
    if not flag_inside:
        msg += " But something inside the object is left there!"

    return True, msg


def drop_obj(robot, grasped_obj, all_objs, drop_position=None):
    if len(grasped_obj) == 0:
        return False, "The agent is not holding an object."

    obj = grasped_obj[0]

    if drop_position is None:
        robot_pos = robot.get_position()
        robot_quat = robot.get_position_orientation()[1]

        try:
            if isinstance(robot_quat, np.ndarray):
                euler = R.from_quat(robot_quat).as_euler("xyz")
                yaw = euler[2]
            else:
                if isinstance(robot_quat, th.Tensor):
                    robot_quat_np = robot_quat.cpu().numpy()
                else:
                    robot_quat_np = np.array(robot_quat)
                euler = R.from_quat(robot_quat_np).as_euler("xyz")
                yaw = euler[2]
        except:
            euler = quat2euler(robot_quat)
            if isinstance(euler, th.Tensor):
                euler = euler.cpu().numpy()
            else:
                euler = np.array(euler)
            yaw = euler[2]

        robot_pos = np.array(robot_pos)
        drop_position = robot_pos.copy()
        drop_position[0] += 0.5 * np.cos(yaw)
        drop_position[1] += 0.5 * np.sin(yaw)
        drop_position[2] = 0.004

    obj_orientation = get_valid_orientation(obj)

    obj_pos = obj.get_position()

    if isinstance(obj_pos, th.Tensor):
        obj_pos = obj_pos.cpu().numpy()
    elif not isinstance(obj_pos, np.ndarray):
        obj_pos = np.array(obj_pos)
    inside_obj_pos = []
    on_top_obj_pos = []
    obj_synset = OBJECT_TAXONOMY.get_synset_from_category(obj.category)
    for o in all_objs:
        if o == obj or not hasattr(o, "states"):
            continue

        if hasattr(obj, "prim_type") and obj.prim_type == PrimType.CLOTH:
            pass
        else:
            if (
                obj_synset in list(IN_RECEPTACLE)
                and og.object_states.Inside in o.states
                and o.states[og.object_states.Inside].get_value(obj)
            ):
                o_pos = o.get_position()

                if isinstance(o_pos, th.Tensor):
                    o_pos = o_pos.cpu().numpy()
                elif not isinstance(o_pos, np.ndarray):
                    o_pos = np.array(o_pos)
                inside_obj_pos.append((o, o_pos - obj_pos))
            if (
                obj_synset in list(ON_RECEPTACLE)
                and og.object_states.OnTop in o.states
                and o.states[og.object_states.OnTop].get_value(obj)
            ):
                o_pos = o.get_position()

                if isinstance(o_pos, th.Tensor):
                    o_pos = o_pos.cpu().numpy()
                elif not isinstance(o_pos, np.ndarray):
                    o_pos = np.array(o_pos)
                on_top_obj_pos.append((o, o_pos - obj_pos))

    obj.set_position_orientation(position=drop_position, orientation=obj_orientation)

    flag_inside = True
    for o, pos in inside_obj_pos:
        if not hasattr(o, "states"):
            continue
        o_orientation = get_valid_orientation(o)

        if isinstance(pos, th.Tensor):
            pos = pos.cpu().numpy()
        elif not isinstance(pos, np.ndarray):
            pos = np.array(pos)

        pos = np.array(pos).flatten()[:3]
        o.set_position_orientation(
            position=drop_position + pos, orientation=o_orientation
        )

        if hasattr(obj, "prim_type") and obj.prim_type == PrimType.CLOTH:
            continue
        if not o.states[og.object_states.Inside].get_value(obj):
            o.states[og.object_states.Inside]._set_value(obj, True)
            if not o.states[og.object_states.Inside].get_value(obj):
                flag_inside = False

    flag_ontop = True
    for o, pos in on_top_obj_pos:
        if not hasattr(o, "states"):
            continue
        o_orientation = get_valid_orientation(o)

        if isinstance(pos, th.Tensor):
            pos = pos.cpu().numpy()
        elif not isinstance(pos, np.ndarray):
            pos = np.array(pos)

        pos = np.array(pos).flatten()[:3]
        o.set_position_orientation(
            position=drop_position + pos, orientation=o_orientation
        )

        if hasattr(obj, "prim_type") and obj.prim_type == PrimType.CLOTH:
            continue
        if not o.states[og.object_states.OnTop].get_value(obj):
            o.states[og.object_states.OnTop]._set_value(obj, True)
            if not o.states[og.object_states.OnTop].get_value(obj):
                flag_ontop = False

    grasped_obj.clear()
    if hasattr(robot, "keep_still"):
        robot.keep_still()
    msg = "Dropped successfully!"
    if not flag_ontop:
        msg += " But something on top of the object is left there!"
    if not flag_inside:
        msg += " But something inside the object is left there!"
    return True, msg


def get_text_color(background_color):
    if (
        0.213 * background_color[0]
        + 0.715 * background_color[1]
        + 0.072 * background_color[2]
        > 255 / 2
    ):
        return (0, 0, 0)
    else:
        return (255, 255, 255)


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def place_text_boxes(image, rects, texts, colors):
    text_boxes = []
    for rect, text in zip(rects, texts):
        x1, y1, x2, y2 = rect
        w = x2 - x1
        h = y2 - y1
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.64, 2
        )
        positions = []
        if y1 - text_height - 1 >= 0 and x1 + text_width < image.shape[1]:
            positions.append((x1, y1 - text_height - 1))
        if y2 + text_height + 1 < image.shape[0] and x1 + text_width < image.shape[1]:
            positions.append((x1, y2 + 1))
        if x1 - text_width - 1 >= 0:
            positions.append((x1 - text_width, y1))
        if x2 + text_width + 1 < image.shape[1]:
            positions.append((x2 + 1, y1))
        if len(positions) == 0:
            positions = [
                (x1, y1 - text_height - 1),
                (x1, y2 + 1),
                (x1 - text_width - 1, y1),
                (x2 + 1, y1),
            ]

        best_position = None
        best_overlap = float("inf")
        for position in positions:
            text_box = [position[0], position[1], text_width + 1, text_height + 1]
            total_overlap = 0
            for existing_text_box in text_boxes:
                total_overlap += overlap(
                    (text_box[0], text_box[0] + text_box[2]),
                    (existing_text_box[0], existing_text_box[0] + existing_text_box[2]),
                ) * overlap(
                    (text_box[1], text_box[1] + text_box[3]),
                    (existing_text_box[1], existing_text_box[1] + existing_text_box[3]),
                )
            if total_overlap < best_overlap:
                best_overlap = total_overlap
                best_position = position
        text_boxes.append(
            [best_position[0], best_position[1], text_width + 1, text_height + 1]
        )
    for rect, text_box, text, color in zip(rects, text_boxes, texts, colors):
        x1, y1, x2, y2 = rect
        cv2.rectangle(
            image,
            (int(text_box[0]), int(text_box[1])),
            (int(text_box[0] + text_box[2]), int(text_box[1] + text_box[3])),
            color,
            -1,
        )
        cv2.putText(
            image,
            text,
            (int(text_box[0] + 1), int(text_box[1] + text_box[3] - 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            get_text_color(color),
            2,
        )
    return image


def is_digit(n):
    try:
        int(n)
        return True, int(n)
    except ValueError:
        return False, None


def move(robot, obj, pos, grasped_obj, all_objs, orientation=None):

    if isinstance(pos, th.Tensor):
        pos = pos.detach().cpu().numpy()
    else:
        pos = np.asarray(pos)
    if pos.shape[0] < 3:
        return False, "Invalid target position: expected 3D position."
    if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
        return False, f"Invalid target position contains NaN/Inf: {pos}"

    if orientation is None:
        try:
            _, orientation = robot.get_position_orientation()

            if np.any(np.isnan(orientation)) or np.any(np.isinf(orientation)):
                orientation = np.array([0, 0, 0, 1.0])
        except:
            orientation = np.array([0, 0, 0, 1.0])

    if not isinstance(orientation, np.ndarray):
        orientation = np.array(orientation)

    if np.any(np.isnan(orientation)) or np.any(np.isinf(orientation)):
        orientation = np.array([0, 0, 0, 1.0])
    norm = np.linalg.norm(orientation)
    if norm < 1e-6:
        orientation = np.array([0, 0, 0, 1.0])
    else:
        orientation = orientation / norm

        if orientation[3] < 0:
            orientation = -orientation

    robot.set_position_orientation(position=pos, orientation=orientation)

    if len(grasped_obj) > 0:
        update_obj(robot, grasped_obj[0], all_objs)

        robot_pos = robot.get_position()

        if isinstance(robot_pos, th.Tensor):
            robot_pos[2] = th.tensor(
                pos[2], dtype=robot_pos.dtype, device=robot_pos.device
            )
        else:
            robot_pos[2] = pos[2]
        robot.set_position_orientation(position=robot_pos, orientation=orientation)

    if hasattr(robot, "keep_still"):
        robot.keep_still()
    return True, "Moved successfully!"


def get_seg_instance(camera):

    assert camera.initialized, "Camera must be initialized first!"

    if "seg_instance" not in camera.modalities:
        camera.add_modality("seg_instance")

    raw_obs = camera._annotators["seg_instance"].get_data(device=og.sim.device)

    img = raw_obs["data"] if isinstance(raw_obs, dict) else raw_obs
    id_to_labels = (
        raw_obs["info"]["idToLabels"]
        if isinstance(raw_obs, dict) and "info" in raw_obs
        else {}
    )

    if og.sim.device == "cpu":
        img = camera._preprocess_cpu_obs(img, "seg_instance")
    elif "cuda" in og.sim.device:
        img = camera._preprocess_gpu_obs(img, "seg_instance")

    obs_dict = {"seg_instance": img}
    mapping_list = {}
    for key, value in id_to_labels.items():
        if key == "0" or key == "1":
            continue
        mapping_list[int(key) - 1] = (int(key), value)
    return obs_dict["seg_instance"], mapping_list


def validate_and_normalize_quaternion(quat):
    if isinstance(quat, th.Tensor):
        quat = quat.cpu().numpy()
    elif not isinstance(quat, np.ndarray):
        quat = np.array(quat)
    if np.any(np.isnan(quat)) or np.any(np.isinf(quat)):
        return np.array([0.0, 0.0, 0.0, 1.0])
    norm = np.linalg.norm(quat)
    if norm < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0])
    quat_normalized = quat / norm

    if quat_normalized[3] < 0:
        quat_normalized = -quat_normalized
    return quat_normalized


TARGET_NOT_VISIBLE_OR_FAR_MSG = "The target is not visible or is too far away."


def normalize_behavior_feedback(msg: str) -> str:
    if not isinstance(msg, str):
        return msg
    s = msg.strip()
    if not s:
        return s
    if s == TARGET_NOT_VISIBLE_OR_FAR_MSG:
        return TARGET_NOT_VISIBLE_OR_FAR_MSG
    low = s.lower()
    if TARGET_NOT_VISIBLE_OR_FAR_MSG.lower() == low:
        return TARGET_NOT_VISIBLE_OR_FAR_MSG

    if any(
        p in low
        for p in (
            "not visible",
            "cannot see",
            "can't see",
            "not within reach",
            "too far to open or close",
            "too far to open",
            "but did not see the object",
        )
    ):
        return TARGET_NOT_VISIBLE_OR_FAR_MSG
    return msg


def norm_pixel_to_coords(norm_pixel, img_width: int, img_height: int):
    nx, ny = float(norm_pixel[0]), float(norm_pixel[1])
    u = nx * img_width
    v = ny * img_height
    ui = int(np.clip(int(round(u)), 0, img_width - 1))
    vi = int(np.clip(int(round(v)), 0, img_height - 1))
    return u, v, ui, vi


def norm_bbox_to_pixel_xyxy(norm_box, img_width: int, img_height: int):
    x_min = int(float(norm_box[0]) * img_width)
    y_min = int(float(norm_box[1]) * img_height)
    x_max = int(float(norm_box[2]) * img_width)
    y_max = int(float(norm_box[3]) * img_height)
    x_min = max(0, min(x_min, img_width - 1))
    x_max = max(0, min(x_max, img_width - 1))
    y_min = max(0, min(y_min, img_height - 1))
    y_max = max(0, min(y_max, img_height - 1))
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    return x_min, y_min, x_max, y_max


def _numpy_intrinsic_matrix(camera) -> np.ndarray:
    K = camera.intrinsic_matrix
    if hasattr(K, "detach"):
        return K.detach().cpu().numpy().reshape(3, 3)
    return np.asarray(K, dtype=np.float64).reshape(3, 3)


def _replicator_to_optical_point(
    u: float, v: float, depth: float, K: np.ndarray
) -> np.ndarray:
    uv1 = np.array([u, v, 1.0], dtype=np.float64)
    p_repr = float(depth) * (np.linalg.inv(K) @ uv1)
    rot_fix = euler2mat(th.tensor([np.pi, 0.0, 0.0], dtype=th.float32)).cpu().numpy()
    return rot_fix @ p_repr


def ensure_depth_linear_modality(camera) -> None:
    if "depth_linear" not in camera.modalities:
        camera.add_modality("depth_linear")
        camera._post_load()


def read_depth_linear_at_pixel(
    camera, norm_pixel
) -> tuple[float, float, float, int, int] | tuple[None, ...]:
    ensure_depth_linear_modality(camera)
    for _ in range(3):
        og.sim.render()
    observation = camera.get_obs()[0]
    depth_linear = observation.get("depth_linear")
    if depth_linear is None:
        return None, None, None, None, None
    if hasattr(depth_linear, "detach"):
        depth_np = depth_linear.detach().cpu().numpy()
    else:
        depth_np = np.asarray(depth_linear)
    img_h = int(camera.image_height)
    img_w = int(camera.image_width)
    u, v, ui, vi = norm_pixel_to_coords(norm_pixel, img_w, img_h)
    depth_val = float(depth_np[vi, ui])
    if not np.isfinite(depth_val) or depth_val <= 0.0:
        return None, u, v, ui, vi
    return depth_val, u, v, ui, vi


def pixel2D_to_world_point(camera, norm_pixel) -> np.ndarray | None:
    depth_val, u, v, _, _ = read_depth_linear_at_pixel(camera, norm_pixel)
    if depth_val is None:
        return None
    K = _numpy_intrinsic_matrix(camera)
    p_optical = _replicator_to_optical_point(u, v, depth_val, K)
    cam_pos, cam_quat = camera.get_position_orientation()
    if hasattr(cam_pos, "detach"):
        cam_pos = cam_pos.detach().cpu().numpy()
    if hasattr(cam_quat, "detach"):
        cam_quat = cam_quat.detach().cpu().numpy()
    rot = quat2mat(th.as_tensor(cam_quat, dtype=th.float32)).cpu().numpy()
    return rot @ p_optical + np.asarray(cam_pos, dtype=np.float64)


def pixel_pitch_delta_from_intrinsics(v: float, fy: float, cy: float) -> float:
    return float(np.arctan2(v - cy, fy))


def pixel2D_to_obj_at_point(env_wrapper, norm_pixel):
    img, mapping_list = get_seg_instance(env_wrapper.camera)
    img_height, img_width = img.shape
    _, _, ui, vi = norm_pixel_to_coords(norm_pixel, img_width, img_height)
    raw_seg = img[vi, ui]
    seg_id = int(raw_seg.item() if hasattr(raw_seg, "item") else raw_seg)
    if seg_id == 0 or seg_id == 1:
        return None
    if seg_id - 1 not in mapping_list:
        return None
    prim_path = mapping_list[seg_id - 1][1]
    obj = env_wrapper.prim_path_to_obj.get(prim_path, None)
    if obj is None:
        obj = env_wrapper.env.scene.object_registry("prim_path", prim_path)
    return obj
