from typing import List, Tuple, Optional, Union
from collections import defaultdict
import re
import math
import copy

import numpy as np
import habitat_sim
from habitat_sim.simulator import Simulator
from habitat_sim.physics import ManagedArticulatedObject
import habitat.sims.habitat_simulator.sim_utilities as sutils
from habitat.datasets.rearrange.samplers.receptacle import Receptacle
import magnum as mn

from scene_graph.unified_scene_graph import UnifiedSceneGraph
from scene_graph.graph_schema import (
    RelationType,
    PropertyType,
    StateType,
    flip_edge,
)
from Hab.hab_action_utils import (
    calculate_distance_2D,
    sample_position_on_receptacle_links_with_reference,
    sample_position_on_room_floor_with_reference,
    place_to_position_on_receptacle_link,
    place_to_position_on_room_floor,
    bind_object_to_parent_link,
)


MAX_PITCH_UP = 50
MIN_PICTH_DOWN = -50
NUM_GRID_POINT_IN_BOX = 5
BOX_TO_ID_IOU_THRESH = 0.1
MIN_DIST_TO_OBJ_FOR_TELEPORT = 0.5
DIST_SCALE_FACTOR_FOR_TELEPORT = 5.0
MANIPULATION_DISTANCE_THRESHOLD = 1.5


def resolve_object(
    obj_ref: Union[str, List[float]],
    sim: Simulator,
    action_inputs: dict,
    force_whole_object: bool = True,
):
    if isinstance(obj_ref, str):
        obj_id = re.search(r"(\d+)$", obj_ref)
        return int(obj_id.group(1)) if obj_id else None
    elif isinstance(obj_ref, List) and isinstance(obj_ref[0], float):
        if len(obj_ref) == 2:
            return pixel_to_object_id(
                sim, obj_ref, action_inputs, None, force_whole_object
            )
        elif len(obj_ref) == 4:
            return box_to_object_id(
                sim, obj_ref, action_inputs, None, force_whole_object
            )
        else:
            raise ValueError(f"Unknown object reference: {obj_ref}")
    elif isinstance(obj_ref, List) and isinstance(obj_ref[0], str):
        if len(obj_ref[1]) == 2:
            return pixel_to_object_id(
                sim, obj_ref[1], action_inputs, obj_ref[0], force_whole_object
            )
        elif len(obj_ref[1]) == 4:
            return box_to_object_id(
                sim, obj_ref[1], action_inputs, obj_ref[0], force_whole_object
            )
        else:
            raise ValueError(f"Unknown object reference: {obj_ref}")
    else:
        raise ValueError(f"Unknown object reference: {obj_ref}")


def pixel_to_object_id(
    sim: Simulator,
    pixel: List[float],
    action_inputs: dict,
    obj_cls: Optional[str] = None,
    force_whole_object: bool = True,
):
    pixel_x, pixel_y = pixel

    seg_img = sim.get_sensor_observations()["semantic_sensor"]
    img_h, img_w = seg_img.shape[:2]

    object_id = seg_img[int(pixel_y * img_h), int(pixel_x * img_w)]

    if force_whole_object:
        if object_id in action_inputs["link_object_id_to_object_id"]:
            object_id = action_inputs["link_object_id_to_object_id"][object_id]

    return object_id


def _norm_bbox_to_pixel_xyxy(
    norm_box: List[float], img_width: int, img_height: int
) -> Tuple[int, int, int, int]:
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


def _axis_iou_inclusive_xyxy(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> float:
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


def _hab_seg_id_matches_cls(
    seg_id: int, obj_cls: Optional[str], action_inputs: dict
) -> bool:
    if obj_cls is None:
        return True
    if seg_id in action_inputs["object_id_to_class"]:
        return match_object_cls(obj_cls, action_inputs["object_id_to_class"][seg_id])
    link_object_id_to_class = action_inputs.get("link_object_id_to_class")
    if link_object_id_to_class and seg_id in link_object_id_to_class:
        category = link_object_id_to_class[seg_id].split("..")[0]
        return match_object_cls(obj_cls, category)
    if seg_id in action_inputs["link_object_id_to_object_id"]:
        parent_id = action_inputs["link_object_id_to_object_id"][seg_id]
        return match_object_cls(obj_cls, action_inputs["object_id_to_class"][parent_id])
    return False


def _is_valid_hab_seg_id(seg_id: int, action_inputs: dict) -> bool:
    if seg_id in action_inputs["object_id_to_class"]:
        return "unknown" not in action_inputs["object_id_to_class"][seg_id]
    link_object_id_to_class = action_inputs.get("link_object_id_to_class")
    if link_object_id_to_class and seg_id in link_object_id_to_class:
        return "unknown" not in link_object_id_to_class[seg_id]
    return seg_id in action_inputs["link_object_id_to_object_id"]


def _build_visible_instance_bboxs(
    sim: Simulator,
    action_inputs: dict,
) -> Tuple[dict[int, Tuple[int, int, int, int]], int, int]:
    seg = sim.get_sensor_observations()["semantic_sensor"]
    img_h, img_w = seg.shape[:2]

    instance_bboxs = {}
    for seg_id in np.unique(seg):
        seg_id = int(seg_id)
        if seg_id in [0, 10000]:
            continue
        if not _is_valid_hab_seg_id(seg_id, action_inputs):
            continue

        ys, xs = np.where(seg == seg_id)
        if len(xs) == 0:
            continue
        instance_bboxs[seg_id] = (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        )
    return instance_bboxs, img_h, img_w


def box_to_object_id(
    sim: Simulator,
    box: List[float],
    action_inputs: dict,
    obj_cls: Optional[str] = None,
    force_whole_object: bool = True,
):

    instance_bboxs, img_h, img_w = _build_visible_instance_bboxs(sim, action_inputs)
    pred_box = _norm_bbox_to_pixel_xyxy(box, img_w, img_h)

    best_iou = -1.0
    best_seg_id = None
    for seg_id, gt_box in instance_bboxs.items():
        if not _hab_seg_id_matches_cls(seg_id, obj_cls, action_inputs):
            continue
        iou = _axis_iou_inclusive_xyxy(pred_box, gt_box)
        if iou > best_iou:
            best_iou = iou
            best_seg_id = seg_id

    if best_seg_id is None or best_iou <= BOX_TO_ID_IOU_THRESH:
        return None

    object_id = best_seg_id
    if force_whole_object and object_id in action_inputs["link_object_id_to_object_id"]:
        object_id = action_inputs["link_object_id_to_object_id"][object_id]
    return object_id


def resolve_all_ids(
    obj_ref: Union[str, List[float]],
    sim: Simulator,
    action_inputs: dict,
    force_whole_object: bool = True,
):
    seg_img = sim.get_sensor_observations()["semantic_sensor"]
    img_h, img_w = seg_img.shape[:2]
    if isinstance(obj_ref, List) and isinstance(obj_ref[0], float):
        if len(obj_ref) == 2:
            pixel_x, pixel_y = obj_ref
            object_ids = [seg_img[int(pixel_y * img_h), int(pixel_x * img_w)]]
            counts = [1.0]
        elif len(obj_ref) == 4:
            x_min, y_min, x_max, y_max = obj_ref
            object_ids, counts = np.unique(
                seg_img[
                    int(y_min * img_h) : int(y_max * img_h),
                    int(x_min * img_w) : int(x_max * img_w),
                ],
                return_counts=True,
            )
        else:
            raise ValueError(f"Unknown object reference: {obj_ref}")
    elif isinstance(obj_ref, List) and isinstance(obj_ref[0], str):
        raise ValueError(f"Unknown object reference: {obj_ref}")
    else:
        raise ValueError(f"Unknown object reference: {obj_ref}")

    freq = dict(zip(object_ids, counts))
    res = set()
    res_freq = defaultdict(int)
    for i in object_ids:
        ii = i
        if force_whole_object:
            if i in action_inputs["link_object_id_to_object_id"]:
                ii = action_inputs["link_object_id_to_object_id"][i]
        res.add(ii)
        res_freq[ii] += freq[i]
    return res, res_freq


def match_object_cls(query_cls: str, obj_ID: str):
    return query_cls.lower().replace(" ", "") == obj_ID.split("|")[0].lower().replace(
        " ", ""
    )


def clamp_rot_pitch(agent, rotation_q):

    current_forward = rotation_q.transform_vector(mn.Vector3(0, 0, -1))

    forward_flat = mn.Vector3(current_forward.x, 0, current_forward.z)

    if forward_flat.length() < 1e-4:
        yaw_q = mn.Quaternion.from_matrix(agent.scene_node.transformation.rotation())

        yaw_angle = mn.Rad(0.0)
    else:
        forward_flat = forward_flat.normalized()

        # "Target" is forward_flat, "Up" is Y
        yaw_matrix = mn.Matrix4.look_at(
            mn.Vector3(0, 0, 0), forward_flat, mn.Vector3(0, 1, 0)
        ).rotation()

        yaw_q = mn.Quaternion.from_matrix(yaw_matrix)

    angle_to_up = mn.math.angle(current_forward, mn.Vector3(0, 1, 0))
    current_pitch_deg = 90.0 - float(mn.Deg(angle_to_up))

    if current_pitch_deg > MAX_PITCH_UP:
        msg = "Pitch is clipped when turning up"
    elif current_pitch_deg < MIN_PICTH_DOWN:
        msg = "Pitch is clipped when turning down"
    else:
        msg = ""

    clamped_pitch_deg = max(min(current_pitch_deg, MAX_PITCH_UP), MIN_PICTH_DOWN)

    pitch_q = mn.Quaternion.rotation(mn.Deg(clamped_pitch_deg), mn.Vector3(1, 0, 0))

    final_q = yaw_q * pitch_q

    return final_q, msg


def object_in_sight(sim: Simulator, unique_ids_in_seg_map: set[int], object_id: int):
    if object_id in unique_ids_in_seg_map:
        return True
    aom = sim.get_articulated_object_manager()
    object = aom.get_object_by_id(object_id)
    if object is None:
        return False
    for x in object.link_object_ids:
        if x in unique_ids_in_seg_map:
            return True
    return False


def transfer_grasped_object(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    gripper_idx: int,
    usg: UnifiedSceneGraph,
):
    in_hand_objects = usg.graph.get_neighbors_of_relation(
        "robot_agent", RelationType.GRASPING
    )
    if len(in_hand_objects) == 0:
        return
    assert len(in_hand_objects) == 1, [x.ID for x in in_hand_objects]
    in_hand_object_node = in_hand_objects[0]
    in_hand_object_id = int(in_hand_object_node.ID.split("_")[-1])

    rom = sim.get_rigid_object_manager()
    in_hand_object = rom.get_object_by_id(in_hand_object_id)
    assert in_hand_object is not None, [in_hand_object_node.ID]

    ee = robot.get_link_scene_node(gripper_idx)
    in_hand_object.translation = ee.absolute_translation

    sim.step_physics(1.0)


# low-level


def moveAhead(sim: Simulator, robot: ManagedArticulatedObject, action_inputs: dict):
    agent = sim.agents[0]
    discrete_action = agent.agent_config.action_space["move_forward"]

    move_vec_world = robot.transformation.transform_vector(
        mn.Vector3(discrete_action.actuation.amount, 0, 0)
    )
    target_pos = robot.translation + move_vec_world

    final_pos = sim.pathfinder.try_step(robot.translation, target_pos)

    moved_dist = (mn.Vector3(final_pos) - robot.translation).length()
    robot.translation = final_pos

    sim.step_physics(1.0)

    success = moved_dist > 0.01
    return success, "" if success else "The movement is blocked."


def moveBack(sim: Simulator, robot: ManagedArticulatedObject, action_inputs: dict):
    agent = sim.agents[0]
    discrete_action = agent.agent_config.action_space["move_backward"]

    move_vec_world = robot.transformation.transform_vector(
        mn.Vector3(-discrete_action.actuation.amount, 0, 0)
    )
    target_pos = robot.translation + move_vec_world

    final_pos = sim.pathfinder.try_step(robot.translation, target_pos)

    moved_dist = (mn.Vector3(final_pos) - robot.translation).length()
    robot.translation = final_pos

    sim.step_physics(1.0)

    success = moved_dist > 0.01
    return success, "" if success else "The movement is blocked."


def moveLeft(sim: Simulator, robot: ManagedArticulatedObject, action_inputs: dict):
    agent = sim.agents[0]
    discrete_action = agent.agent_config.action_space["move_left"]

    move_vec_world = robot.transformation.transform_vector(
        mn.Vector3(0, 0, -discrete_action.actuation.amount)
    )
    target_pos = robot.translation + move_vec_world

    final_pos = sim.pathfinder.try_step(robot.translation, target_pos)

    moved_dist = (mn.Vector3(final_pos) - robot.translation).length()
    robot.translation = final_pos

    sim.step_physics(1.0)

    success = moved_dist > 0.01
    return success, "" if success else "The movement is blocked."


def moveRight(sim: Simulator, robot: ManagedArticulatedObject, action_inputs: dict):
    agent = sim.agents[0]
    discrete_action = agent.agent_config.action_space["move_right"]

    move_vec_world = robot.transformation.transform_vector(
        mn.Vector3(0, 0, discrete_action.actuation.amount)
    )
    target_pos = robot.translation + move_vec_world

    final_pos = sim.pathfinder.try_step(robot.translation, target_pos)

    moved_dist = (mn.Vector3(final_pos) - robot.translation).length()
    robot.translation = final_pos

    sim.step_physics(1.0)

    success = moved_dist > 0.01
    return success, "" if success else "The movement is blocked."


def turnRight(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    action_inputs: dict,
    delta_rad: Optional[float] = None,
):
    agent = sim.agents[0]
    discrete_action = agent.agent_config.action_space["turn_right"]
    if delta_rad is None:
        delta_rad = discrete_action.actuation.amount
    robot.rotate_y(mn.Rad(mn.Deg(-delta_rad)))

    sim.step_physics(1.0)
    return True, ""


def turnLeft(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    action_inputs: dict,
    delta_rad: Optional[float] = None,
):
    agent = sim.agents[0]
    discrete_action = agent.agent_config.action_space["turn_left"]
    if delta_rad is None:
        delta_rad = discrete_action.actuation.amount
    robot.rotate_y(mn.Rad(mn.Deg(delta_rad)))

    sim.step_physics(1.0)
    return True, ""


def lookLeft(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    action_inputs: dict,
    delta_rad: Optional[float] = None,
):
    pan_idx = action_inputs["pan_idx"]
    agent = sim.agents[0]
    discrete_action = agent.agent_config.action_space["turn_left"]

    joint_positions = robot.joint_positions
    current_pan = joint_positions[pan_idx]
    if delta_rad is None:
        delta_rad = float(mn.Rad(mn.Deg(discrete_action.actuation.amount)))
    target_pan = current_pan + delta_rad

    limits_low, limits_high = robot.joint_position_limits
    lower_limit, upper_limit = limits_low[pan_idx], limits_high[pan_idx]
    if target_pan < lower_limit:
        target_pan = lower_limit
        msg = "Reach right limit when turning"
    elif target_pan > upper_limit:
        target_pan = upper_limit
        msg = "Reach left limit when turning"
    else:
        msg = ""
    if current_pan > upper_limit - 0.001 and delta_rad > 0:
        ret = False
    else:
        ret = True

    joint_positions[pan_idx] = target_pan
    robot.joint_positions = joint_positions
    sim.step_physics(1.0)
    return ret, msg


def lookRight(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    action_inputs: dict,
    delta_rad: Optional[float] = None,
):
    pan_idx = action_inputs["pan_idx"]
    agent = sim.agents[0]
    discrete_action = agent.agent_config.action_space["turn_right"]

    joint_positions = robot.joint_positions
    current_pan = joint_positions[pan_idx]
    if delta_rad is None:
        delta_rad = float(mn.Rad(mn.Deg(discrete_action.actuation.amount)))
    target_pan = current_pan - delta_rad

    limits_low, limits_high = robot.joint_position_limits
    lower_limit, upper_limit = limits_low[pan_idx], limits_high[pan_idx]
    if target_pan < lower_limit:
        target_pan = lower_limit
        msg = "Reach right limit when turning"
    elif target_pan > upper_limit:
        target_pan = upper_limit
        msg = "Reach left limit when turning"
    else:
        msg = ""
    if current_pan < lower_limit + 0.001 and delta_rad > 0:
        ret = False
    else:
        ret = True

    joint_positions[pan_idx] = target_pan
    robot.joint_positions = joint_positions
    sim.step_physics(1.0)
    return ret, msg


def turnToLook(sim: Simulator, robot: ManagedArticulatedObject, action_inputs: dict):
    joint_positions = robot.joint_positions

    pan_idx = 3
    current_pan = joint_positions[pan_idx]

    robot.rotate_y(mn.Rad(current_pan))
    joint_positions[pan_idx] = 0.0
    robot.joint_positions = joint_positions
    sim.step_physics(1.0)
    return True, ""


def resetHead(sim: Simulator, robot: ManagedArticulatedObject, action_inputs: dict):
    pan_idx = action_inputs["pan_idx"]
    tilt_idx = action_inputs["tilt_idx"]
    rest_pose = action_inputs["rest_pose"]
    joint_positions = robot.joint_positions
    joint_positions[pan_idx] = rest_pose[pan_idx]
    joint_positions[tilt_idx] = rest_pose[tilt_idx]
    robot.joint_positions = joint_positions
    sim.step_physics(1.0)
    return True, ""


def lookUp(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    action_inputs: dict,
    delta_rad: Optional[float] = None,
):
    pan_idx = action_inputs["pan_idx"]
    tilt_idx = action_inputs["tilt_idx"]
    agent = sim.agents[0]
    discrete_action = agent.agent_config.action_space["look_up"]

    joint_positions = robot.joint_positions
    current_tilt = joint_positions[tilt_idx]
    if delta_rad is None:
        delta_rad = float(mn.Rad(mn.Deg(discrete_action.actuation.amount)))
    target_tilt = current_tilt - delta_rad

    limits_low, limits_high = robot.joint_position_limits
    lower_limit, upper_limit = limits_low[tilt_idx], limits_high[tilt_idx]
    if target_tilt < lower_limit:
        target_tilt = lower_limit
        msg = "Reach up limit when looking"
    elif target_tilt > upper_limit:
        target_tilt = upper_limit
        msg = "Reach down limit when looking"
    else:
        msg = ""
    if current_tilt < lower_limit + 0.001 and delta_rad > 0:
        ret = False
    else:
        ret = True

    joint_positions[tilt_idx] = target_tilt
    robot.joint_positions = joint_positions
    sim.step_physics(1.0)
    return ret, msg


def lookDown(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    action_inputs: dict,
    delta_rad: Optional[float] = None,
):
    pan_idx = action_inputs["pan_idx"]
    tilt_idx = action_inputs["tilt_idx"]
    agent = sim.agents[0]
    discrete_action = agent.agent_config.action_space["look_down"]

    joint_positions = robot.joint_positions
    current_tilt = joint_positions[tilt_idx]
    if delta_rad is None:
        delta_rad = float(mn.Rad(mn.Deg(discrete_action.actuation.amount)))
    target_tilt = current_tilt + delta_rad

    limits_low, limits_high = robot.joint_position_limits
    lower_limit, upper_limit = limits_low[tilt_idx], limits_high[tilt_idx]
    if target_tilt < lower_limit:
        target_tilt = lower_limit
        msg = "Reach up limit when looking"
    elif target_tilt > upper_limit:
        target_tilt = upper_limit
        msg = "Reach down limit when looking"
    else:
        msg = ""
    if current_tilt > upper_limit - 0.001 and delta_rad > 0:
        ret = False
    else:
        ret = True

    joint_positions[tilt_idx] = target_tilt
    robot.joint_positions = joint_positions
    sim.step_physics(1.0)
    return ret, msg


def turnTo_obj(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    usg: UnifiedSceneGraph,
    action_inputs: dict,
    object_ref: Union[str, List[float]],
    check_seen: bool = True,
):
    pan_idx = action_inputs["pan_idx"]
    tilt_idx = action_inputs["tilt_idx"]
    object_id = resolve_object(object_ref, sim, action_inputs, force_whole_object=False)

    if object_id is None:
        return False, f"Invalid action argument"

    object = sutils.get_obj_from_id(
        sim, object_id
    )
    if object is None:
        return False, f"Invalid action argument"
    target_pos = object.translation

    robot_trans = robot.transformation
    local_target_pos = robot_trans.inverted().transform_point(target_pos)

    yaw_angle_rad = math.atan2(local_target_pos.z, local_target_pos.x)
    yaw_angle_deg = math.degrees(yaw_angle_rad)

    ret, msg = turnLeft(sim, robot, action_inputs, -yaw_angle_deg)
    assert ret, msg

    sim.step_physics(1.0)
    head_camera_node = robot.get_link_scene_node(
        robot.get_link_id_from_name("head_camera_link")
    )
    camera_pos = head_camera_node.absolute_translation

    dy = target_pos.y - camera_pos.y

    dxz = (
        mn.Vector2(target_pos.x, target_pos.z) - mn.Vector2(camera_pos.x, camera_pos.z)
    ).length()

    geometry_pitch_rad = -math.atan2(dy, dxz)
    target_tilt = geometry_pitch_rad
    current_tilt = robot.joint_positions[tilt_idx]

    ret, msg = lookDown(sim, robot, action_inputs, target_tilt - current_tilt)

    if check_seen:
        observations = sim.get_sensor_observations()
        seg = observations["semantic_sensor"]
        seen = object_in_sight(sim, set(np.unique(seg)), object_id)

        if not seen:
            return False, f"Cannot see the target object"

    return ret, msg


def turnTo_link(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    usg: UnifiedSceneGraph,
    action_inputs: dict,
    link_object_id: str,
):
    pan_idx = action_inputs["pan_idx"]
    tilt_idx = action_inputs["tilt_idx"]

    object = sutils.get_obj_from_id(
        sim, link_object_id
    )
    if object is None:
        return False, f"Invalid action argument"

    object_node_ID = usg.cache["sim_handle_to_SG_node_ID"][object.handle]
    object_node = usg.graph.get_node_from_ID(object_node_ID)
    link_dict = object_node.get_info("receptacle_links")

    if link_dict is None:
        link_dict = {}

    rec_name = None
    for link_idx in link_dict:
        if link_dict[link_idx][0][2] == link_object_id:
            rec_name = link_dict[link_idx][0][0]
    if rec_name:
        rec = usg.cache["rec_name_to_rec"][rec_name]

        T = rec.get_global_transform(sim)
        center_local = rec.bounds.center()
        target_pos = T.transform_point(center_local)
    else:
        link_idx = None
        for _link_idx, obj_id in object.link_ids_to_object_ids.items():
            if obj_id == int(link_object_id):
                link_idx = _link_idx
                break
        if link_idx is None:
            aabb = object.aabb
            obj_size = aabb.size()
            obj_center = object.translation
        else:
            link_node = object.get_link_scene_node(link_idx)
            local_bb = link_node.compute_cumulative_bb()
            world_bb = habitat_sim.geo.get_transformed_bb(
                local_bb, link_node.absolute_transformation()
            )
            obj_size = world_bb.size()
            obj_center = world_bb.center()
            if obj_size.length() < 1e-5:
                obj_center = link_node.absolute_translation
                obj_size = object.aabb.size()
        target_pos = obj_center

    robot_trans = robot.transformation
    local_target_pos = robot_trans.inverted().transform_point(target_pos)

    yaw_angle_rad = math.atan2(local_target_pos.z, local_target_pos.x)
    yaw_angle_deg = math.degrees(yaw_angle_rad)

    ret, msg = turnLeft(sim, robot, action_inputs, -yaw_angle_deg)
    assert ret, msg

    sim.step_physics(1.0)
    head_camera_node = robot.get_link_scene_node(
        robot.get_link_id_from_name("head_camera_link")
    )
    camera_pos = head_camera_node.absolute_translation

    dy = target_pos.y - camera_pos.y

    dxz = (
        mn.Vector2(target_pos.x, target_pos.z) - mn.Vector2(camera_pos.x, camera_pos.z)
    ).length()

    geometry_pitch_rad = -math.atan2(dy, dxz)
    target_tilt = geometry_pitch_rad
    current_tilt = robot.joint_positions[tilt_idx]

    ret, msg = lookDown(sim, robot, action_inputs, target_tilt - current_tilt)

    return ret, msg


def turnTo_point(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    usg: UnifiedSceneGraph,
    action_inputs: dict,
    pixel: List[float],
):
    pan_idx = action_inputs["pan_idx"]
    tilt_idx = action_inputs["tilt_idx"]
    agent_cfg = sim.agents[0].agent_config
    sensor_spec = agent_cfg.sensor_specifications[0]
    H, W = sensor_spec.resolution
    hfov = sensor_spec.hfov

    ndc_x = pixel[0] * 2.0 - 1.0
    ndc_y = 1.0 - pixel[1] * 2.0

    # tan(theta/2) = (W/2) / f
    aspect_ratio = W / H
    tan_half_hfov = np.tan(float(mn.Rad(hfov)) / 2.0)
    tan_half_vfov = tan_half_hfov / aspect_ratio

    local_ray_dir = mn.Vector3(
        ndc_x * tan_half_hfov, ndc_y * tan_half_vfov, -1.0
    ).normalized()

    yaw_angle_rad = math.atan2(local_ray_dir.x, -local_ray_dir.z)
    yaw_angle_deg = math.degrees(yaw_angle_rad)
    ret, msg = turnLeft(sim, robot, action_inputs, -yaw_angle_deg)
    assert ret, msg
    pitch_angle_rad = math.atan2(local_ray_dir.y, -local_ray_dir.z)
    ret, msg = lookDown(sim, robot, action_inputs, -pitch_angle_rad)

    return ret, msg


def goTo_loc(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    usg: UnifiedSceneGraph,
    action_inputs: dict,
    location: List[float],
):
    snapped_pos = sim.pathfinder.snap_point(
        location, island_index=action_inputs["largest_indoor_island_idx"]
    )

    if np.isnan(snapped_pos).any():
        return False, f"Failed to find a valid location."

    target_vec = mn.Vector3(snapped_pos)
    robot.translation = target_vec
    sim.step_physics(1.0)
    dist = np.linalg.norm(np.array(location) - np.array(snapped_pos))
    if dist > 0.01:
        return (
            True,
            f"Teleported to a snapped location ({location}->{snapped_pos}, dist: {dist})",
        )
    return True, ""


def goTo_obj(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    usg: UnifiedSceneGraph,
    action_inputs: dict,
    object_ref: Union[str, List[float]],
):
    pan_idx = action_inputs["pan_idx"]
    tilt_idx = action_inputs["tilt_idx"]

    object_id = resolve_object(object_ref, sim, action_inputs, force_whole_object=False)
    if object_id is None:
        return False, f"Invalid action argument"

    if int(object_id) in usg.cache["idx2region_idx"]:
        region_idx = usg.cache["idx2region_idx"][int(object_id)]
        region = sim.semantic_scene.regions[region_idx]
        navmesh_points = [
            point
            for point in sim.pathfinder.build_navmesh_vertices()
            if region.contains(point)
        ]
        np.random.shuffle(navmesh_points)
        if len(navmesh_points) == 0:
            for _ in range(10):
                goal_point = sutils.get_floor_point_in_region(sim, region_idx)
                if goal_point is not None:
                    navmesh_points.append(goal_point)

        original_position = mn.Vector3(robot.translation)
        for goal_point in navmesh_points[:10]:
            robot.translation = mn.Vector3(goal_point)
            sim.step_physics(1.0)
            snapped_pos = sim.pathfinder.snap_point(robot.translation)
            if (
                not np.isnan(snapped_pos).any()
                and np.linalg.norm(
                    np.array(robot.translation) - np.array(snapped_pos)
                )
                <= 0.01
                and region.contains(robot.translation)
            ):
                return True, ""
        robot.translation = original_position
        sim.step_physics(1.0)
        return False, "No valid navigation position was found."

    obj = sutils.get_obj_from_id(sim, object_id)
    if obj is None:
        return False, f"Invalid action argument"
    if obj.object_id != int(object_id):
        return goTo_link(sim, robot, usg, action_inputs, object_id)
    aabb = obj.aabb
    obj_size = aabb.size()
    obj_center = obj.translation

    obj_radius = max(obj_size.x, obj_size.z) / 2.0

    target_dist = MIN_DIST_TO_OBJ_FOR_TELEPORT + obj_radius
    target_dist = min(target_dist, MANIPULATION_DISTANCE_THRESHOLD - 0.1)

    found_valid_point = False
    best_pos = None
    min_travel_dist = float("inf")
    current_agent_pos = [
        float(robot.translation.x),
        float(robot.translation.y),
        float(robot.translation.z),
    ]
    agent_y = current_agent_pos[1]
    samples = range(0, 360, 90)
    for angle_deg in samples:
        angle_rad = math.radians(angle_deg)

        candidate_x = obj_center.x + target_dist * math.cos(angle_rad)
        candidate_z = obj_center.z + target_dist * math.sin(angle_rad)
        candidate_y = agent_y
        candidate_vec = mn.Vector3(candidate_x, candidate_y, candidate_z)

        snapped_pt = sim.pathfinder.snap_point(
            candidate_vec, island_index=action_inputs["largest_indoor_island_idx"]
        )

        if not np.isnan(snapped_pt).any():
            dist_diff = (mn.Vector3(snapped_pt) - candidate_vec).length()

            if dist_diff < 10000 and abs(snapped_pt.y - agent_y) < 0.15:
                robot.translation = mn.Vector3(snapped_pt)
                sim.step_physics(1.0)
                ret, msg = turnTo_obj(
                    sim, robot, usg, action_inputs, object_ref, check_seen=False
                )

                observations = sim.get_sensor_observations()
                seg = observations["semantic_sensor"]

                seen = object_in_sight(sim, set(np.unique(seg)), object_id)

                if seen:
                    dist_to_agent = (snapped_pt - current_agent_pos).length()

                    if dist_to_agent < min_travel_dist:
                        min_travel_dist = dist_to_agent
                        best_pos = snapped_pt

    if best_pos is None:
        return False, "No valid navigation position was found."

    robot.translation = mn.Vector3(best_pos)
    sim.step_physics(1.0)
    ret, msg = turnTo_obj(sim, robot, usg, action_inputs, object_ref)

    return True, msg


def goTo_link(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    usg: UnifiedSceneGraph,
    action_inputs: dict,
    link_object_id: str,
):
    pan_idx = action_inputs["pan_idx"]
    tilt_idx = action_inputs["tilt_idx"]

    obj = sutils.get_obj_from_id(sim, int(link_object_id))
    if obj is None:
        return False, f"Invalid action argument"

    object_node_ID = usg.cache["sim_handle_to_SG_node_ID"][obj.handle]
    object_node = usg.graph.get_node_from_ID(object_node_ID)
    link_dict = object_node.get_info("receptacle_links")

    if link_dict is None:
        link_dict = {}

    rec_name = None
    for link_idx in link_dict:
        if link_dict[link_idx][0][2] == link_object_id:
            rec_name = link_dict[link_idx][0][0]
    if rec_name:
        rec = usg.cache["rec_name_to_rec"][rec_name]
        aabb = rec.bounds
        obj_size = aabb.size()

        T = rec.get_global_transform(sim)
        center_local = rec.bounds.center()
        obj_center = T.transform_point(center_local)
    else:
        link_idx = None
        for _link_idx, obj_id in obj.link_ids_to_object_ids.items():
            if obj_id == int(link_object_id):
                link_idx = _link_idx
                break
        if link_idx is None:
            aabb = obj.aabb
            obj_size = aabb.size()
            obj_center = obj.translation
        else:
            # Per-link bounds: ManagedArticulatedObject has no get_link_aabb; use link SceneNode
            # subtree BB + world transform (see habitat_sim.scene.SceneNode.compute_cumulative_bb,
            # habitat_sim.geo.get_transformed_bb).
            link_node = obj.get_link_scene_node(link_idx)
            local_bb = link_node.compute_cumulative_bb()
            world_bb = habitat_sim.geo.get_transformed_bb(
                local_bb, link_node.absolute_transformation()
            )
            obj_size = world_bb.size()
            obj_center = world_bb.center()
            if obj_size.length() < 1e-5:
                obj_center = link_node.absolute_translation
                obj_size = obj.aabb.size()

    obj_radius = max(obj_size.x, obj_size.z) / 2.0

    target_dist = MIN_DIST_TO_OBJ_FOR_TELEPORT + obj_radius
    target_dist = min(target_dist, MANIPULATION_DISTANCE_THRESHOLD - 0.1)

    found_valid_point = False
    best_pos = None
    min_travel_dist = float("inf")
    current_agent_pos = [
        float(robot.translation.x),
        float(robot.translation.y),
        float(robot.translation.z),
    ]
    agent_y = current_agent_pos[1]
    samples = range(0, 360, 90)
    for angle_deg in samples:
        angle_rad = math.radians(angle_deg)

        candidate_x = obj_center.x + target_dist * math.cos(angle_rad)
        candidate_z = obj_center.z + target_dist * math.sin(angle_rad)
        candidate_y = agent_y
        candidate_vec = mn.Vector3(candidate_x, candidate_y, candidate_z)

        snapped_pt = sim.pathfinder.snap_point(
            candidate_vec, island_index=action_inputs["largest_indoor_island_idx"]
        )

        if not np.isnan(snapped_pt).any():
            if (
                abs(snapped_pt.x - current_agent_pos[0]) < 0.01
                and abs(snapped_pt.z - current_agent_pos[2]) < 0.01
            ):
                continue

            dist_diff = (mn.Vector3(snapped_pt) - candidate_vec).length()

            if dist_diff < 10000 and abs(snapped_pt.y - agent_y) < 0.15:
                robot.translation = mn.Vector3(snapped_pt)
                sim.step_physics(1.0)
                ret, msg = turnTo_link(sim, robot, usg, action_inputs, link_object_id)
                assert ret, msg

                observations = sim.get_sensor_observations()
                seg = observations["semantic_sensor"]
                seen = object_in_sight(sim, set(np.unique(seg)), int(link_object_id))
                if seen:
                    dist_to_agent = (snapped_pt - current_agent_pos).length()

                    if dist_to_agent < min_travel_dist:
                        min_travel_dist = dist_to_agent
                        best_pos = snapped_pt

    if best_pos is None:
        return False, "No valid navigation position was found."

    robot.translation = mn.Vector3(best_pos)
    sim.step_physics(1.0)
    ret, msg = turnTo_link(sim, robot, usg, action_inputs, link_object_id)
    assert ret, msg

    return True, msg


def pick_obj(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    usg: UnifiedSceneGraph,
    action_inputs: dict,
    object_ref: Union[str, List[float]],
):
    carry_pose = action_inputs["carry_pose"]
    gripper_idx = action_inputs["gripper_idx"]

    in_hand_objects = usg.graph.get_neighbors_of_relation(
        "robot_agent", RelationType.GRASPING
    )
    if len(in_hand_objects):
        return False, "The agent is already holding an object."

    object_id = resolve_object(object_ref, sim, action_inputs, force_whole_object=True)
    if object_id is None:
        return False, f"Invalid action argument"
    object = sutils.get_obj_from_id(sim, object_id)
    if object is None:
        return False, f"Invalid action argument"
    object_node_ID = usg.cache["sim_handle_to_SG_node_ID"][object.handle]
    object_node = usg.graph.get_node_from_ID(object_node_ID)
    if not object_node.has_property(PropertyType.GRASPABLE):
        return False, "The target object is not graspable."

    observations = sim.get_sensor_observations()
    seg = observations["semantic_sensor"]
    if not object_in_sight(sim, set(np.unique(seg)), object.object_id):
        parents = usg.graph.get_neighbors_of_relation(
            object_node_ID, RelationType.INSIDE_RECEPTACLE
        )
        for parent in parents:
            if parent.has_state(StateType.CLOSED):
                return False, "The target object is inside a closed receptacle."
        return False, "The target object is not visible."

    assert not object.is_articulated, object_node_ID

    agent_pos = np.array(robot.translation)
    obj_pos = np.array(object.translation)

    dist_to_target = calculate_distance_2D(agent_pos, obj_pos)
    if dist_to_target > MANIPULATION_DISTANCE_THRESHOLD:
        return False, "The target is too far away."

    carry_pose_now = copy.deepcopy(carry_pose)
    carry_pose_now[action_inputs["pan_idx"]] = robot.joint_positions[
        action_inputs["pan_idx"]
    ]
    carry_pose_now[action_inputs["tilt_idx"]] = robot.joint_positions[
        action_inputs["tilt_idx"]
    ]
    robot.joint_positions = carry_pose_now

    ee = robot.get_link_scene_node(gripper_idx)
    object.translation = ee.absolute_translation
    usg.graph.remove_all_edges(object_node_ID)
    usg.graph.add_edge(
        "robot_agent",
        object_node_ID,
        RelationType.GRASPING,
        flip_edge(RelationType.GRASPING),
    )
    bind_object_to_parent_link(
        action_inputs["sync_object_dict"], object, robot, action_inputs["gripper_idx"]
    )

    assert object.motion_type == habitat_sim.physics.MotionType.KINEMATIC, [
        object_node_ID,
        object.motion_type,
    ]

    sim.step_physics(1.0)

    return True, ""


def placeTo_recep(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    usg: UnifiedSceneGraph,
    action_inputs: dict,
    object_ref: Union[str, List[float]],
    allowed_propositions: List[str] = ["on", "within"],
    next_to_object_ref: Union[str, List[float]] = None,
):

    in_hand_objects = usg.graph.get_neighbors_of_relation(
        "robot_agent", RelationType.GRASPING
    )
    if len(in_hand_objects) == 0:
        return False, "The agent is not holding an object."
    assert len(in_hand_objects) == 1, [x.ID for x in in_hand_objects]

    in_hand_object_node = in_hand_objects[0]
    in_hand_object = sutils.get_obj_from_id(
        sim, int(in_hand_object_node.ID.split("_")[-1])
    )
    assert in_hand_object is not None

    if isinstance(object_ref, str) and object_ref == str(
        sutils.get_obj_from_id(sim, int(object_ref)).object_id
    ):
        recep_object_id = resolve_object(
            object_ref, sim, action_inputs, force_whole_object=True
        )
        if recep_object_id is None:
            return False, f"Invalid action argument"
        all_ids, all_ids_freq = resolve_all_ids(
            [0.0, 0.0, 1.0, 1.0], sim, action_inputs, force_whole_object=False
        )

        all_link_object_ids = []
        if recep_object_id in action_inputs["object_id_to_link_object_id"]:
            for i in all_ids:
                if (
                    i in action_inputs["link_object_id_to_object_id"]
                    and action_inputs["link_object_id_to_object_id"][i]
                    == recep_object_id
                ):
                    all_link_object_ids.append(i)
            if len(all_link_object_ids) == 0:
                return False, ""

    elif isinstance(object_ref, str):
        all_ids = [int(object_ref)]
        all_ids_freq = {int(object_ref): 100}
        recep_object_id = int(object_ref)

    else:
        recep_object_id = resolve_object(
            object_ref, sim, action_inputs, force_whole_object=True
        )
        if recep_object_id is None:
            return False, f"Invalid action argument"
        all_ids, all_ids_freq = resolve_all_ids(
            object_ref, sim, action_inputs, force_whole_object=False
        )

        all_link_object_ids = []
        if recep_object_id in action_inputs["object_id_to_link_object_id"]:
            for i in all_ids:
                if (
                    i in action_inputs["link_object_id_to_object_id"]
                    and action_inputs["link_object_id_to_object_id"][i]
                    == recep_object_id
                ):
                    all_link_object_ids.append(i)
            if len(all_link_object_ids) == 0:
                return False, "The target is not a valid receptacle."

    recep_object = sutils.get_obj_from_id(sim, recep_object_id)
    recep_object_node_ID = usg.cache["sim_handle_to_SG_node_ID"][recep_object.handle]
    recep_object_node = usg.graph.get_node_from_ID(recep_object_node_ID)
    if not recep_object_node.has_property(PropertyType.IS_RECEPTACLE):
        return False, "The target is not a valid receptacle."
    recep_links_dict = recep_object_node.get_info("receptacle_links")

    assert recep_links_dict is not None, [recep_object_node_ID]

    all_candidate_receps: List[Receptacle] = []
    all_candidate_recep_names: List[str] = []
    all_candidate_recep_freq: List[int] = []
    recep_to_link_idxs: dict[Receptacle, int] = {}
    recep_to_prop: dict[Receptacle, str] = {}
    closed_receptacle = False
    for link_idx, recep_infos in recep_links_dict.items():
        if link_idx == -1:
            for recep_info in recep_infos:
                recep_name, proposition, recep_link_object_id, _ = recep_info
                if proposition in allowed_propositions:
                    if proposition == "within":
                        pass

                    all_candidate_receps.append(
                        usg.cache["rec_name_to_rec"][recep_name]
                    )
                    all_candidate_recep_names.append(recep_name)
                    all_candidate_recep_freq.append(all_ids_freq[recep_link_object_id])
                    recep_to_link_idxs[usg.cache["rec_name_to_rec"][recep_name]] = (
                        link_idx
                    )
                    recep_to_prop[usg.cache["rec_name_to_rec"][recep_name]] = (
                        proposition
                    )
        else:
            for recep_info in recep_infos:
                recep_name, proposition, recep_link_object_id, _ = recep_info
                assert (
                    recep_link_object_id
                    == recep_object.link_ids_to_object_ids[link_idx]
                )

                if (
                    proposition in allowed_propositions
                    and recep_link_object_id in all_ids
                ):
                    if proposition == "within":
                        assert recep_object.is_articulated

                        if recep_object.get_link_joint_type(link_idx) in [
                            habitat_sim.physics.JointType.Revolute,
                            habitat_sim.physics.JointType.Prismatic,
                        ]:
                            if not recep_object_node.has_state(StateType.OPEN):
                                closed_receptacle = True
                                continue
                            detailed_state = recep_object_node.attributes.states[
                                StateType.OPEN
                            ]
                            assert len(detailed_state.params["opened_link_idxs"])
                            recep_link_idx = recep_object.link_object_ids[
                                recep_link_object_id
                            ]
                            if (
                                recep_link_idx
                                not in detailed_state.params["opened_link_idxs"]
                            ):
                                closed_receptacle = True
                                continue

                    all_candidate_receps.append(
                        usg.cache["rec_name_to_rec"][recep_name]
                    )
                    all_candidate_recep_names.append(recep_name)
                    all_candidate_recep_freq.append(all_ids_freq[recep_link_object_id])
                    recep_to_link_idxs[usg.cache["rec_name_to_rec"][recep_name]] = (
                        link_idx
                    )
                    recep_to_prop[usg.cache["rec_name_to_rec"][recep_name]] = (
                        proposition
                    )

    if len(all_candidate_receps) == 0:
        if closed_receptacle:
            return False, "The target receptacle is closed."
        if allowed_propositions == ["on"]:
            return False, "The target is not a valid receptacle."
        elif allowed_propositions == ["within"]:
            return False, "The target is not a valid receptacle."
        return False, "The target is not a valid receptacle."

    if next_to_object_ref is not None:
        next_to_object_id = resolve_object(
            next_to_object_ref, sim, action_inputs, force_whole_object=True
        )
        next_to_object = sutils.get_obj_from_id(next_to_object_id)
        if next_to_object is None:
            return False, f"Invalid action argument"
    else:
        next_to_object = None

    if sum(all_candidate_recep_freq) == 0:
        all_candidate_recep_freq = [1] * len(all_candidate_receps)
    target_poses, target_receptacles, msg = (
        sample_position_on_receptacle_links_with_reference(
            sim,
            robot,
            in_hand_object,
            recep_object,
            all_candidate_receps,
            next_to_object,
            dist_thresh=MANIPULATION_DISTANCE_THRESHOLD,
            weights=all_candidate_recep_freq,
        )
    )
    if len(target_poses) == 0:
        if msg == "dist issue":
            return False, "The target is too far away."
        if next_to_object is not None:
            target_poses_no_reference, _ = (
                sample_position_on_receptacle_links_with_reference(
                    sim,
                    robot,
                    in_hand_object,
                    recep_object,
                    all_candidate_receps,
                    None,
                    dist_thresh=MANIPULATION_DISTANCE_THRESHOLD,
                    weights=all_candidate_recep_freq,
                )
            )
            if len(target_poses_no_reference) > 0:
                return False, "No valid placement position was found."
        return False, "No valid placement position was found."

    ori_trans, ori_rot = in_hand_object.translation, in_hand_object.rotation

    max_place_trials = 10
    n_try = min(max_place_trials, len(target_poses))
    chosen_receptacle = None
    for i in range(n_try):
        target_pos, target_rot = target_poses[i]
        chosen_receptacle = target_receptacles[i]
        in_hand_object.translation = target_pos
        in_hand_object.rotation = target_rot
        sim.step_physics(1.0)

        observations = sim.get_sensor_observations()
        seg = observations["semantic_sensor"]
        if object_in_sight(sim, set(np.unique(seg)), in_hand_object.object_id):
            break
    else:
        in_hand_object.translation = ori_trans
        in_hand_object.rotation = ori_rot
        sim.step_physics(1.0)
        return False, "No valid placement position was found."

    target_receptacle = chosen_receptacle

    rest_pose_now = copy.deepcopy(action_inputs["rest_pose"])
    rest_pose_now[action_inputs["pan_idx"]] = robot.joint_positions[
        action_inputs["pan_idx"]
    ]
    rest_pose_now[action_inputs["tilt_idx"]] = robot.joint_positions[
        action_inputs["tilt_idx"]
    ]
    robot.joint_positions = rest_pose_now

    usg.graph.remove_all_edges(in_hand_object_node)
    target_proposition = recep_to_prop[target_receptacle]
    if target_proposition == "on":
        usg.graph.add_edge(
            in_hand_object_node,
            recep_object_node,
            RelationType.ON_RECEPTACLE,
            flip_edge(RelationType.ON_RECEPTACLE),
            {"rec_name": target_receptacle.unique_name},
        )

    elif target_proposition == "within":
        usg.graph.add_edge(
            in_hand_object_node,
            recep_object_node,
            RelationType.INSIDE_RECEPTACLE,
            flip_edge(RelationType.INSIDE_RECEPTACLE),
            {"rec_name": target_receptacle.unique_name},
        )

    else:
        assert False, target_proposition
    bind_object_to_parent_link(
        action_inputs["sync_object_dict"],
        in_hand_object,
        recep_object,
        recep_to_link_idxs[target_receptacle],
    )
    sim.step_physics(1.0)

    return True, ""


def placeIn_recep(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    usg: UnifiedSceneGraph,
    action_inputs: dict,
    object_ref: Union[str, List[float]],
    next_to_object_ref: Union[str, List[float]] = None,
):
    return placeTo_recep(
        sim,
        robot,
        usg,
        action_inputs,
        object_ref,
        allowed_propositions=["within"],
        next_to_object_ref=next_to_object_ref,
    )


def placeOn_recep(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    usg: UnifiedSceneGraph,
    action_inputs: dict,
    object_ref: Union[str, List[float]],
    next_to_object_ref: Union[str, List[float]] = None,
):
    return placeTo_recep(
        sim,
        robot,
        usg,
        action_inputs,
        object_ref,
        allowed_propositions=["on"],
        next_to_object_ref=next_to_object_ref,
    )


def placeOn_roomFloor(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    usg: UnifiedSceneGraph,
    action_inputs: dict,
    next_to_object_ref: Union[str, List[float]] = None,
):
    in_hand_objects = usg.graph.get_neighbors_of_relation(
        "robot_agent", RelationType.GRASPING
    )
    if len(in_hand_objects) == 0:
        return False, "The agent is not holding an object."
    assert len(in_hand_objects) == 1, [x.ID for x in in_hand_objects]

    in_hand_object_node = in_hand_objects[0]
    in_hand_object = sutils.get_obj_from_id(
        sim, int(in_hand_object_node.ID.split("_")[-1])
    )
    assert in_hand_object is not None

    agent_room_node = usg.graph.get_neighbors_of_relation(
        "robot_agent", RelationType.INSIDE_ROOM
    )
    assert len(agent_room_node) == 1
    agent_room_node = agent_room_node[0]
    agent_region_id = None
    for region in sim.semantic_scene.regions:
        if region.contains(robot.translation):
            agent_region_id = region.id
            break
    assert (
        usg.cache["sim_region_id_to_SG_node_ID"][agent_region_id] == agent_room_node.ID
    )

    if next_to_object_ref is not None:
        next_to_object_id = resolve_object(
            next_to_object_ref, sim, action_inputs, force_whole_object=True
        )
        next_to_object = sutils.get_obj_from_id(next_to_object_id)
        if next_to_object is None:
            return False, f"Invalid action argument"
    else:
        next_to_object = None

    target_poses, _, msg = sample_position_on_room_floor_with_reference(
        sim,
        robot,
        in_hand_object,
        action_inputs["largest_indoor_island_idx"],
        next_to_object,
        dist_thresh=MANIPULATION_DISTANCE_THRESHOLD,
        require_region_id=agent_region_id,
    )
    if len(target_poses) == 0:
        if msg == "dist issue":
            return False, "The target is too far away."
        if next_to_object is not None:
            target_poses_no_reference, _ = sample_position_on_room_floor_with_reference(
                sim,
                robot,
                in_hand_object,
                action_inputs["largest_indoor_island_idx"],
                None,
                dist_thresh=MANIPULATION_DISTANCE_THRESHOLD,
                require_region_id=agent_region_id,
            )
            if len(target_poses_no_reference) > 0:
                return False, "No valid placement position was found."
        return False, "No valid placement position was found."

    # Use the closest valid pose.
    target_pos, target_rot = target_poses[0]
    in_hand_object.translation = target_pos
    in_hand_object.rotation = target_rot

    rest_pose_now = copy.deepcopy(action_inputs["rest_pose"])
    rest_pose_now[action_inputs["pan_idx"]] = robot.joint_positions[
        action_inputs["pan_idx"]
    ]
    rest_pose_now[action_inputs["tilt_idx"]] = robot.joint_positions[
        action_inputs["tilt_idx"]
    ]
    robot.joint_positions = rest_pose_now
    sim.step_physics(1.0)

    room_ID = None
    for region in sim.semantic_scene.regions:
        if region.contains(list(target_pos)):
            room_ID = usg.cache["sim_region_id_to_SG_node_ID"][region.id]
            break
    if room_ID is None:
        assert False, f"placed point ({target_pos}) does not belong to any room"
    room_node = usg.graph.get_node_from_ID(room_ID)

    usg.graph.remove_all_edges(in_hand_object_node)
    usg.graph.add_edge(
        in_hand_object_node,
        room_node,
        RelationType.INSIDE_ROOM,
        flip_edge(RelationType.INSIDE_ROOM),
    )
    usg.graph.add_edge(
        in_hand_object_node,
        room_node,
        RelationType.ON_ROOM_FLOOR,
        flip_edge(RelationType.ON_ROOM_FLOOR),
    )

    action_inputs["sync_object_dict"][in_hand_object.object_id] = None

    assert agent_room_node.ID == room_node.ID, [agent_room_node.ID, room_node.ID]

    turnTo_obj(sim, robot, usg, action_inputs, in_hand_object_node.ID, check_seen=True)

    return True, ""


def placeTo_point(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    usg: UnifiedSceneGraph,
    action_inputs: dict,
    pixel: List[float],
):

    recep_object_id = resolve_object(pixel, sim, action_inputs, force_whole_object=True)
    if recep_object_id is None:
        return False, f"No object found at the target position"
    if recep_object_id == 0:
        return placeTo_floor_point(sim, robot, usg, action_inputs, pixel)
    else:
        return placeTo_recep_point(
            sim, robot, usg, action_inputs, pixel, recep_object_id=recep_object_id
        )


def placeTo_recep_point(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    usg: UnifiedSceneGraph,
    action_inputs: dict,
    pixel: List[float],
    recep_object_id: Optional[int] = None,
):
    in_hand_objects = usg.graph.get_neighbors_of_relation(
        "robot_agent", RelationType.GRASPING
    )
    if len(in_hand_objects) == 0:
        return False, "The agent is not holding an object."
    assert len(in_hand_objects) == 1, [x.ID for x in in_hand_objects]

    in_hand_object_node = in_hand_objects[0]
    in_hand_object = sutils.get_obj_from_id(
        sim, int(in_hand_object_node.ID.split("_")[-1])
    )
    assert in_hand_object is not None

    if recep_object_id is None:
        recep_object_id = resolve_object(
            pixel, sim, action_inputs, force_whole_object=True
        )
        if recep_object_id is None:
            return False, f"No object found at the target position"
    target_id, _ = resolve_all_ids(pixel, sim, action_inputs, force_whole_object=False)
    assert len(target_id) == 1
    target_id = list(target_id)[0]

    if recep_object_id in action_inputs["object_id_to_link_object_id"]:
        if (
            target_id in action_inputs["link_object_id_to_object_id"]
            and action_inputs["link_object_id_to_object_id"][target_id]
            == recep_object_id
        ):
            pass
        else:
            return False, "The target is not a valid receptacle."

    recep_object = sutils.get_obj_from_id(sim, recep_object_id)
    assert recep_object is not None, recep_object_id
    recep_object_node_ID = usg.cache["sim_handle_to_SG_node_ID"][recep_object.handle]
    recep_object_node = usg.graph.get_node_from_ID(recep_object_node_ID)
    if not recep_object_node.has_property(PropertyType.IS_RECEPTACLE):
        return False, "The target is not a valid receptacle."
    recep_links_dict = recep_object_node.get_info("receptacle_links")
    assert recep_links_dict is not None, [recep_object_node_ID]

    render_camera = sim.agents[0]._sensors["color_sensor"].render_camera
    viewport_size = render_camera.viewport
    W, H = viewport_size.x, viewport_size.y
    x_norm, y_norm = pixel

    px = int(x_norm * W)
    py = int(y_norm * H)
    ray = render_camera.unproject(mn.Vector2i(px, py))
    raycast_results = sim.cast_ray(ray)
    if not raycast_results.has_hits():
        return False, f"No object found at the target position"

    hit_info = raycast_results.hits[0]
    hit_obj_id = hit_info.object_id
    assert hit_obj_id == target_id, (
        f"hit object id {hit_obj_id} does not match target object id {target_id}"
    )

    success, target_position, target_rotation = place_to_position_on_receptacle_link(
        sim,
        robot,
        in_hand_object,
        [hit_obj_id],
        hit_info.point,
    )

    if not success:
        return False, "No valid placement position was found."

    agent_pos = np.array(robot.translation)
    obj_pos = np.array(target_position)

    dist_to_target = calculate_distance_2D(agent_pos, obj_pos)
    if dist_to_target > MANIPULATION_DISTANCE_THRESHOLD:
        return False, "The target is too far away."

    ori_translation = in_hand_object.translation
    ori_rotation = in_hand_object.rotation

    in_hand_object.translation = target_position
    in_hand_object.rotation = target_rotation
    sim.step_physics(1.0)
    rec_names, _confidence, info_string = sutils.get_obj_receptacle_and_confidence(
        sim,
        in_hand_object,
        usg.cache["rec_name_to_rec"],
        island_index=action_inputs["largest_indoor_island_idx"],
    )
    assert len(rec_names) > 0, info_string
    rec_name = rec_names[0]
    rec = usg.cache["rec_name_to_rec"][rec_name]
    _obj_handle, target_proposition = usg.cache["rev_receptacle_dict"][rec]
    assert _obj_handle == recep_object.handle, [
        _obj_handle,
        recep_object.object_id,
        recep_object.handle,
        target_proposition,
    ]

    if target_proposition == "within":
        if not recep_object.is_articulated:
            pass
        else:
            link_idx = recep_object.link_object_ids[target_id]

            if recep_object.get_link_joint_type(link_idx) in [
                habitat_sim.physics.JointType.Revolute,
                habitat_sim.physics.JointType.Prismatic,
            ]:
                if not recep_object_node.has_state(StateType.OPEN):
                    in_hand_object.translation = ori_translation
                    in_hand_object.rotation = ori_rotation
                    return False, "The target receptacle is closed."
                detailed_state = recep_object_node.attributes.states[StateType.OPEN]
                assert len(detailed_state.params["opened_link_idxs"])
                recep_link_idx = recep_object.link_object_ids[target_id]
                if not recep_link_idx in detailed_state.params["opened_link_idxs"]:
                    in_hand_object.translation = ori_translation
                    in_hand_object.rotation = ori_rotation
                    return False, "The target receptacle is closed."

    in_hand_object.translation = target_position
    in_hand_object.rotation = target_rotation

    rest_pose_now = copy.deepcopy(action_inputs["rest_pose"])
    rest_pose_now[action_inputs["pan_idx"]] = robot.joint_positions[
        action_inputs["pan_idx"]
    ]
    rest_pose_now[action_inputs["tilt_idx"]] = robot.joint_positions[
        action_inputs["tilt_idx"]
    ]
    robot.joint_positions = rest_pose_now

    usg.graph.remove_all_edges(in_hand_object_node)
    if target_proposition == "on":
        usg.graph.add_edge(
            in_hand_object_node,
            recep_object_node,
            RelationType.ON_RECEPTACLE,
            flip_edge(RelationType.ON_RECEPTACLE),
            {"rec_name": rec_name},
        )
    elif target_proposition == "within":
        usg.graph.add_edge(
            in_hand_object_node,
            recep_object_node,
            RelationType.INSIDE_RECEPTACLE,
            flip_edge(RelationType.INSIDE_RECEPTACLE),
            {"rec_name": rec_name},
        )
    else:
        assert False, target_proposition

    if recep_object.is_articulated:
        bind_object_to_parent_link(
            action_inputs["sync_object_dict"],
            in_hand_object,
            recep_object,
            recep_object.link_object_ids[target_id],
        )
    else:
        bind_object_to_parent_link(
            action_inputs["sync_object_dict"], in_hand_object, recep_object
        )

    sim.step_physics(1.0)
    return True, ""


def placeTo_floor_point(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    usg: UnifiedSceneGraph,
    action_inputs: dict,
    pixel: List[float],
):
    in_hand_objects = usg.graph.get_neighbors_of_relation(
        "robot_agent", RelationType.GRASPING
    )
    if len(in_hand_objects) == 0:
        return False, "The agent is not holding an object."
    assert len(in_hand_objects) == 1, [x.ID for x in in_hand_objects]

    in_hand_object_node = in_hand_objects[0]
    in_hand_object = sutils.get_obj_from_id(
        sim, int(in_hand_object_node.ID.split("_")[-1])
    )
    assert in_hand_object is not None

    render_camera = sim.agents[0]._sensors["color_sensor"].render_camera
    viewport_size = render_camera.viewport
    W, H = viewport_size.x, viewport_size.y
    x_norm, y_norm = pixel

    px = int(x_norm * W)
    py = int(y_norm * H)
    ray = render_camera.unproject(mn.Vector2i(px, py))
    raycast_results = sim.cast_ray(ray)
    if not raycast_results.has_hits():
        return False, f"No object found at the target position"

    hit_info = raycast_results.hits[0]
    hit_obj_id = hit_info.object_id
    assert hit_obj_id == 0, f"hit object id {hit_obj_id} does not match floor"

    success, target_position, target_rotation = place_to_position_on_room_floor(
        sim,
        robot,
        in_hand_object,
        hit_info.point,
    )

    if not success:
        return False, "No valid placement position was found."

    agent_pos = np.array(robot.translation)
    obj_pos = np.array(target_position)

    dist_to_target = calculate_distance_2D(agent_pos, obj_pos)
    if dist_to_target > MANIPULATION_DISTANCE_THRESHOLD:
        return False, "The target is too far away."

    in_hand_object.translation = target_position
    in_hand_object.rotation = target_rotation

    rest_pose_now = copy.deepcopy(action_inputs["rest_pose"])
    rest_pose_now[action_inputs["pan_idx"]] = robot.joint_positions[
        action_inputs["pan_idx"]
    ]
    rest_pose_now[action_inputs["tilt_idx"]] = robot.joint_positions[
        action_inputs["tilt_idx"]
    ]
    robot.joint_positions = rest_pose_now

    room_ID = None
    for region in sim.semantic_scene.regions:
        if region.contains(list(hit_info.point)):
            room_ID = usg.cache["sim_region_id_to_SG_node_ID"][region.id]
            break
    if room_ID is None:
        assert False, (
            f"pixel {pixel} hit point {hit_info.point} does not belong to any room"
        )
    room_node = usg.graph.get_node_from_ID(room_ID)

    usg.graph.remove_all_edges(in_hand_object_node)
    usg.graph.add_edge(
        in_hand_object_node,
        room_node,
        RelationType.INSIDE_ROOM,
        flip_edge(RelationType.INSIDE_ROOM),
    )
    usg.graph.add_edge(
        in_hand_object_node,
        room_node,
        RelationType.ON_ROOM_FLOOR,
        flip_edge(RelationType.ON_ROOM_FLOOR),
    )

    action_inputs["sync_object_dict"][in_hand_object.object_id] = None

    sim.step_physics(1.0)
    return True, ""


def open_recep(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    usg: UnifiedSceneGraph,
    action_inputs: dict,
    object_ref: Union[str, List[float]],
    openness: float = 1.0,
):

    in_hand_objects = usg.graph.get_neighbors_of_relation(
        "robot_agent", RelationType.GRASPING
    )

    object_id = resolve_object(object_ref, sim, action_inputs, force_whole_object=True)
    if object_id is None:
        return False, f"Invalid action argument"
    object = sutils.get_obj_from_id(sim, object_id)
    object_node_ID = usg.cache["sim_handle_to_SG_node_ID"][object.handle]
    object_node = usg.graph.get_node_from_ID(object_node_ID)
    if not object_node.has_property(PropertyType.OPENABLE):
        return False, "The target object is not openable."

    if not object_node.get_info("is_articulated"):
        return False, "The target object is not openable."

    if isinstance(object_ref, str) and object_id == object.object_id:
        seg_img = np.unique(sim.get_sensor_observations()["semantic_sensor"])
        candidate_link_object_ids = [x for x in object.link_object_ids if x in seg_img]
        sorted_candidate_link_object_ids = candidate_link_object_ids
        np.random.shuffle(sorted_candidate_link_object_ids)
    elif isinstance(object_ref, str):
        sorted_candidate_link_object_ids = [int(object_ref)]
    else:
        all_ids, all_ids_freq = resolve_all_ids(
            object_ref, sim, action_inputs, force_whole_object=False
        )
        candidate_link_object_ids = []
        for i in all_ids:
            if i in object.link_object_ids:
                candidate_link_object_ids.append(i)
        sorted_candidate_link_object_ids = sorted(
            candidate_link_object_ids, key=lambda x: all_ids_freq[x], reverse=True
        )

    agent_pos = np.array(robot.translation)

    detailed_state = object_node.attributes.states.get(StateType.OPEN, None)
    if (
        detailed_state is None
        or detailed_state.params is None
        or detailed_state.params == {}
    ):
        already_opened = set()
    else:
        already_opened = detailed_state.params["opened_link_idxs"]

    default_link = sutils.get_ao_default_link(object, compute_if_not_found=True)

    dist_issue = False
    for link_object_id in sorted_candidate_link_object_ids:
        link_idx = object.link_object_ids[link_object_id]
        if link_idx in already_opened:
            continue
        if object.get_link_joint_type(link_idx) in [
            habitat_sim.physics.JointType.Revolute,
            habitat_sim.physics.JointType.Prismatic,
        ]:
            link_pos = np.array(
                object.get_link_scene_node(link_idx).absolute_translation
            )

            dist_to_target = calculate_distance_2D(agent_pos, link_pos)
            if dist_to_target > MANIPULATION_DISTANCE_THRESHOLD:
                dist_issue = True
                continue

            if (
                detailed_state is None
                or detailed_state.params is None
                or detailed_state.params == {}
            ):
                sutils.open_link(object, link_idx)
                params = {
                    "opened_link_idxs": set(
                        [
                            link_idx,
                        ]
                    )
                }
                object_node.add_state(StateType.OPEN, params)
            else:
                old_params = detailed_state.params
                sutils.open_link(object, link_idx)
                params = {
                    "opened_link_idxs": old_params["opened_link_idxs"].union(
                        [
                            link_idx,
                        ]
                    )
                }
                object_node.add_state(StateType.OPEN, params)

            return True, ""
    if dist_issue:
        return False, "The target part is too far away."
    return False, ""


def close_recep(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    usg: UnifiedSceneGraph,
    action_inputs: dict,
    object_ref: Union[str, List[float]],
):

    in_hand_objects = usg.graph.get_neighbors_of_relation(
        "robot_agent", RelationType.GRASPING
    )

    object_id = resolve_object(object_ref, sim, action_inputs, force_whole_object=True)
    if object_id is None:
        return False, f"Invalid action argument"
    object = sutils.get_obj_from_id(sim, object_id)
    object_node_ID = usg.cache["sim_handle_to_SG_node_ID"][object.handle]
    object_node = usg.graph.get_node_from_ID(object_node_ID)
    if not object_node.has_property(PropertyType.OPENABLE):
        return False, "The target object is not openable."
    if not object_node.has_state(StateType.OPEN):
        return False, "The target object is not open."

    if not object_node.get_info("is_articulated"):
        return False, "The target object is not openable."

    if isinstance(object_ref, str) and object_id == object.object_id:
        seg_img = np.unique(sim.get_sensor_observations()["semantic_sensor"])
        candidate_link_object_ids = [x for x in object.link_object_ids if x in seg_img]
        sorted_candidate_link_object_ids = candidate_link_object_ids
        np.random.shuffle(sorted_candidate_link_object_ids)
    elif isinstance(object_ref, str):
        sorted_candidate_link_object_ids = [int(object_ref)]
    else:
        all_ids, all_ids_freq = resolve_all_ids(
            object_ref, sim, action_inputs, force_whole_object=False
        )
        candidate_link_object_ids = []
        for i in all_ids:
            if i in object.link_object_ids:
                candidate_link_object_ids.append(i)
        sorted_candidate_link_object_ids = sorted(
            candidate_link_object_ids, key=lambda x: all_ids_freq[x], reverse=True
        )

    agent_pos = np.array(robot.translation)

    detailed_state = object_node.attributes.states[StateType.OPEN]
    assert len(detailed_state.params["opened_link_idxs"])

    default_link = sutils.get_ao_default_link(object, compute_if_not_found=True)

    dist_issue = False
    for link_object_id in sorted_candidate_link_object_ids:
        link_idx = object.link_object_ids[link_object_id]
        if link_idx not in detailed_state.params["opened_link_idxs"]:
            continue
        if object.get_link_joint_type(link_idx) in [
            habitat_sim.physics.JointType.Revolute,
            habitat_sim.physics.JointType.Prismatic,
        ]:
            link_pos = np.array(
                object.get_link_scene_node(link_idx).absolute_translation
            )

            dist_to_target = calculate_distance_2D(agent_pos, link_pos)
            if dist_to_target > MANIPULATION_DISTANCE_THRESHOLD:
                dist_issue = True
                continue

            sutils.close_link(object, link_idx)

            if len(detailed_state.params["opened_link_idxs"]) == 1:
                object_node.add_state(StateType.CLOSED)
            else:
                detailed_state.params["opened_link_idxs"].remove(link_idx)

            return True, ""

    if dist_issue:
        return False, "The target part is too far away."
    return False, ""


# high-level


def find_objCls(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    action_inputs: dict,
    object_cls: str,
    by_dist: bool = True,
):
    candidates = []
    for object_id in action_inputs["object_id_to_class"]:
        if match_object_cls(object_cls, action_inputs["object_id_to_class"][object_id]):
            candidates.append(sutils.get_obj_from_id(sim, object_id))

    if len(candidates) == 0:
        return False, f"There is no object of class {object_cls}"

    agent = sim.agents[0]
    if by_dist:
        candidates = sorted(
            candidates,
            key=lambda x: calculate_distance_2D(
                x.translation, agent.scene_node.translation
            ),
        )
    else:
        np.random.shuffle(candidates)

    for object in candidates:
        ret, msg = goTo_obj(sim, robot, action_inputs, str(object.object_id))
        assert ret
        return True, msg

    return (
        False,
        f"Cannot find an object of class {object_cls} (been to several candidates but did not see the object)",
    )
