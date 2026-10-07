from typing import List, Tuple, Optional, Union
import numpy as np
import math

from virtualhome.simulation.unity_simulator import comm_unity

NUM_GRID_POINT_IN_BOX = 5
BOX_TO_ID_IOU_THRESH = 0.1
VH_BBOX_DISTANCE_THRESHOLD = 3.0
DEFAULT_VH_IMG_W = 640
DEFAULT_VH_IMG_H = 480
SKIP_ANIMATION = True
RECORDING = False

from scipy.spatial.transform import Rotation as R


def configure_vh_image_size(
    comm: comm_unity.UnityCommunication, img_w: int, img_h: int
) -> None:
    comm.vh_img_w = int(img_w)
    comm.vh_img_h = int(img_h)


def configure_vh_camera_idx(
    comm: comm_unity.UnityCommunication, camera_idx: int
) -> None:
    comm.vh_camera_idx = int(camera_idx)


def get_vh_image_size(comm: comm_unity.UnityCommunication) -> Tuple[int, int]:
    return int(getattr(comm, "vh_img_w", DEFAULT_VH_IMG_W)), int(
        getattr(comm, "vh_img_h", DEFAULT_VH_IMG_H)
    )


def get_vh_camera_idx(comm: comm_unity.UnityCommunication) -> int:
    if hasattr(comm, "vh_camera_idx"):
        return int(comm.vh_camera_idx)
    success, count = comm.camera_count()
    if not success or count <= 0:
        raise RuntimeError("VH camera is not available")
    return count - 1


def vh_camera_image(
    comm: comm_unity.UnityCommunication,
    mode: str,
    camera_idx: Optional[int] = None,
):
    if camera_idx is None:
        camera_idx = get_vh_camera_idx(comm)
    img_w, img_h = get_vh_image_size(comm)
    return comm.camera_image(
        [camera_idx],
        mode=mode,
        image_width=img_w,
        image_height=img_h,
    )


def vh_environment_graph(comm: comm_unity.UnityCommunication) -> dict:
    success, graph = comm.environment_graph()
    if not success:
        raise RuntimeError("Failed to get VH environment graph")
    return graph


def vh_instance_colors(comm: comm_unity.UnityCommunication) -> dict:
    success, colors = comm.instance_colors()
    if not success:
        raise RuntimeError("Failed to get VH instance colors")
    return colors


def update_camera_lookat(comm, target_pos_list):
    target_pos = np.array(target_pos_list)

    success, graph = comm.environment_graph()
    if not success:
        print("Error: Failed to get environment graph")
        return

    char_node = None
    for node in graph["nodes"]:
        if node["class_name"] == "character":
            char_node = node
            break

    if char_node is None:
        print("Error: No character found in the scene")
        return

    char_pos_world = np.array(char_node["obj_transform"]["position"])

    char_rot_quat = char_node["obj_transform"]["rotation"]
    r_char = R.from_quat(char_rot_quat)

    cam_offset_local = np.array([0, 1.5, 0])

    cam_pos_world = char_pos_world + r_char.apply(cam_offset_local)

    look_dir = target_pos - cam_pos_world

    if np.linalg.norm(look_dir) < 1e-6:
        print("Warning: Camera is at the target position")
        return
    look_dir = look_dir / np.linalg.norm(look_dir)

    look_dir_local = r_char.inv().apply(look_dir)
    dx_local, dy_local, dz_local = look_dir_local

    yaw = math.degrees(math.atan2(dx_local, dz_local))
    dist_xz = math.hypot(dx_local, dz_local)
    pitch = -math.degrees(math.atan2(dy_local, dist_xz))
    roll = 0.0

    local_euler = np.array([pitch, yaw, roll], dtype=float)

    success, count = comm.camera_count()
    if success and count > 0:
        target_cam_index = get_vh_camera_idx(comm)

        ok = comm.update_camera(
            camera_index=target_cam_index,
            position=[0, 1.5, 0],
            rotation=local_euler.tolist(),
            field_view=60,
        )
        if ok:
            return True
        else:
            return False


def id_int_to_object_dict(sg: dict[str, dict[str]], object_id_int: int):
    for x in sg["nodes"]:
        if x["id"] == object_id_int:
            return x
    return None


def _target_failure_feedback(
    sg: dict,
    object_id: int,
    required_property: Optional[str] = None,
    missing_property_feedback: str = "",
):
    target = id_int_to_object_dict(sg, object_id)
    if target is None:
        return ""
    property_exceptions = {
        "GRABBABLE": {"water", "child"},
        "CAN_OPEN": {"desk", "window"},
    }
    if (
        required_property is not None
        and required_property not in target.get("properties", [])
        and target["class_name"].lower()
        not in property_exceptions.get(required_property, set())
    ):
        return missing_property_feedback
    if not any(
        edge["from_id"] == 1
        and edge["to_id"] == object_id
        and edge["relation_type"] == "CLOSE"
        for edge in sg["edges"]
    ):
        return "The target is too far away."
    return ""


def resolve_object(
    obj_ref: Union[str, List[float]], comm: comm_unity.UnityCommunication
):
    if isinstance(obj_ref, str):
        return obj_ref
    elif isinstance(obj_ref, List) and isinstance(obj_ref[0], float):
        if len(obj_ref) == 2:
            return pixel_to_ID(comm, obj_ref)
        elif len(obj_ref) == 4:
            return box_to_ID(comm, obj_ref)
        else:
            raise ValueError(f"Unknown object reference: {obj_ref}")
    elif isinstance(obj_ref, List) and isinstance(obj_ref[0], str):
        if len(obj_ref[1]) == 2:
            return pixel_to_ID(comm, obj_ref[1], obj_ref[0])
        elif len(obj_ref[1]) == 4:
            return box_to_ID(comm, obj_ref[1], obj_ref[0])
        else:
            raise ValueError(f"Unknown object reference: {obj_ref}")
    else:
        raise ValueError(f"Unknown object reference: {obj_ref}")


def refine_visible_objects(
    comm: comm_unity.UnityCommunication,
    camera_idx: int,
    seg: np.ndarray,
    instance_colors: dict,
) -> dict[str, np.ndarray]:
    _visible_objects = comm.get_visible_objects(camera_idx)[1]
    vhid2node = {str(node["id"]): node for node in vh_environment_graph(comm)["nodes"]}

    seg_colors = np.unique(seg.reshape(-1, 3)[:, ::-1], axis=0) / 255.0
    vis_objs = {}
    seen_colors = set()
    for oid in _visible_objects:
        if oid in instance_colors:
            color = instance_colors[oid]
            for c in seg_colors:
                if (
                    abs(c[0] - color[0]) < 0.01
                    and abs(c[1] - color[1]) < 0.01
                    and abs(c[2] - color[2]) < 0.01
                ):
                    vis_objs[vhid2node[oid]["class_name"] + "_" + str(oid)] = c
                    seen_colors.add(tuple(c))
                    break

    for c in seg_colors:
        if tuple(c) == (0, 0, 0):
            continue
        if tuple(c) not in seen_colors:
            cand_oids = [
                [x, vhid2node[x]["class_name"]]
                for x, y in instance_colors.items()
                if x != "1"
                and abs(c[0] - y[0]) < 0.01
                and abs(c[1] - y[1]) < 0.01
                and abs(c[2] - y[2]) < 0.01
                and vhid2node[x]["class_name"] not in ["wall", "floor", "ceiling"]
            ]
            if len(cand_oids) == 1:
                vis_objs[
                    vhid2node[cand_oids[0][0]]["class_name"] + "_" + cand_oids[0][0]
                ] = c
            else:
                agent_pos = vhid2node["1"]["bounding_box"]["center"]

                min_dist = float("inf")
                min_oid = None
                for oid, class_name in cand_oids:
                    node_pos = vhid2node[oid]["bounding_box"]["center"]
                    dist = (
                        (agent_pos[0] - node_pos[0]) ** 2
                        + (agent_pos[1] - node_pos[1]) ** 2
                        + (agent_pos[2] - node_pos[2]) ** 2
                    ) ** 0.5
                    if dist < min_dist:
                        if dist < VH_BBOX_DISTANCE_THRESHOLD:
                            min_dist = dist
                            min_oid = oid
                if min_oid is not None:
                    vis_objs[vhid2node[min_oid]["class_name"] + "_" + min_oid] = c
    return vis_objs


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


def build_visible_instance_bboxs(
    comm: comm_unity.UnityCommunication,
    camera_idx: Optional[int] = None,
) -> Tuple[dict[str, Tuple[int, int, int, int]], int, int]:
    if camera_idx is None:
        camera_idx = get_vh_camera_idx(comm)
    instance_seg_frame = vh_camera_image(comm, mode="seg_inst", camera_idx=camera_idx)[
        1
    ][0]
    instance_colors = vh_instance_colors(comm)
    vhid2node = {str(node["id"]): node for node in vh_environment_graph(comm)["nodes"]}
    img_h, img_w = instance_seg_frame.shape[:2]

    seg_colors = np.unique(instance_seg_frame.reshape(-1, 3)[:, ::-1], axis=0) / 255.0
    instance_bboxs = {}
    for color in seg_colors:
        if tuple(color) == (0, 0, 0):
            continue
        object_id_int = None
        for o, c_tmp in instance_colors.items():
            if (
                abs(c_tmp[0] - color[0]) < 0.01
                and abs(c_tmp[1] - color[1]) < 0.01
                and abs(c_tmp[2] - color[2]) < 0.01
            ):
                object_id_int = o
                break
        if object_id_int is None or str(object_id_int) == "1":
            continue
        if str(object_id_int) not in vhid2node:
            continue
        oid = f"{vhid2node[str(object_id_int)]['class_name']}_{object_id_int}"
        mask = np.all(
            instance_seg_frame[..., ::-1] == (np.array(color) * 255).astype(np.uint8),
            axis=-1,
        )
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue
        instance_bboxs[oid] = (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        )
    return instance_bboxs, img_h, img_w


def pixel_to_ID(
    comm: comm_unity.UnityCommunication,
    pixel: List[float],
    obj_cls: Optional[str] = None,
):
    pixel_x, pixel_y = pixel

    instance_seg_frame = vh_camera_image(comm, mode="seg_inst")[1][0]
    img_h, img_w = instance_seg_frame.shape[:2]
    px = max(0, min(int(float(pixel_x) * img_w), img_w - 1))
    py = max(0, min(int(float(pixel_y) * img_h), img_h - 1))
    color = instance_seg_frame[py, px, ::-1] / 255.0
    color_to_object_id = vh_instance_colors(comm)
    object_id_int = None
    for o, c_tmp in color_to_object_id.items():
        if (
            abs(c_tmp[0] - color[0]) < 0.01
            and abs(c_tmp[1] - color[1]) < 0.01
            and abs(c_tmp[2] - color[2]) < 0.01
        ):
            object_id_int = int(o)
            break
    assert object_id_int is not None
    object_dict = id_int_to_object_dict(vh_environment_graph(comm), object_id_int)
    assert object_dict is not None

    object_id = f"{object_dict['class_name']}_{object_id_int}"
    return object_id


def box_to_ID(
    comm: comm_unity.UnityCommunication, box: List[float], obj_cls: Optional[str] = None
):

    camera_idx = get_vh_camera_idx(comm)
    instance_bboxs, img_h, img_w = build_visible_instance_bboxs(comm, camera_idx)
    pred_box = _norm_bbox_to_pixel_xyxy(box, img_w, img_h)

    best_iou = -1.0
    best_obj_id = None
    for obj_id, gt_box in instance_bboxs.items():
        if obj_cls is not None and not match_object_cls(obj_cls, obj_id):
            continue
        iou = _axis_iou_inclusive_xyxy(pred_box, gt_box)
        if iou > best_iou:
            best_iou = iou
            best_obj_id = obj_id

    if best_obj_id is None or best_iou <= BOX_TO_ID_IOU_THRESH:
        return None
    return best_obj_id


def point_to_ori(target_pos: List[float], camera_pos: List[float]):
    dx = target_pos[0] - camera_pos[0]
    dy = target_pos[1] - camera_pos[1]
    dz = target_pos[2] - camera_pos[2]

    # Yaw: left-right rotation
    yaw = math.degrees(math.atan2(dx, dz))
    # Horizontal distance
    dist_xz = math.sqrt(dx * dx + dz * dz)
    # Pitch: up-down rotation (negative = look up in Unity/AI2-THOR)
    pitch = -math.degrees(math.atan2(dy, dist_xz))
    roll = 0.0
    return yaw, pitch, roll


def parse_object_id(object_id: str) -> Tuple[str, str]:
    """Split ``object_id`` into (class_name, id_str): id is after the last ``_``, class is everything before it."""
    parts = object_id.rsplit("_", 1)
    if len(parts) != 2:
        print(f"Invalid object_id (expected <class>_<id>): {object_id!r}")
        return object_id, "0"
    return parts[0], parts[1]


def match_object_cls(query_cls: str, obj_ID: str):
    obj_cls_part = obj_ID.rsplit("_", 1)[0] if "_" in obj_ID else obj_ID
    return query_cls.lower().replace(" ", "") == obj_cls_part.lower().replace(" ", "")


def calculate_distance_2D(p1: dict[str, float], p2: dict[str, float]):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2)


def calculate_distance_3D(p1: dict[str, float], p2: dict[str, float]):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


# low-level


def moveAhead(comm: comm_unity.UnityCommunication):
    ret, msg = comm.render_script(
        script=["<char0> [walkforward]"],
        recording=RECORDING,
        skip_animation=SKIP_ANIMATION,
        random_seed=0,
    )
    return ret, "" if ret else "The movement is blocked."


def turnLeft(comm: comm_unity.UnityCommunication):
    ret, msg = comm.render_script(
        script=["<char0> [turnleft]"],
        recording=RECORDING,
        skip_animation=SKIP_ANIMATION,
        random_seed=0,
    )
    msg = ""
    return ret, msg


def turnRight(comm: comm_unity.UnityCommunication):
    ret, msg = comm.render_script(
        script=["<char0> [turnright]"],
        recording=RECORDING,
        skip_animation=SKIP_ANIMATION,
        random_seed=0,
    )
    msg = ""
    return ret, msg


# mid-level


def turnTo_obj(
    comm: comm_unity.UnityCommunication, object_ref: Union[str, List[float]]
):
    object_id = resolve_object(object_ref, comm)
    if object_id is None:
        return False, f"Invalid action argument"

    cam_pos = None
    obj_center = None
    vheg = vh_environment_graph(comm)
    for node in vheg["nodes"]:
        if int(node["id"]) == int(object_id.rsplit("_", 1)[-1]):
            obj_center = node["bounding_box"]["center"]
            obj_pos = node["obj_transform"]["position"]
        if node["category"] == "Characters":
            cam_pos = node["obj_transform"]["position"]
    cam_pos[1] += 1.5

    assert obj_center is not None

    ox, oy, oz = obj_center

    if oy < 0 or oy > 3:
        oy = 0.9
    ret = update_camera_lookat(comm, [ox, oy, oz])

    if ret:
        return True, ""
    return False, ""


def goTo_obj(comm: comm_unity.UnityCommunication, object_ref: Union[str, List[float]]):
    object_id = resolve_object(object_ref, comm)
    if object_id is None:
        return False, f"Invalid action argument"

    obj_cls, obj_id_int = parse_object_id(object_id)

    vheg = vh_environment_graph(comm)

    already_closed = False
    for e in vheg["edges"]:
        if (
            e["from_id"] == 1
            and e["to_id"] == int(obj_id_int)
            and e["relation_type"] == "CLOSE"
        ):
            already_closed = True
            break
    if not already_closed:
        ret, msg = comm.render_script(
            script=[f"<char0> [run] <{obj_cls}> ({obj_id_int})"],
            recording=RECORDING,
            skip_animation=SKIP_ANIMATION,
            random_seed=0,
        )
        msg = ""
        if not ret:
            return ret, msg

    ret, msg = turnTo_obj(comm, object_id)
    return ret, msg


def pick_obj(comm: comm_unity.UnityCommunication, object_ref: Union[str, List[float]]):
    object_id = resolve_object(object_ref, comm)
    if object_id is None:
        return False, f"Invalid action argument"

    obj_cls, obj_id_int = parse_object_id(object_id)
    sg = vh_environment_graph(comm)
    if any(
        edge["from_id"] == 1
        and edge["relation_type"] in ["HOLDS_LH", "HOLDS_RH"]
        for edge in sg["edges"]
    ):
        return False, "The agent is already holding an object."

    ret, msg = comm.render_script(
        script=[f"<char0> [grab] <{obj_cls}> ({obj_id_int})"],
        recording=RECORDING,
        skip_animation=SKIP_ANIMATION,
        random_seed=0,
    )
    if ret:
        return True, ""

    return False, _target_failure_feedback(
        vh_environment_graph(comm),
        int(obj_id_int),
        "GRABBABLE",
        "The target object is not graspable.",
    )


def placeLeftOn_recep(
    comm: comm_unity.UnityCommunication, object_ref: Union[str, List[float]]
):
    invent_obj_id_int = None
    sg = vh_environment_graph(comm)
    for x in sg["edges"]:
        if x["from_id"] == 1 and x["relation_type"] == "HOLDS_LH":
            invent_obj_id_int = x["to_id"]
            break
    if invent_obj_id_int is None:
        return False, "The agent is not holding an object."

    invent_obj_cls = id_int_to_object_dict(sg, invent_obj_id_int)["class_name"]

    object_id = resolve_object(object_ref, comm)
    if object_id is None:
        return False, f"Invalid action argument"

    obj_cls, obj_id_int = parse_object_id(object_id)
    ret, msg = comm.render_script(
        script=[
            f"<char0> [putback] <{invent_obj_cls}> ({invent_obj_id_int}) <{obj_cls}> ({obj_id_int})"
        ],
        recording=RECORDING,
        skip_animation=SKIP_ANIMATION,
        random_seed=0,
    )
    if ret:
        turnTo_obj(comm, f"{invent_obj_cls}_{invent_obj_id_int}")
        return True, ""
    return False, _target_failure_feedback(
        vh_environment_graph(comm),
        int(obj_id_int),
    )


def placeLeftIn_recep(
    comm: comm_unity.UnityCommunication, object_ref: Union[str, List[float]]
):
    invent_obj_id_int = None
    sg = vh_environment_graph(comm)
    for x in sg["edges"]:
        if x["from_id"] == 1 and x["relation_type"] == "HOLDS_LH":
            invent_obj_id_int = x["to_id"]
            break
    if invent_obj_id_int is None:
        return False, "The agent is not holding an object."

    invent_obj_cls = id_int_to_object_dict(sg, invent_obj_id_int)["class_name"]

    object_id = resolve_object(object_ref, comm)
    if object_id is None:
        return False, f"Invalid action argument"

    obj_cls, obj_id_int = parse_object_id(object_id)
    ret, msg = comm.render_script(
        script=[
            f"<char0> [putin] <{invent_obj_cls}> ({invent_obj_id_int}) <{obj_cls}> ({obj_id_int})"
        ],
        recording=RECORDING,
        skip_animation=SKIP_ANIMATION,
        random_seed=0,
    )
    if ret:
        turnTo_obj(comm, f"{invent_obj_cls}_{invent_obj_id_int}")
        return True, ""
    return False, _target_failure_feedback(
        vh_environment_graph(comm),
        int(obj_id_int),
    )


def placeRightOn_recep(
    comm: comm_unity.UnityCommunication, object_ref: Union[str, List[float]]
):
    invent_obj_id_int = None
    sg = vh_environment_graph(comm)
    for x in sg["edges"]:
        if x["from_id"] == 1 and x["relation_type"] == "HOLDS_RH":
            invent_obj_id_int = x["to_id"]
            break
    if invent_obj_id_int is None:
        return False, "The agent is not holding an object."

    invent_obj_cls = id_int_to_object_dict(sg, invent_obj_id_int)["class_name"]

    object_id = resolve_object(object_ref, comm)
    if object_id is None:
        return False, f"Invalid action argument"

    obj_cls, obj_id_int = parse_object_id(object_id)
    ret, msg = comm.render_script(
        script=[
            f"<char0> [putback] <{invent_obj_cls}> ({invent_obj_id_int}) <{obj_cls}> ({obj_id_int})"
        ],
        recording=RECORDING,
        skip_animation=SKIP_ANIMATION,
        random_seed=0,
    )
    if ret:
        turnTo_obj(comm, f"{invent_obj_cls}_{invent_obj_id_int}")
        return True, ""
    return False, _target_failure_feedback(
        vh_environment_graph(comm),
        int(obj_id_int),
    )


def placeRightIn_recep(
    comm: comm_unity.UnityCommunication, object_ref: Union[str, List[float]]
):
    invent_obj_id_int = None
    sg = vh_environment_graph(comm)
    for x in sg["edges"]:
        if x["from_id"] == 1 and x["relation_type"] == "HOLDS_RH":
            invent_obj_id_int = x["to_id"]
            break
    if invent_obj_id_int is None:
        return False, "The agent is not holding an object."

    invent_obj_cls = id_int_to_object_dict(sg, invent_obj_id_int)["class_name"]

    object_id = resolve_object(object_ref, comm)
    if object_id is None:
        return False, f"Invalid action argument"

    obj_cls, obj_id_int = parse_object_id(object_id)
    ret, msg = comm.render_script(
        script=[
            f"<char0> [putin] <{invent_obj_cls}> ({invent_obj_id_int}) <{obj_cls}> ({obj_id_int})"
        ],
        recording=RECORDING,
        skip_animation=SKIP_ANIMATION,
        random_seed=0,
    )
    if ret:
        turnTo_obj(comm, f"{invent_obj_cls}_{invent_obj_id_int}")
        return True, ""
    return False, _target_failure_feedback(
        vh_environment_graph(comm),
        int(obj_id_int),
    )


def open_obj(comm: comm_unity.UnityCommunication, object_ref: Union[str, List[float]]):
    object_id = resolve_object(object_ref, comm)
    if object_id is None:
        return False, f"Invalid action argument"

    obj_cls, obj_id_int = parse_object_id(object_id)
    ret, msg = comm.render_script(
        script=[f"<char0> [open] <{obj_cls}> ({obj_id_int})"],
        recording=RECORDING,
        skip_animation=SKIP_ANIMATION,
        random_seed=0,
    )

    if ret:
        turnTo_obj(comm, object_ref)
        return True, ""
    return False, _target_failure_feedback(
        vh_environment_graph(comm),
        int(obj_id_int),
        "CAN_OPEN",
        "The target object is not openable.",
    )


def close_obj(comm: comm_unity.UnityCommunication, object_ref: Union[str, List[float]]):
    object_id = resolve_object(object_ref, comm)
    if object_id is None:
        return False, f"Invalid action argument"

    obj_cls, obj_id_int = parse_object_id(object_id)
    ret, msg = comm.render_script(
        script=[f"<char0> [close] <{obj_cls}> ({obj_id_int})"],
        recording=RECORDING,
        skip_animation=SKIP_ANIMATION,
        random_seed=0,
    )
    if ret:
        turnTo_obj(comm, object_ref)
        return True, ""
    return False, _target_failure_feedback(
        vh_environment_graph(comm),
        int(obj_id_int),
        "CAN_OPEN",
        "The target object is not openable.",
    )


def turnOn_obj(
    comm: comm_unity.UnityCommunication, object_ref: Union[str, List[float]]
):
    object_id = resolve_object(object_ref, comm)
    if object_id is None:
        return False, f"Invalid action argument"

    obj_cls, obj_id_int = parse_object_id(object_id)
    ret, msg = comm.render_script(
        script=[f"<char0> [switchon] <{obj_cls}> ({obj_id_int})"],
        recording=RECORDING,
        skip_animation=SKIP_ANIMATION,
        random_seed=0,
    )
    if ret:
        return True, ""
    return False, _target_failure_feedback(
        vh_environment_graph(comm),
        int(obj_id_int),
        "HAS_SWITCH",
        "The target object is not toggleable.",
    )


def turnOff_obj(
    comm: comm_unity.UnityCommunication, object_ref: Union[str, List[float]]
):
    object_id = resolve_object(object_ref, comm)
    if object_id is None:
        return False, f"Invalid action argument"

    obj_cls, obj_id_int = parse_object_id(object_id)
    ret, msg = comm.render_script(
        script=[f"<char0> [switchoff] <{obj_cls}> ({obj_id_int})"],
        recording=RECORDING,
        skip_animation=SKIP_ANIMATION,
        random_seed=0,
    )
    if ret:
        return True, ""
    return False, _target_failure_feedback(
        vh_environment_graph(comm),
        int(obj_id_int),
        "HAS_SWITCH",
        "The target object is not toggleable.",
    )


# high-level


def find_objCls(
    comm: comm_unity.UnityCommunication, object_cls: str, by_dist: bool = True
):

    sg = vh_environment_graph(comm)

    candidates = []
    for obj_dict in sg["nodes"]:
        if match_object_cls(object_cls, obj_dict["class_name"]):
            all_parents = []
            while True:
                new = 0
                for x in sg["edges"]:
                    if (
                        x["from_id"] in all_parents or x["from_id"] == obj_dict["id"]
                    ) and x["relation_type"] == "INSIDE":
                        if x["to_id"] not in all_parents:
                            new += 1
                            all_parents.append(x["to_id"])
                if new == 0:
                    break
            Flag = True
            for x in sg["nodes"]:
                if (
                    x["id"] in all_parents
                    and "CAN_OPEN" in x["properties"]
                    and "CLOSED" in x["states"]
                ):
                    Flag = False
                    break
            if Flag:
                candidates.append(
                    [
                        obj_dict["id"],
                        obj_dict["obj_transform"]["position"],
                        obj_dict["class_name"],
                    ]
                )

    if len(candidates) == 0:
        return False, "No accessible object of the requested class was found."

    if by_dist:
        assert sg["nodes"][0]["id"] == 1, sg["nodes"][0]
        obj = min(
            candidates,
            key=lambda x: calculate_distance_2D(
                x[1], sg["nodes"][0]["obj_transform"]["position"]
            ),
        )
        object_id = f"{obj[2]}_{obj[0]}"
    else:
        obj = np.random.choice(candidates)
        object_id = f"{obj[2]}_{obj[0]}"
    return goTo_obj(comm, object_id)
