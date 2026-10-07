from ai2thor.controller import Controller
from typing import List, Tuple, Optional, Union
import numpy as np
import math

from task.commonsense_knowledge.thor import (
    OBJ_HEATING,
    OBJ_COOLING,
    OBJ_WASHING,
    OBJ_FILLING_WATER,
    OBJ_FILLING_COFFEE,
    OBJ_FILLING_WINE,
)

NUM_GRID_POINT_IN_BOX = 5
BOX_TO_ID_IOU_THRESH = 0.1
DEFAULT_THROW_FORCE = 150.0
LOOK_UP_DOWN_DEGREE = 30.0
CAMERA_HEIGHT_OFFSET = 0.6
NUM_TRY_TELEPORT = 5
DEFAULT_MOVE_HAND_MAGNITUDE = 0.2
DEFAULT_ROTATE_HAND_ANGLE = 30


def position_in_room(position: dict[str, float], floor_polygon: List[dict]) -> bool:
    """Return whether an x-z position is inside a ProcTHOR room polygon."""
    x, z = position["x"], position["z"]
    inside = False
    for i, point_a in enumerate(floor_polygon):
        point_b = floor_polygon[i - 1]
        ax, az = point_a["x"], point_a["z"]
        bx, bz = point_b["x"], point_b["z"]

        cross = (x - ax) * (bz - az) - (z - az) * (bx - ax)
        if (
            abs(cross) < 1e-6
            and min(ax, bx) - 1e-6 <= x <= max(ax, bx) + 1e-6
            and min(az, bz) - 1e-6 <= z <= max(az, bz) + 1e-6
        ):
            return True

        if (az > z) != (bz > z):
            intersect_x = ax + (z - az) * (bx - ax) / (bz - az)
            if x < intersect_x:
                inside = not inside
    return inside


def resolve_object(obj_ref: Union[str, List[float]], controller: Controller):
    if isinstance(obj_ref, str):
        return obj_ref
    elif isinstance(obj_ref, List) and isinstance(obj_ref[0], float):
        if len(obj_ref) == 2:
            return pixel_to_ID(controller, obj_ref)
        elif len(obj_ref) == 4:
            return box_to_ID(controller, obj_ref)
        else:
            raise ValueError(f"Unknown object reference: {obj_ref}")
    elif isinstance(obj_ref, List) and isinstance(obj_ref[0], str):
        if len(obj_ref[1]) == 2:
            return pixel_to_ID(controller, obj_ref[1], obj_ref[0])
        elif len(obj_ref[1]) == 4:
            return box_to_ID(controller, obj_ref[1], obj_ref[0])
        else:
            raise ValueError(f"Unknown object reference: {obj_ref}")
    else:
        raise ValueError(f"Unknown object reference: {obj_ref}")


def pixel_to_ID(
    controller: Controller, pixel: List[float], obj_cls: Optional[str] = None
):
    pixel_x, pixel_y = pixel

    img_h, img_w = controller.last_event.instance_segmentation_frame.shape[:2]
    color = controller.last_event.instance_segmentation_frame[
        int(pixel_y * img_h), int(pixel_x * img_w)
    ]
    object_id = controller.last_event.color_to_object_id[tuple(color)]

    return object_id


def _is_irrelevant_thor_object_id(object_id: str) -> bool:
    obj_lower = object_id.lower()
    return (
        obj_lower.startswith("wall")
        or obj_lower.startswith("ceiling")
        or obj_lower.startswith("floor")
        or obj_lower.startswith("fp")
    )


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


def box_to_ID(controller: Controller, box: List[float], obj_cls: Optional[str] = None):

    event = controller.last_event
    img_h, img_w = event.frame.shape[:2]
    pred_box = _norm_bbox_to_pixel_xyxy(box, img_w, img_h)

    best_iou = -1.0
    best_obj_id = None
    for obj_id, bbox in dict(event.instance_detections2D).items():
        obj_id = str(obj_id)
        if _is_irrelevant_thor_object_id(obj_id):
            continue
        if bbox is None:
            continue
        gt_box = tuple(int(x) for x in bbox)
        if obj_cls is not None and not match_object_cls(obj_cls, obj_id):
            continue
        iou = _axis_iou_inclusive_xyxy(pred_box, gt_box)
        if iou > best_iou:
            best_iou = iou
            best_obj_id = obj_id

    if best_obj_id is None or best_iou <= BOX_TO_ID_IOU_THRESH:
        return None
    return best_obj_id


def point_to_rotation_and_horizon(
    target_pos: dict[str, float], agent_pos: dict[str, float]
):
    dx = target_pos["x"] - agent_pos["x"]
    dy = target_pos["y"] - (agent_pos["y"] + CAMERA_HEIGHT_OFFSET)
    dz = target_pos["z"] - agent_pos["z"]

    # yaw (agent body rotation around y-axis)
    # we use atan2(dx, dz) because forward is aligned with +z in Unity-like coords
    desired_yaw = math.degrees(math.atan2(dx, dz))

    # pitch: angle above/below horizontal
    horizontal_dist = math.sqrt(dx * dx + dz * dz)
    desired_pitch = math.degrees(math.atan2(dy, horizontal_dist))

    # AI2-THOR uses horizon: negative = look up, positive = look down
    target_horizon = -desired_pitch
    return desired_yaw, max(min(target_horizon, 60), -30)


def match_object_cls(query_cls: str, obj_ID: str):
    if "Sliced" in obj_ID:
        return query_cls.lower() == obj_ID.lower().split("|")[-1].split("_")[0]
    if "Cracked" in obj_ID:
        return query_cls.lower() == obj_ID.lower().split("|")[-1].split("_")[0]
    return query_cls.lower().replace(" ", "") == obj_ID.split("|")[0].lower().replace(
        " ", ""
    )


def calculate_distance_2D(p1: dict[str, float], p2: dict[str, float]):
    return math.sqrt((p1["x"] - p2["x"]) ** 2 + (p1["z"] - p2["z"]) ** 2)


def calculate_distance_3D(p1: dict[str, float], p2: dict[str, float]):
    return math.sqrt(
        (p1["x"] - p2["x"]) ** 2
        + (p1["y"] - (p2["y"] + CAMERA_HEIGHT_OFFSET)) ** 2
        + (p1["z"] - p2["z"]) ** 2
    )


def process_error_msg(error_msg: str):
    if "Target object not found within the specified visibility" in error_msg:
        return "The target is not visible or is too far away."
    if "is not an Openable object" in error_msg:
        return "The target object is not openable."
    if "is not toggleable" in error_msg:
        return "The target object is not toggleable."
    if "must have the property CanPickup to be picked up" in error_msg:
        return "The target object is not graspable."
    return ""


# low-level


def moveAhead(controller: Controller):
    controller.step(action="MoveAhead")
    success = controller.last_event.metadata["lastActionSuccess"]
    return success, "" if success else "The movement is blocked."


def moveBack(controller: Controller):
    controller.step(action="MoveBack")
    success = controller.last_event.metadata["lastActionSuccess"]
    return success, "" if success else "The movement is blocked."


def moveLeft(controller: Controller):
    controller.step(action="MoveLeft")
    success = controller.last_event.metadata["lastActionSuccess"]
    return success, "" if success else "The movement is blocked."


def moveRight(controller: Controller):
    controller.step(action="MoveRight")
    success = controller.last_event.metadata["lastActionSuccess"]
    return success, "" if success else "The movement is blocked."


def turnRight(controller: Controller):
    controller.step(
        action="RotateRight",
        degrees=None,  # defined by rotateStepDegrees
    )
    return controller.last_event.metadata["lastActionSuccess"], process_error_msg(
        controller.last_event.metadata["errorMessage"]
    )


def turnLeft(controller: Controller):
    controller.step(
        action="RotateLeft",
        degrees=None,  # defined by rotateStepDegrees
    )
    return controller.last_event.metadata["lastActionSuccess"], process_error_msg(
        controller.last_event.metadata["errorMessage"]
    )


def lookUp(controller: Controller):
    controller.step(
        action="LookUp",
        degrees=LOOK_UP_DOWN_DEGREE,
    )
    return controller.last_event.metadata["lastActionSuccess"], process_error_msg(
        controller.last_event.metadata["errorMessage"]
    )


def lookDown(controller: Controller):
    controller.step(
        action="LookDown",
        degrees=LOOK_UP_DOWN_DEGREE,
    )
    return controller.last_event.metadata["lastActionSuccess"], process_error_msg(
        controller.last_event.metadata["errorMessage"]
    )


def moveHandAhead(
    controller: Controller, moveMagnitude: float = DEFAULT_MOVE_HAND_MAGNITUDE
):
    controller.step(
        action="MoveHeldObjectAhead",
        moveMagnitude=moveMagnitude,
    )
    return controller.last_event.metadata["lastActionSuccess"], process_error_msg(
        controller.last_event.metadata["errorMessage"]
    )


def moveHandBack(
    controller: Controller, moveMagnitude: float = DEFAULT_MOVE_HAND_MAGNITUDE
):
    controller.step(
        action="MoveHeldObjectBack",
        moveMagnitude=moveMagnitude,
    )
    return controller.last_event.metadata["lastActionSuccess"], process_error_msg(
        controller.last_event.metadata["errorMessage"]
    )


def moveHandLeft(
    controller: Controller, moveMagnitude: float = DEFAULT_MOVE_HAND_MAGNITUDE
):
    controller.step(
        action="MoveHeldObjectLeft",
        moveMagnitude=moveMagnitude,
    )
    return controller.last_event.metadata["lastActionSuccess"], process_error_msg(
        controller.last_event.metadata["errorMessage"]
    )


def moveHandRight(
    controller: Controller, moveMagnitude: float = DEFAULT_MOVE_HAND_MAGNITUDE
):
    controller.step(
        action="MoveHeldObjectRight",
        moveMagnitude=moveMagnitude,
    )
    return controller.last_event.metadata["lastActionSuccess"], process_error_msg(
        controller.last_event.metadata["errorMessage"]
    )


def rotateHand(controller: Controller, rotateAngle: float = DEFAULT_ROTATE_HAND_ANGLE):
    controller.step(
        action="RotateHeldObject",
        roll=rotateAngle,
    )
    return controller.last_event.metadata["lastActionSuccess"], process_error_msg(
        controller.last_event.metadata["errorMessage"]
    )


# mid-level


def turnTo_obj(controller: Controller, object_ref: Union[str, List[float]]):
    object_id = resolve_object(object_ref, controller)
    if object_id is None:
        return False, f"Invalid action argument"

    obj_center = None
    for obj_dict in controller.last_event.metadata["objects"]:
        if obj_dict["objectId"] == object_id:
            obj_center = obj_dict["position"]
            break
    if obj_center is None:
        return False, f"Cannot find any object in the given area"

    agent_position = controller.last_event.metadata["agent"]["position"]
    rotation, horizon = point_to_rotation_and_horizon(obj_center, agent_position)

    controller.step(
        action="TeleportFull",
        position=agent_position,
        rotation=dict(x=0, y=rotation, z=0),
        horizon=horizon,
        standing=True,
    )
    return controller.last_event.metadata["lastActionSuccess"], process_error_msg(
        controller.last_event.metadata["errorMessage"]
    )


def turnTo_point(controller: Controller, target_position_pixel: List[float]):
    query = controller.step(
        action="GetCoordinateFromRaycast",
        x=target_position_pixel[0],
        y=target_position_pixel[1],
    )
    coordinate = query.metadata["actionReturn"]
    agent_position = controller.last_event.metadata["agent"]["position"]
    rotation, horizon = point_to_rotation_and_horizon(coordinate, agent_position)
    controller.step(
        action="TeleportFull",
        position=agent_position,
        rotation=dict(x=0, y=rotation, z=0),
        horizon=horizon,
        standing=True,
    )
    return controller.last_event.metadata["lastActionSuccess"], process_error_msg(
        controller.last_event.metadata["errorMessage"]
    )


def goTo_loc(controller: Controller, location: dict[str, float]):
    controller.step(action="Teleport", position=location)
    return controller.last_event.metadata["lastActionSuccess"], process_error_msg(
        controller.last_event.metadata["errorMessage"]
    )


def goTo_obj(
    controller: Controller,
    object_ref: Union[str, List[float]],
    possible_locations: Optional[List[dict[str, float]]] = None,
    room_polygons: Optional[dict[str, List[dict]]] = None,
):
    object_id = resolve_object(object_ref, controller)
    if object_id is None:
        return False, f"Invalid action argument"

    if object_id.startswith("room|"):
        floor_polygon = (room_polygons or {}).get(object_id)
        if floor_polygon is None or possible_locations is None:
            return False, "No valid navigation position was found."
        valid_locations = [
            position
            for position in possible_locations
            if position_in_room(position, floor_polygon)
        ]
        if len(valid_locations) == 0:
            return False, "No valid navigation position was found."
        for idx in np.random.permutation(len(valid_locations))[:NUM_TRY_TELEPORT]:
            controller.step(action="Teleport", position=valid_locations[idx])
            if controller.last_event.metadata["lastActionSuccess"]:
                agent_position = controller.last_event.metadata["agent"]["position"]
                if position_in_room(agent_position, floor_polygon):
                    return True, ""
        return False, ""

    event = controller.step(
        action="GetInteractablePoses",
        objectId=object_id,
        positions=possible_locations,
        standings=[
            True,
        ],  # only standing, not crouching
    )

    poses = event.metadata["actionReturn"]

    if poses is None or len(poses) == 0:
        return False, "No valid navigation position was found."

    for _ in range(NUM_TRY_TELEPORT):
        pose = np.random.choice(poses)
        controller.step(action="TeleportFull", **pose)
        if controller.last_event.metadata["lastActionSuccess"]:
            turnTo_obj(controller, object_id)
            break
        if not controller.last_event.metadata["errorMessage"].startswith(
            "ArgumentOutOfRangeException"
        ):
            break
    if not controller.last_event.metadata["lastActionSuccess"]:
        return False, ""
    return True, ""


def pick_obj(controller: Controller, object_ref: Union[str, List[float]]):
    object_id = resolve_object(object_ref, controller)
    if object_id is None:
        return False, f"Invalid action argument"
    controller.step(
        action="PickupObject",
        objectId=object_id,
        forceAction=False,
        manualInteract=False,  # teleport object
    )
    success = controller.last_event.metadata["lastActionSuccess"]
    if not success:
        target = next(
            (
                obj
                for obj in controller.last_event.metadata["objects"]
                if obj["objectId"] == object_id
            ),
            None,
        )
        parent_ids = target.get("parentReceptacles") if target is not None else None
        if parent_ids:
            closed_receptacles = {
                obj["objectId"]
                for obj in controller.last_event.metadata["objects"]
                if obj.get("openable") and not obj.get("isOpen")
            }
            if any(parent_id in closed_receptacles for parent_id in parent_ids):
                return False, "The target object is inside a closed receptacle."
    return success, process_error_msg(controller.last_event.metadata["errorMessage"])


def placeTo_recep(controller: Controller, object_ref: Union[str, List[float]]):
    recep_id = resolve_object(object_ref, controller)
    if recep_id is None:
        return False, f"Invalid action argument"

    controller.step(
        action="PutObject",
        objectId=recep_id,
        forceAction=False,
        placeStationary=True,
    )

    if controller.last_event.metadata["lastActionSuccess"]:
        return True, ""

    receptacle = next(
        (
            obj
            for obj in controller.last_event.metadata["objects"]
            if obj["objectId"] == recep_id
        ),
        None,
    )
    if receptacle is not None and not receptacle.get("receptacle"):
        return False, "The target is not a valid receptacle."

    event = controller.step(
        action="GetSpawnCoordinatesAboveReceptacle",
        objectId=recep_id,
        anywhere=False,  # whether the point can be out of view
    )
    recep_positions = event.metadata["actionReturn"]

    if recep_positions is None or len(recep_positions) == 0:
        return False, "No valid placement position was found."

    for _ in range(10):
        chosen_position = np.random.choice(recep_positions)

        invent_obj_id = None
        for obj_dict in controller.last_event.metadata["objects"]:
            if obj_dict["isPickedUp"]:
                invent_obj_id = obj_dict["objectId"]
        if invent_obj_id is None:
            return False, "The agent is not holding an object."

        controller.step(
            action="PlaceObjectAtPoint",
            objectId=invent_obj_id,
            position=chosen_position,
        )
        success, msg = (
            controller.last_event.metadata["lastActionSuccess"],
            controller.last_event.metadata["errorMessage"],
        )

        if success:
            return True, ""

    return False, "No valid placement position was found."


def placeTo_point(
    controller: Controller, target_position_pixel: Optional[List[float]] = None
):
    invent_obj_id = None
    for obj_dict in controller.last_event.metadata["objects"]:
        if obj_dict["isPickedUp"]:
            invent_obj_id = obj_dict["objectId"]
    if invent_obj_id is None:
        return False, "The agent is not holding an object."

    recep_id = pixel_to_ID(controller, target_position_pixel)
    if recep_id is None:
        return False, f"No object found at the target position"

    if (
        invent_obj_id.split("|")[0].lower() == "bread"
        and "sliced" in invent_obj_id.lower()
        and recep_id.split("|")[0].lower() == "toaster"
    ):
        return placeTo_recep(controller, recep_id)
    if (
        invent_obj_id.split("|")[0].lower() in ["mug"]
        and recep_id.split("|")[0].lower() == "coffeemachine"
    ):
        return placeTo_recep(controller, recep_id)
    if recep_id.split("|")[0].lower() == "stoveburner":
        if invent_obj_id.split("|")[0].lower() in ["pan", "pot", "kettle"]:
            return placeTo_recep(controller, recep_id)
        else:
            return False, f"{invent_obj_id} cannot be placed on StoveBurner"
    if recep_id.split("|")[0].lower() == "stove":
        for x in controller.last_event.instance_detections2D:
            if x.split("|")[0].lower() == "stoveburner":
                x1, y1, x2, y2 = controller.last_event.instance_detections2D[x]
                x1 = float(x1 / controller.last_event.frame.shape[1])
                x2 = float(x2 / controller.last_event.frame.shape[1])
                y1 = float(y1 / controller.last_event.frame.shape[0])
                y2 = float(y2 / controller.last_event.frame.shape[0])
                if (
                    target_position_pixel[0] >= x1
                    and target_position_pixel[0] <= x2
                    and target_position_pixel[1] >= y1
                    and target_position_pixel[1] <= y2
                ):
                    if invent_obj_id.split("|")[0].lower() in ["pan", "pot", "kettle"]:
                        return placeTo_recep(controller, x)
                    else:
                        return False, f"{invent_obj_id} cannot be placed on StoveBurner"

    event = controller.step(
        action="GetSpawnCoordinatesAboveReceptacle",
        objectId=recep_id,
        anywhere=False,  # whether the point can be out of view
    )
    recep_positions = event.metadata["actionReturn"]

    if recep_positions is None or len(recep_positions) == 0:
        return False, "The target is not a valid receptacle."

    query = controller.step(
        action="GetCoordinateFromRaycast",
        x=target_position_pixel[0],
        y=target_position_pixel[1],
    )
    target_position = query.metadata["actionReturn"]

    chosen_position = sorted(
        recep_positions, key=lambda x: calculate_distance_3D(x, target_position)
    )[0]

    controller.step(
        action="PlaceObjectAtPoint", objectId=invent_obj_id, position=chosen_position
    )
    return controller.last_event.metadata["lastActionSuccess"], process_error_msg(
        controller.last_event.metadata["errorMessage"]
    )


def drop(
    controller: Controller,
):
    controller.step(action="DropHandObject", forceAction=False)
    return controller.last_event.metadata["lastActionSuccess"], process_error_msg(
        controller.last_event.metadata["errorMessage"]
    )


def throw_force(controller: Controller, force: float = DEFAULT_THROW_FORCE):
    controller.step(action="ThrowObject", moveMagnitude=force, forceAction=False)
    return controller.last_event.metadata["lastActionSuccess"], process_error_msg(
        controller.last_event.metadata["errorMessage"]
    )


def pourTo_recep(controller: Controller, object_ref: Union[str, List[float]]):

    object_id = resolve_object(object_ref, controller)
    if object_id is None:
        return False, f"Invalid action argument"

    invent_obj_id = None
    for obj_dict in controller.last_event.metadata["objects"]:
        if obj_dict["isPickedUp"]:
            invent_obj_id = obj_dict["objectId"]
            if not obj_dict["canFillWithLiquid"]:
                return False, "The held object is not a liquid container."
            liquid = obj_dict["fillLiquid"]

    if invent_obj_id is None:
        return False, "The agent is not holding an object."

    if liquid is None:
        return False, "The held container is empty."

    obj_category = None
    for obj_dict in controller.last_event.metadata["objects"]:
        if obj_dict["objectId"] == object_id:
            obj_category = obj_dict["objectType"]
            break
    assert obj_category
    if obj_category.lower() in ["sinkbasin", "bathtubbasin"]:
        event2 = controller.step(
            action="EmptyLiquidFromObject", objectId=invent_obj_id, forceAction=False
        )
        assert controller.last_event.metadata["lastActionSuccess"], (
            controller.last_event.metadata["errorMessage"]
        )
        return controller.last_event.metadata["lastActionSuccess"], process_error_msg(
            controller.last_event.metadata["errorMessage"]
        )

    event1 = controller.step(
        action="FillObjectWithLiquid",
        objectId=object_id,
        fillLiquid=liquid,
        forceAction=False,
    )
    if not controller.last_event.metadata["lastActionSuccess"]:
        return controller.last_event.metadata["lastActionSuccess"], process_error_msg(
            controller.last_event.metadata["errorMessage"]
        )
    event2 = controller.step(
        action="EmptyLiquidFromObject", objectId=invent_obj_id, forceAction=False
    )
    assert controller.last_event.metadata["lastActionSuccess"], (
        controller.last_event.metadata["errorMessage"]
    )
    return controller.last_event.metadata["lastActionSuccess"], process_error_msg(
        controller.last_event.metadata["errorMessage"]
    )


def open_recep(
    controller: Controller, object_ref: Union[str, List[float]], openness: float = 1.0
):
    object_id = resolve_object(object_ref, controller)
    if object_id is None:
        return False, f"Invalid action argument"
    controller.step(
        action="OpenObject",
        objectId=object_id,
        openness=openness,
        forceAction=False,
    )
    return controller.last_event.metadata["lastActionSuccess"], process_error_msg(
        controller.last_event.metadata["errorMessage"]
    )


def close_recep(controller: Controller, object_ref: Union[str, List[float]]):
    object_id = resolve_object(object_ref, controller)
    if object_id is None:
        return False, f"Invalid action argument"
    controller.step(
        action="CloseObject",
        objectId=object_id,
        forceAction=False,
    )
    return controller.last_event.metadata["lastActionSuccess"], process_error_msg(
        controller.last_event.metadata["errorMessage"]
    )


def slice_obj(controller: Controller, object_ref: Union[str, List[float]]):
    object_id = resolve_object(object_ref, controller)
    if object_id is None:
        return False, f"Invalid action argument"

    invent_obj_id = None
    for obj_dict in controller.last_event.metadata["objects"]:
        if obj_dict["isPickedUp"]:
            invent_obj_id = obj_dict["objectId"]
            if obj_dict["objectType"].lower() not in ["knife", "butterknife"]:
                return False, "The agent is not holding a valid slicing tool."
    if invent_obj_id is None:
        return False, "The agent is not holding a valid slicing tool."

    controller.step(
        action="SliceObject",
        objectId=object_id,
        forceAction=False,
    )
    success = controller.last_event.metadata["lastActionSuccess"]
    if not success:
        target = next(
            (
                obj
                for obj in controller.last_event.metadata["objects"]
                if obj["objectId"] == object_id
            ),
            None,
        )
        if target is not None and not target.get("sliceable"):
            return False, "The target object cannot be sliced."
    return success, process_error_msg(controller.last_event.metadata["errorMessage"])


def turnOn_obj(controller: Controller, object_ref: Union[str, List[float]]):
    object_id = resolve_object(object_ref, controller)
    if object_id is None:
        return False, f"Invalid action argument"
    controller.step(
        action="ToggleObjectOn",
        objectId=object_id,
        forceAction=False,
    )

    success = controller.last_event.metadata["lastActionSuccess"]
    if not success:
        msg = process_error_msg(controller.last_event.metadata["errorMessage"])
        return success, msg

    if success:
        obj_category = None
        for obj_dict in controller.last_event.metadata["objects"]:
            if obj_dict["objectId"] == object_id:
                obj_category = obj_dict["objectType"]
                faucet_pos = obj_dict["position"]
                break
        assert obj_category
        if obj_category.lower() in [
            "faucet",
        ]:
            pass
            sinks = []
            for obj_dict in controller.last_event.metadata["objects"]:
                if obj_dict["objectType"].lower() in ["sinkbasin", "bathtubbasin"]:
                    sinks.append([obj_dict["objectId"], obj_dict["position"]])

            closest_sink = None
            if faucet_pos and sinks:

                def euclidean(p1, p2):
                    return (
                        (p1["x"] - p2["x"]) ** 2
                        + (p1["y"] - p2["y"]) ** 2
                        + (p1["z"] - p2["z"]) ** 2
                    ) ** 0.5

                closest_sink = min(sinks, key=lambda s: euclidean(faucet_pos, s[1]))

                sink_id = closest_sink[0]
                for obj_dict in controller.last_event.metadata["objects"]:
                    if (
                        obj_dict.get("parentReceptacles")
                        and sink_id in obj_dict["parentReceptacles"]
                    ):
                        controller.step(
                            action="CleanObject",
                            objectId=obj_dict["objectId"],
                            forceAction=True,
                        )
                        controller.step(
                            action="FillObjectWithLiquid",
                            objectId=obj_dict["objectId"],
                            fillLiquid="water",
                            forceAction=True,
                        )

    return True, ""


def turnOff_obj(controller: Controller, object_ref: Union[str, List[float]]):
    object_id = resolve_object(object_ref, controller)
    if object_id is None:
        return False, f"Invalid action argument"
    controller.step(
        action="ToggleObjectOff",
        objectId=object_id,
        forceAction=False,
    )
    return controller.last_event.metadata["lastActionSuccess"], process_error_msg(
        controller.last_event.metadata["errorMessage"]
    )


def useUp_obj(controller: Controller, object_ref: Union[str, List[float]]):
    object_id = resolve_object(object_ref, controller)
    if object_id is None:
        return False, f"Invalid action argument"
    controller.step(
        action="UseUpObject",
        objectId=object_id,
        forceAction=False,
    )
    return controller.last_event.metadata["lastActionSuccess"], process_error_msg(
        controller.last_event.metadata["errorMessage"]
    )


# high-level


def find_objCls(
    controller: Controller,
    object_cls: str,
    by_dist: bool = True,
    possible_locations: Optional[List[dict[str, float]]] = None,
):
    closed_receps = []
    for obj_dict in controller.last_event.metadata["objects"]:
        if (
            obj_dict["receptacle"]
            and obj_dict["openable"]
            and obj_dict["isOpen"] == False
        ):
            closed_receps.append(obj_dict["objectId"])

    candidates = []
    has_matching_object = False
    for obj_dict in controller.last_event.metadata["objects"]:
        if match_object_cls(object_cls, obj_dict["objectId"]):
            has_matching_object = True
            if obj_dict["parentReceptacles"] is not None and any(
                x in closed_receps for x in obj_dict["parentReceptacles"]
            ):
                continue
            candidates.append([obj_dict["objectId"], obj_dict["position"]])

    if len(candidates) == 0:
        if has_matching_object:
            return False, "The target object is inside a closed receptacle."
        return False, f"No object of class {object_cls} was found."

    if by_dist:
        object_id = min(
            candidates,
            key=lambda x: calculate_distance_2D(
                x[1], controller.last_event.metadata["agent"]["position"]
            ),
        )[0]
    else:
        object_id = np.random.choice(candidates)[0]

    return goTo_obj(controller, object_id, possible_locations)


def empty_recep(controller: Controller, object_ref: Union[str, List[float]]):
    object_id = resolve_object(object_ref, controller)
    if object_id is None:
        return False, f"Invalid action argument"
    controller.step(
        action="EmptyLiquidFromObject",
        objectId=object_id,
        forceAction=False,
    )
    return controller.last_event.metadata["lastActionSuccess"], process_error_msg(
        controller.last_event.metadata["errorMessage"]
    )


def heatWith_tool(controller: Controller, tool_ref: Union[str, List[float]]):
    invent_obj_id = None
    invent_obj_category = None
    for obj_dict in controller.last_event.metadata["objects"]:
        if obj_dict["isPickedUp"]:
            invent_obj_id = obj_dict["objectId"]
            invent_obj_category = obj_dict["objectType"]
    if invent_obj_id is None:
        return False, "The agent is not holding an object."

    tool_id = resolve_object(tool_ref, controller)
    if tool_id is None:
        return False, f"Invalid action argument"
    if tool_id not in controller.last_event.instance_detections2D:
        return False, "The selected tool is not visible."
    tool_category = None
    for obj_dict in controller.last_event.metadata["objects"]:
        if obj_dict["objectId"] == tool_id:
            tool_category = obj_dict["objectType"]
            break
    assert tool_category

    if (invent_obj_category, tool_category) in OBJ_HEATING:
        if tool_category == "Microwave":
            controller.step(
                action="OpenObject",
                objectId=tool_id,
                forceAction=True,
            )
            controller.step(
                action="PutObject",
                objectId=tool_id,
                forceAction=True,
            )
            ret1 = controller.last_event.metadata["lastActionSuccess"]
            if not ret1:
                msg1 = process_error_msg(controller.last_event.metadata["errorMessage"])
            if ret1:
                controller.step(
                    action="CloseObject",
                    objectId=tool_id,
                    forceAction=True,
                )
                controller.step(
                    action="ToggleObjectOn",
                    objectId=tool_id,
                    forceAction=True,
                )
                controller.step(
                    action="ToggleObjectOff",
                    objectId=tool_id,
                    forceAction=True,
                )
                controller.step(
                    action="OpenObject",
                    objectId=tool_id,
                    forceAction=True,
                )
                controller.step(
                    action="PickupObject",
                    objectId=invent_obj_id,
                    forceAction=True,
                )
                ret2 = controller.last_event.metadata["lastActionSuccess"]
                if not ret2:
                    msg2 = process_error_msg(
                        controller.last_event.metadata["errorMessage"]
                    )
                    return ret2, msg2
                return True, ""
            else:
                return ret1, msg1
        elif tool_category == "CoffeeMachine":
            controller.step(
                action="PutObject",
                objectId=tool_id,
                forceAction=True,
            )
            ret1, msg1 = (
                controller.last_event.metadata["lastActionSuccess"],
                controller.last_event.metadata["errorMessage"],
            )
            if ret1:
                controller.step(
                    action="ToggleObjectOn",
                    objectId=tool_id,
                    forceAction=True,
                )
                controller.step(
                    action="ToggleObjectOff",
                    objectId=tool_id,
                    forceAction=True,
                )
                controller.step(
                    action="PickupObject",
                    objectId=invent_obj_id,
                    forceAction=True,
                )
                ret2 = controller.last_event.metadata["lastActionSuccess"]
                if not ret2:
                    msg2 = process_error_msg(
                        controller.last_event.metadata["errorMessage"]
                    )
                    return ret2, msg2
                return True, ""
            else:
                return ret1, msg1
        elif tool_category == "StoveBurner":
            controller.step(
                action="PutObject",
                objectId=tool_id,
                forceAction=True,
            )
            ret1, msg1 = (
                controller.last_event.metadata["lastActionSuccess"],
                controller.last_event.metadata["errorMessage"],
            )
            if ret1:
                knobs = []
                for obj_dict in controller.last_event.metadata["objects"]:
                    if obj_dict["objectType"].lower() == "stoveknob":
                        knobs.append(obj_dict["objectId"])
                for knob in knobs:
                    controller.step(
                        action="ToggleObjectOn",
                        objectId=knob,
                        forceAction=True,
                    )
                controller.step(
                    action="PickupObject",
                    objectId=invent_obj_id,
                    forceAction=True,
                )
                ret2 = controller.last_event.metadata["lastActionSuccess"]
                if not ret2:
                    msg2 = process_error_msg(
                        controller.last_event.metadata["errorMessage"]
                    )
                    return ret2, msg2
                return True, ""
            else:
                return ret1, msg1
        else:
            assert False, [invent_obj_category, tool_category]
    else:
        return False, "The held object cannot be heated with the selected tool."


def coolWith_tool(controller: Controller, tool_ref: Union[str, List[float]]):
    invent_obj_id = None
    invent_obj_category = None
    for obj_dict in controller.last_event.metadata["objects"]:
        if obj_dict["isPickedUp"]:
            invent_obj_id = obj_dict["objectId"]
            invent_obj_category = obj_dict["objectType"]
    if invent_obj_id is None:
        return False, "The agent is not holding an object."

    tool_id = resolve_object(tool_ref, controller)
    if tool_id is None:
        return False, f"Invalid action argument"
    if tool_id not in controller.last_event.instance_detections2D:
        return False, "The selected tool is not visible."
    tool_category = None
    for obj_dict in controller.last_event.metadata["objects"]:
        if obj_dict["objectId"] == tool_id:
            tool_category = obj_dict["objectType"]
            break
    assert tool_category

    if (invent_obj_category, tool_category) in OBJ_COOLING:
        if tool_category == "Fridge":
            controller.step(
                action="OpenObject",
                objectId=tool_id,
                forceAction=True,
            )
            controller.step(
                action="PutObject",
                objectId=tool_id,
                forceAction=True,
            )
            ret1, msg1 = (
                controller.last_event.metadata["lastActionSuccess"],
                controller.last_event.metadata["errorMessage"],
            )
            if ret1:
                controller.step(
                    action="PickupObject",
                    objectId=invent_obj_id,
                    forceAction=True,
                )
                ret2 = controller.last_event.metadata["lastActionSuccess"]
                if not ret2:
                    msg2 = process_error_msg(
                        controller.last_event.metadata["errorMessage"]
                    )
                    return ret2, msg2
                return True, ""
            else:
                return ret1, msg1
        else:
            assert False, [invent_obj_category, tool_category]
    else:
        return False, "The held object cannot be cooled with the selected tool."


def washWith_tool(controller: Controller, tool_ref: Union[str, List[float]]):
    invent_obj_id = None
    invent_obj_category = None
    for obj_dict in controller.last_event.metadata["objects"]:
        if obj_dict["isPickedUp"]:
            invent_obj_id = obj_dict["objectId"]
            invent_obj_category = obj_dict["objectType"]
    if invent_obj_id is None:
        return False, "The agent is not holding an object."

    tool_id = resolve_object(tool_ref, controller)
    if tool_id is None:
        return False, f"Invalid action argument"
    tool_category = None
    for obj_dict in controller.last_event.metadata["objects"]:
        if obj_dict["objectId"] == tool_id:
            tool_category = obj_dict["objectType"]
            break
    assert tool_category

    if tool_id not in controller.last_event.instance_detections2D:
        return False, "The selected tool is not visible."

    if (invent_obj_category, tool_category) in OBJ_WASHING or (
        (
            (invent_obj_category, "Faucet") in OBJ_WASHING
            or (invent_obj_category, "ShowerHead") in OBJ_WASHING
        )
        and tool_category in ["SinkBasin", "BathtubBasin", "Sink", "Bathtub"]
    ):
        controller.step(
            action="CleanObject",
            objectId=invent_obj_id,
            forceAction=True,
        )
        clean_success, clean_feedback = (
            controller.last_event.metadata["lastActionSuccess"],
            process_error_msg(controller.last_event.metadata["errorMessage"]),
        )
        controller.step(
            action="FillObjectWithLiquid",
            objectId=invent_obj_id,
            fillLiquid="water",
            forceAction=True,
        )
        return clean_success, clean_feedback
    else:
        return False, "The held object cannot be washed with the selected tool."


def cookWith_tool(controller: Controller, tool_ref: Union[str, List[float]]):
    invent_obj_id = None
    invent_obj_category = None
    for obj_dict in controller.last_event.metadata["objects"]:
        if obj_dict["isPickedUp"]:
            invent_obj_id = obj_dict["objectId"]
            invent_obj_category = obj_dict["objectType"]
    if invent_obj_id is None:
        return False, "The agent is not holding an object."

    tool_id = resolve_object(tool_ref, controller)
    if tool_id is None:
        return False, f"Invalid action argument"

    if tool_id not in controller.last_event.instance_detections2D:
        return False, "The selected tool is not visible."

    tool_category = None
    for obj_dict in controller.last_event.metadata["objects"]:
        if obj_dict["objectId"] == tool_id:
            tool_category = obj_dict["objectType"]
            break
    assert tool_category

    if invent_obj_category in ["Bread", "BreadSliced"] and "Sliced" in invent_obj_id:
        if tool_category in ["Toaster", "StoveBurner"]:
            controller.step(
                action="CookObject",
                objectId=invent_obj_id,
                forceAction=True,
            )
            return controller.last_event.metadata[
                "lastActionSuccess"
            ], process_error_msg(controller.last_event.metadata["errorMessage"])
    elif invent_obj_category in ["Potato", "PotatoSliced"]:
        if tool_category in ["Microwave", "StoveBurner"]:
            controller.step(
                action="CookObject",
                objectId=invent_obj_id,
                forceAction=True,
            )
            return controller.last_event.metadata[
                "lastActionSuccess"
            ], process_error_msg(controller.last_event.metadata["errorMessage"])
    elif invent_obj_category in ["EggCracked"]:
        if tool_category in ["Microwave", "StoveBurner"]:
            controller.step(
                action="CookObject",
                objectId=invent_obj_id,
                forceAction=True,
            )
            return controller.last_event.metadata[
                "lastActionSuccess"
            ], process_error_msg(controller.last_event.metadata["errorMessage"])

    return False, "The held object cannot be cooked with the selected tool."


def fillWith_tool(controller: Controller, tool_ref: Union[str, List[float]]):
    invent_obj_id = None
    invent_obj_category = None
    for obj_dict in controller.last_event.metadata["objects"]:
        if obj_dict["isPickedUp"]:
            invent_obj_id = obj_dict["objectId"]
            invent_obj_category = obj_dict["objectType"]
    if invent_obj_id is None:
        return False, "The agent is not holding an object."

    tool_id = resolve_object(tool_ref, controller)
    if tool_id is None:
        return False, f"Invalid action argument"

    if tool_id not in controller.last_event.instance_detections2D:
        return False, "The selected tool is not visible."

    tool_category = None
    for obj_dict in controller.last_event.metadata["objects"]:
        if obj_dict["objectId"] == tool_id:
            tool_category = obj_dict["objectType"]
            break
    assert tool_category

    if (invent_obj_category, tool_category) in OBJ_FILLING_WATER:
        controller.step(
            action="FillObjectWithLiquid",
            objectId=invent_obj_id,
            fillLiquid="water",
            forceAction=True,
        )
        return controller.last_event.metadata["lastActionSuccess"], process_error_msg(
            controller.last_event.metadata["errorMessage"]
        )
    elif (invent_obj_category, tool_category) in OBJ_FILLING_COFFEE:
        controller.step(
            action="FillObjectWithLiquid",
            objectId=invent_obj_id,
            fillLiquid="coffee",
            forceAction=True,
        )
        return controller.last_event.metadata["lastActionSuccess"], process_error_msg(
            controller.last_event.metadata["errorMessage"]
        )
    elif (invent_obj_category, tool_category) in OBJ_FILLING_WINE:
        controller.step(
            action="FillObjectWithLiquid",
            objectId=invent_obj_id,
            fillLiquid="wine",
            forceAction=True,
        )
        return controller.last_event.metadata["lastActionSuccess"], process_error_msg(
            controller.last_event.metadata["errorMessage"]
        )

    return False, "The held object cannot be filled with the selected tool."
