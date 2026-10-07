from typing import Optional
import re


from THOR.thor_env import THOR_Environment
from scene_graph.graph_schema import (
    Graph,
    Entity,
    EntityType,
    RelationType,
    ObjectAttributes,
    PropertyType,
    StateType,
    flip_edge,
)
from scene_graph.unified_scene_graph import UnifiedSceneGraph
from task.commonsense_knowledge.thor import (
    OBJ_WITH_TEMPERATURE,
    RECEP_OBJS_IN,
    OBJ_RECEP,
)
from THOR.thor_actions import position_in_room


def _normalize_category_for_id(category: str) -> str:
    """Convert category to a stable id-friendly token."""
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", category).strip("_")
    return normalized if normalized else "obj"


def _build_thor_id_cache(prev_cache: Optional[dict]) -> dict:
    if not isinstance(prev_cache, dict):
        return {
            "object_id_to_node_id": {},
            "node_id_to_object_id": {},
            "next_idx_per_category": {},
        }

    object_id_to_node_id = prev_cache.get("object_id_to_node_id", {})
    node_id_to_object_id = prev_cache.get("node_id_to_object_id", {})
    next_idx_per_category = prev_cache.get("next_idx_per_category", {})

    if not isinstance(object_id_to_node_id, dict):
        object_id_to_node_id = {}
    if not isinstance(node_id_to_object_id, dict):
        node_id_to_object_id = {}
    if not isinstance(next_idx_per_category, dict):
        next_idx_per_category = {}

    normalized_next_idx_per_category = {}
    for category_token, idx in next_idx_per_category.items():
        if isinstance(category_token, str) and isinstance(idx, int) and idx >= 0:
            normalized_next_idx_per_category[category_token] = idx

    return {
        "object_id_to_node_id": dict(object_id_to_node_id),
        "node_id_to_object_id": dict(node_id_to_object_id),
        "next_idx_per_category": normalized_next_idx_per_category,
    }


def thor_event_to_SG(
    env: THOR_Environment, id_cache: Optional[dict] = None
) -> tuple[UnifiedSceneGraph, dict]:
    graph = Graph()
    cache = _build_thor_id_cache(id_cache)
    object_id_to_node_id = cache["object_id_to_node_id"]
    node_id_to_object_id = cache["node_id_to_object_id"]
    next_idx_per_category = cache["next_idx_per_category"]

    def get_node_id(object_id: str, category: str) -> str:
        if object_id in object_id_to_node_id:
            return object_id_to_node_id[object_id]
        category_token = _normalize_category_for_id(category)
        category_idx = next_idx_per_category.get(category_token, 0)
        node_id = f"{category_token}_{category_idx}"
        next_idx_per_category[category_token] = category_idx + 1
        object_id_to_node_id[object_id] = node_id
        node_id_to_object_id[node_id] = object_id
        return node_id

    agent_node = Entity(
        ID="robot_agent",
        type=EntityType.ROBOT,
        category="agent",
        attributes=ObjectAttributes(),
        additional_info={
            "position": [
                env.controller.last_event.metadata["agent"]["position"]["x"],
                env.controller.last_event.metadata["agent"]["position"]["y"],
                env.controller.last_event.metadata["agent"]["position"]["z"],
            ]
        },
    )
    graph.add_node(agent_node)

    objects_info = env.controller.last_event.metadata["objects"]
    for object_info in objects_info:
        if object_info["objectType"] == "Floor":
            if object_info["objectId"] == "Floor" or object_info["objectId"].startswith(
                "Floor|"
            ):
                object_node = Entity(
                    ID="house",
                    type=EntityType.HOUSE,
                    category="house",
                    attributes=ObjectAttributes(),
                    additional_info={
                        "position": [
                            object_info["position"]["x"],
                            object_info["position"]["y"],
                            object_info["position"]["z"],
                        ]
                    },
                )
            else:
                assert object_info["objectId"].startswith("room|"), object_info
                room_node_id = get_node_id(
                    object_info["objectId"],
                    env.room_id_to_category[object_info["objectId"]],
                )
                object_node = Entity(
                    ID=room_node_id,
                    type=EntityType.ROOM,
                    category=env.room_id_to_category[object_info["objectId"]],
                    attributes=ObjectAttributes(),
                    additional_info={
                        "position": [
                            object_info["position"]["x"],
                            object_info["position"]["y"],
                            object_info["position"]["z"],
                        ]
                    },
                )
            graph.add_node(object_node)
            continue
        if object_info["objectType"] == "Wall":
            assert object_info["receptacle"] == False
            continue

        assert object_info["objectType"] in set(x[0] for x in OBJ_RECEP) | {
            "Doorway",
            "Doorframe",
            "Cart",
            "WashingMachine",
            "ClothesDryer",
        } or object_info["objectType"] + "*" in set(x[0] for x in OBJ_RECEP), (
            object_info
        )

        object_node_id = get_node_id(object_info["objectId"], object_info["objectType"])
        object_node = Entity(
            ID=object_node_id,
            type=EntityType.OBJECT,
            category=object_info["objectType"],
            attributes=ObjectAttributes(),
            additional_info={
                "position": [
                    object_info["position"]["x"],
                    object_info["position"]["y"],
                    object_info["position"]["z"],
                ]
            },
        )
        graph.add_node(object_node)
        if object_info["pickupable"]:
            object_node.set_property(PropertyType.GRASPABLE)
            if object_info["isPickedUp"]:
                graph.add_edge(
                    agent_node,
                    object_node,
                    RelationType.GRASPING,
                    flip_edge(RelationType.GRASPING),
                )
        if object_info["openable"]:
            object_node.set_property(PropertyType.OPENABLE)
            if object_info["isOpen"]:
                object_node.add_state(StateType.OPEN)
            else:
                object_node.add_state(StateType.CLOSED)
        if object_info["toggleable"]:
            object_node.set_property(PropertyType.HAS_POWER)
            if object_info["isToggled"]:
                object_node.add_state(StateType.POWER_ON)
            else:
                object_node.add_state(StateType.POWER_OFF)
        if object_info["canFillWithLiquid"]:
            object_node.set_property(PropertyType.CAN_BE_FILLED_WITH_LIQUID)
            if (
                object_info["isFilledWithLiquid"]
                and object_info["fillLiquid"] is not None
            ):
                assert object_info["fillLiquid"] in ["coffee", "water", "wine"], (
                    object_info
                )
                object_node.add_state(
                    StateType.FILLED_WITH_LIQUID,
                    {"liquid_type": object_info["fillLiquid"]},
                )
            else:
                object_node.add_state(StateType.NO_LIQUID)
        if object_info["cookable"]:
            object_node.set_property(PropertyType.CAN_BE_COOKED)
            if object_info["isCooked"]:
                object_node.add_state(StateType.COOKED)
            else:
                object_node.add_state(StateType.NOT_COOKED)
        if object_info["breakable"]:
            object_node.set_property(PropertyType.CAN_BE_BROKEN)
            if object_info["isBroken"]:
                object_node.add_state(StateType.BROKEN)
            else:
                object_node.add_state(StateType.NOT_BROKEN)
        if object_info["dirtyable"]:
            object_node.set_property(PropertyType.CAN_BE_CLEANED)
            if object_info["isDirty"]:
                object_node.add_state(StateType.DIRTY)
            else:
                object_node.add_state(StateType.CLEAN)
        if object_info["canBeUsedUp"]:
            object_node.set_property(PropertyType.CAN_BE_USED_UP)
            if object_info["isUsedUp"]:
                object_node.add_state(StateType.USED_UP)
            else:
                object_node.add_state(StateType.NOT_USED_UP)

        if object_info["objectType"] in OBJ_WITH_TEMPERATURE:
            object_node.set_property(PropertyType.CAN_BE_HEATED)
            assert object_info["temperature"] in ["Cold", "RoomTemp", "Hot"], (
                object_info["temperature"]
            )
            if object_info["temperature"] == "Hot":
                object_node.add_state(StateType.HOT)
            elif object_info["temperature"] == "Cold":
                object_node.add_state(StateType.COOL)
            else:
                object_node.add_state(StateType.NORMAL_TEMPERATURE)

        if object_info["objectType"] == "Doorway":
            object_node.update_info(
                {"connected_rooms": env.door_to_rooms[object_info["objectId"]]}
            )

    for object_info in objects_info:
        # object_info["parentReceptacles"] A list of objectId strings of all receptacles that contain this object.
        # object_info["receptacleObjectIds"] If the object is a receptacle, this is an array of objectIds that the receptacle contains.
        if (
            object_info["objectType"] == "Floor"
            and object_info["receptacleObjectIds"] is not None
        ):
            if object_info["objectId"] == "Floor" or object_info["objectId"].startswith(
                "Floor|"
            ):
                continue
            receptacle_id = object_id_to_node_id[object_info["objectId"]]
            assert receptacle_id, object_info["objectId"]
            for oid in object_info["receptacleObjectIds"]:
                source_id = object_id_to_node_id[oid]
                assert source_id, oid
                graph.add_edge(
                    source_id,
                    receptacle_id,
                    RelationType.ON_ROOM_FLOOR,
                    flip_edge(RelationType.ON_ROOM_FLOOR),
                )

        elif object_info["receptacle"]:
            receptacle_id = object_id_to_node_id[object_info["objectId"]]
            assert receptacle_id, object_info["objectId"]
            for oid in object_info["receptacleObjectIds"]:
                if oid not in object_id_to_node_id:
                    continue
                source_id = object_id_to_node_id[oid]
                assert source_id, oid
                if object_info["objectType"] in RECEP_OBJS_IN:
                    graph.add_edge(
                        source_id,
                        receptacle_id,
                        RelationType.INSIDE_RECEPTACLE,
                        flip_edge(RelationType.INSIDE_RECEPTACLE),
                    )
                else:
                    graph.add_edge(
                        source_id,
                        receptacle_id,
                        RelationType.ON_RECEPTACLE,
                        flip_edge(RelationType.ON_RECEPTACLE),
                    )

    agent_position = env.controller.last_event.metadata["agent"]["position"]
    for room_id, floor_polygon in env.room_id_to_floor_polygon.items():
        if position_in_room(agent_position, floor_polygon):
            graph.add_edge(
                agent_node,
                object_id_to_node_id[room_id],
                RelationType.INSIDE_ROOM,
                flip_edge(RelationType.INSIDE_ROOM),
            )
            break

    return graph, {
        "object_id_to_node_id": object_id_to_node_id,
        "node_id_to_object_id": node_id_to_object_id,
        "next_idx_per_category": next_idx_per_category,
    }
