import json
import os
from enum import Enum
from typing import Dict, Any
import numpy as np
from scene_graph.unified_scene_graph import UnifiedSceneGraph
from scene_graph.graph_schema.entity import Entity, EntityType
from scene_graph.graph_schema.attribute import (
    ObjectAttributes,
    PropertyType,
    State,
    StateType,
)
from scene_graph.graph_schema.relation import Relations, RelationType


class USGJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, (EntityType, PropertyType, StateType, RelationType)):
            return obj.value
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


def serialize_entity(entity: Entity) -> Dict[str, Any]:
    return {
        "ID": entity.ID,
        "type": entity.type.value
        if isinstance(entity.type, EntityType)
        else str(entity.type),
        "category": entity.category,
        "attributes": serialize_attributes(entity.attributes),
        "additional_info": serialize_additional_info(entity.additional_info),
    }


def serialize_attributes(attributes: ObjectAttributes) -> Dict[str, Any]:
    properties = [
        prop.value if isinstance(prop, PropertyType) else str(prop)
        for prop in attributes.properties
    ]

    states = []
    for state_type, state in attributes.states.items():
        state_dict = {
            "type": state_type.value
            if isinstance(state_type, StateType)
            else str(state_type),
            "params": state.params,
        }
        states.append(state_dict)

    return {"properties": properties, "states": states}


def serialize_relations(relations: Relations) -> Dict[str, Any]:
    rel_list = [
        rel.value if isinstance(rel, RelationType) else str(rel)
        for rel in relations.relations
    ]
    return {
        "relations": rel_list,
        "details": serialize_additional_info(relations.details),
    }


def serialize_additional_info(info: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in info.items():
        if not isinstance(key, str):
            key = str(key)

        if isinstance(value, Entity):
            result[key] = serialize_entity(value)
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            result[key] = value.item()
        elif isinstance(value, dict):
            serialized_dict = {}
            for k, v in value.items():
                dict_key = str(k) if not isinstance(k, str) else k
                serialized_dict[dict_key] = serialize_value(v)
            result[key] = serialized_dict
        elif isinstance(value, (list, tuple, set)):
            result[key] = [serialize_value(v) for v in value]
        else:
            result[key] = serialize_value(value)
    return result


def serialize_state(state: State) -> Dict[str, Any]:
    return {
        "type": state.type.value
        if isinstance(state.type, StateType)
        else str(state.type),
        "params": serialize_value(state.params) if state.params else None,
    }


def serialize_value(value: Any) -> Any:
    if isinstance(value, Entity):
        return serialize_entity(value)
    elif isinstance(value, State):
        return serialize_state(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (np.integer, np.floating)):
        return value.item()
    elif isinstance(value, dict):
        result = {}
        for k, v in value.items():
            if isinstance(k, (StateType, EntityType, PropertyType, RelationType, Enum)):
                dict_key = k.value if hasattr(k, "value") else str(k)
            elif not isinstance(k, str):
                dict_key = str(k)
            else:
                dict_key = k
            result[dict_key] = serialize_value(v)
        return result
    elif isinstance(value, (list, tuple, set)):
        return [serialize_value(v) for v in value]
    elif isinstance(value, (EntityType, PropertyType, StateType, RelationType, Enum)):
        return value.value if hasattr(value, "value") else str(value)
    else:
        return value


def serialize_graph(graph) -> Dict[str, Any]:

    node_id_to_idx = {}
    nodes = []
    edges = []

    for node in graph.graph.keys():
        node_id_to_idx[node.ID] = len(nodes)
        nodes.append(serialize_entity(node))

    for src_node, neighbors in graph.graph.items():
        src_idx = node_id_to_idx[src_node.ID]
        for dst_node, relations in neighbors.items():
            dst_idx = node_id_to_idx[dst_node.ID]
            edges.append(
                {
                    "source": src_idx,
                    "target": dst_idx,
                    "source_id": src_node.ID,
                    "target_id": dst_node.ID,
                    "relations": serialize_relations(relations),
                }
            )

    return {"nodes": nodes, "edges": edges}


def serialize_usg(usg: UnifiedSceneGraph) -> Dict[str, Any]:

    cache_dict = None
    if usg.cache:
        cache_dict = {}
        for key, value in usg.cache.items():
            cache_key = str(key) if not isinstance(key, str) else key

            if cache_key == "og_obj_to_entity_id":
                serialized_dict = {}
                for obj_key, entity_id_value in value.items():
                    obj_key_str = (
                        str(obj_key) if not isinstance(obj_key, str) else obj_key
                    )

                    if hasattr(obj_key, "name"):
                        obj_key_str = f"{type(obj_key).__name__}_{obj_key.name}"
                    elif hasattr(obj_key, "__class__"):
                        obj_key_str = f"{obj_key.__class__.__name__}_{id(obj_key)}"
                    serialized_dict[obj_key_str] = serialize_value(entity_id_value)
                cache_dict[cache_key] = serialized_dict
            elif cache_key == "room_name_to_entity_id":
                serialized_dict = {}
                for room_key, entity_id_value in value.items():
                    room_key_str = (
                        str(room_key) if not isinstance(room_key, str) else room_key
                    )
                    serialized_dict[room_key_str] = serialize_value(entity_id_value)
                cache_dict[cache_key] = serialized_dict
            elif cache_key in ["prev_edges", "prev_nodes"]:
                if isinstance(value, set):
                    cache_dict[cache_key] = [serialize_value(v) for v in value]
                else:
                    cache_dict[cache_key] = serialize_value(value)
            else:
                cache_dict[cache_key] = serialize_value(value)

    return {
        "env_metadata": usg.env_metadata,
        "sim_name": usg.sim_name,
        "graph": serialize_graph(usg.graph),
        "cache": cache_dict,
        "full_obs": usg.full_obs,
        "debug": usg.debug,
    }


def save_usg_to_json(usg: UnifiedSceneGraph, filepath: str, indent: int = 2) -> None:

    os.makedirs(
        os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True
    )

    usg_dict = serialize_usg(usg)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(usg_dict, f, indent=indent, ensure_ascii=False, cls=USGJSONEncoder)


def load_usg_from_json(filepath: str) -> Dict[str, Any]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
