import sys
import os
from typing import Optional
import numpy as np
import torch as th
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
from scene_graph.graph_schema.attribute import STATE_REQUIREMENTS
from scene_graph.unified_scene_graph import UnifiedSceneGraph


_current_dir = os.path.dirname(os.path.abspath(__file__))
_omnigibson_path = os.path.join(_current_dir, "BEHAVIOR-1K", "OmniGibson")
if os.path.exists(_omnigibson_path) and _omnigibson_path not in sys.path:
    sys.path.insert(0, _omnigibson_path)


_scene_graph_path = os.path.join(os.path.dirname(_current_dir), "scene_graph")
if os.path.exists(_scene_graph_path) and _scene_graph_path not in sys.path:
    sys.path.insert(0, os.path.dirname(_current_dir))

from omnigibson import object_states
from omnigibson.object_states.factory import get_state_name
from omnigibson.object_states.object_state_base import (
    AbsoluteObjectState,
    BooleanStateMixin,
    RelativeObjectState,
)
from omnigibson.object_states.open_state import Open
from omnigibson.object_states.contains import Contains
from omnigibson.object_states.filled import Filled
from omnigibson.object_states.cooked import Cooked
from omnigibson.object_states.burnt import Burnt
from omnigibson.object_states.frozen import Frozen
from omnigibson.object_states.saturated import Saturated
from omnigibson.object_states.toggle import ToggledOn
from omnigibson.object_states.heated import Heated
from omnigibson.object_states.covered import Covered
from omnigibson.robots import BaseRobot
from omnigibson.utils.constants import PrimType


class UnifiedSceneGraphBuilder:
    def __init__(self, full_obs: bool = True):
        self.full_obs = full_obs
        self.usg: Optional[UnifiedSceneGraph] = None
        self.robot = None

    def start(
        self, scene, robot=None, env_metadata=None, grasped_obj=None
    ) -> UnifiedSceneGraph:
        graph = Graph()
        cache = {
            "object_position": {},
            "og_obj_to_entity_id": {},
            "room_name_to_entity_id": {},
        }

        if robot is None:
            if len(scene.robots) > 0:
                robot = scene.robots[0]
        self.robot = robot

        _add_house_node(graph)

        room_name_to_entity_id = _add_room_nodes(graph, scene, cache)

        if robot is not None:
            _add_robot_node(graph, scene, robot, room_name_to_entity_id, cache)

        objs_to_add = _add_object_nodes(
            graph, scene, robot, room_name_to_entity_id, cache, self.full_obs
        )

        _set_object_properties_and_initial_states(graph, scene, objs_to_add, cache)

        _add_object_relations(graph, objs_to_add, robot, cache)

        if robot is not None:
            _update_robot_grasping_relations(graph, robot, grasped_obj, cache)

        if env_metadata is None:
            env_metadata = {"simulator_name": "BEHAVIOR"}

        self.usg = UnifiedSceneGraph(
            env_metadata, graph, cache, full_obs=self.full_obs, debug=False
        )

        return self.usg


def step(usg: UnifiedSceneGraph, scene, robot, grasped_obj=None) -> UnifiedSceneGraph:
    _update_object_additions_and_removals(
        usg.graph, scene, robot, usg.cache, usg.full_obs
    )

    _update_object_relations(usg.graph, scene, robot, usg.cache)

    _update_object_object_relations(usg.graph, scene, robot, usg.cache)

    if robot is not None:
        _update_robot_room_associations(usg.graph, scene, robot, usg.cache)

    _update_object_states(usg.graph, scene, usg.cache)

    if robot is not None:
        _update_robot_grasping_relations(usg.graph, robot, grasped_obj, usg.cache)

    return usg


def _add_house_node(graph: Graph):
    house = Entity(
        ID="house",
        type=EntityType.HOUSE,
        category="house",
        attributes=ObjectAttributes(),
        additional_info={"asset_name": "house_0"},
    )
    graph.add_node(house)


def _add_room_nodes(graph: Graph, scene, cache: dict) -> dict:
    room_name_to_entity_id = {}

    if hasattr(scene, "_seg_map") and hasattr(
        scene._seg_map, "room_ins_id_to_ins_name"
    ):
        room_ins_id_to_ins_name = scene._seg_map.room_ins_id_to_ins_name
        node_idx = 10000

        for room_ins_id, room_name in room_ins_id_to_ins_name.items():
            room_entity_id = room_name
            room = Entity(
                ID=room_entity_id,
                type=EntityType.ROOM,
                category=room_name.split("_")[0] if "_" in room_name else room_name,
                attributes=ObjectAttributes(),
                additional_info={"asset_name": room_name},
            )
            graph.add_node(room)
            graph.add_edge(
                room,
                "house",
                RelationType.INSIDE_HOUSE,
                flip_edge(RelationType.INSIDE_HOUSE),
            )
            room_name_to_entity_id[room_name] = room_entity_id
            node_idx += 1

        unknown_room = Entity(
            ID=f"unknown_room_{node_idx}",
            type=EntityType.ROOM,
            category="unknown_room",
            attributes=ObjectAttributes(),
            additional_info={"asset_name": "unknown_room"},
        )
        graph.add_node(unknown_room)
        graph.add_edge(
            unknown_room,
            "house",
            RelationType.INSIDE_HOUSE,
            flip_edge(RelationType.INSIDE_HOUSE),
        )
        room_name_to_entity_id["unknown_room"] = f"unknown_room_{node_idx}"
    else:
        default_room = Entity(
            ID="room_0",
            type=EntityType.ROOM,
            category="room",
            attributes=ObjectAttributes(),
            additional_info={"asset_name": "room_0"},
        )
        graph.add_node(default_room)
        graph.add_edge(
            default_room,
            "house",
            RelationType.INSIDE_HOUSE,
            flip_edge(RelationType.INSIDE_HOUSE),
        )
        room_name_to_entity_id["room_0"] = "room_0"

    cache["room_name_to_entity_id"] = room_name_to_entity_id
    return room_name_to_entity_id


def _get_room_id_for_position(
    scene, pos, room_name_to_entity_id: dict, obj=None, debug=False
) -> Optional[str]:
    unknown_room_id = room_name_to_entity_id.get("unknown_room")

    if obj and hasattr(obj, "in_rooms") and obj.in_rooms:
        in_rooms = obj.in_rooms if isinstance(obj.in_rooms, list) else [obj.in_rooms]
        for room_name in in_rooms:
            room_entity_id = room_name_to_entity_id.get(room_name)
            if room_entity_id:
                return room_entity_id

        return None

    if hasattr(scene, "_seg_map") and hasattr(
        scene._seg_map, "get_room_instance_by_point"
    ):
        try:
            if isinstance(pos, th.Tensor):
                pos_np = pos.cpu().numpy()
            else:
                pos_np = np.array(pos)
            room_name = scene._seg_map.get_room_instance_by_point(pos_np[:2])

            if room_name is not None and hasattr(
                scene._seg_map, "room_ins_id_to_ins_name"
            ):
                room_entity_id = room_name_to_entity_id.get(room_name)
                if room_entity_id:
                    return room_entity_id
                elif debug and obj:
                    obj_name = getattr(obj, "name", "unknown")
        except Exception as e:
            if debug and obj:
                obj_name = getattr(obj, "name", "unknown")

    if obj and hasattr(obj, "category") and obj.category == "walls":
        return unknown_room_id
    return None


def _add_robot_node(
    graph: Graph, scene, robot, room_name_to_entity_id: dict, cache: dict
):

    robot_entity_id = "robot_agent"
    robot_entity = Entity(
        ID=robot_entity_id,
        type=EntityType.ROBOT,
        category="agent",
        attributes=ObjectAttributes(),
        additional_info={"asset_name": robot.name, "is_articulated": True},
    )
    graph.add_node(robot_entity)
    cache["og_obj_to_entity_id"][robot] = robot_entity_id

    robot_pos, _ = robot.get_position_orientation()
    robot_room_id = _get_room_id_for_position(
        scene, robot_pos, room_name_to_entity_id, obj=robot
    )
    if robot_room_id:
        graph.add_edge(
            robot_entity,
            robot_room_id,
            RelationType.INSIDE_ROOM,
            flip_edge(RelationType.INSIDE_ROOM),
        )

    cache["object_position"][robot_entity_id] = [
        float(robot_pos[0]),
        float(robot_pos[1]),
        float(robot_pos[2]),
    ]
    robot_entity.update_info({"position": cache["object_position"][robot_entity_id]})


def _get_covered_system_names(obj, scene) -> list:
    covered_system_names = []

    if Covered not in obj.states:
        return covered_system_names

    for system in scene.system_registry.objects:
        is_visual = scene.is_visual_particle_system(system_name=system.name)
        is_physical = scene.is_physical_particle_system(system_name=system.name)
        if not (is_visual or is_physical):
            continue

        if (
            hasattr(obj, "prim_type")
            and obj.prim_type == PrimType.CLOTH
            and is_physical
        ):
            continue
        try:
            if obj.states[Covered].get_value(system):
                covered_system_names.append(system.name)
        except (ValueError, AssertionError):
            continue
    return covered_system_names


def _get_filled_system_names(obj, scene) -> list:
    filled_system_names = []

    if Filled not in obj.states:
        return filled_system_names

    for system in scene.system_registry.objects:
        if not scene.is_physical_particle_system(system_name=system.name):
            continue
        try:
            if obj.states[Filled].get_value(system):
                filled_system_names.append(system.name)
        except (ValueError, AssertionError):
            continue
    return filled_system_names


def _get_contains_system_names(obj, scene) -> list:
    contains_system_names = []

    if Contains not in obj.states:
        return contains_system_names

    for system in scene.system_registry.objects:
        is_visual = scene.is_visual_particle_system(system_name=system.name)
        is_physical = scene.is_physical_particle_system(system_name=system.name)
        if not (is_visual or is_physical):
            continue
        try:
            if obj.states[Contains].get_value(system):
                contains_system_names.append(system.name)
        except (ValueError, AssertionError):
            continue
    return contains_system_names


def _get_saturated_system_names(obj, scene) -> list:
    saturated_system_names = []
    if Saturated not in obj.states:
        return saturated_system_names
    for system in scene.system_registry.objects:
        is_visual = scene.is_visual_particle_system(system_name=system.name)
        is_physical = scene.is_physical_particle_system(system_name=system.name)
        if not (is_visual or is_physical):
            continue
        try:
            if obj.states[Saturated].get_value(system):
                saturated_system_names.append(system.name)
        except (ValueError, AssertionError) as e:
            continue
    return saturated_system_names


def _is_object_dirty(obj, scene) -> bool:
    if Covered not in obj.states:
        return False
    dirty_systems = ["stain", "dust"]
    for system in scene.system_registry.objects:
        if system.name not in dirty_systems:
            continue
        is_visual = scene.is_visual_particle_system(system_name=system.name)
        is_physical = scene.is_physical_particle_system(system_name=system.name)
        if not (is_visual or is_physical):
            continue
        if (
            hasattr(obj, "prim_type")
            and obj.prim_type == PrimType.CLOTH
            and is_physical
        ):
            continue
        try:
            if obj.states[Covered].get_value(system):
                return True
        except (ValueError, AssertionError):
            continue
    return False


def _add_object_nodes(
    graph: Graph,
    scene,
    robot,
    room_name_to_entity_id: dict,
    cache: dict,
    full_obs: bool,
) -> set:
    objs_to_add = set(scene.objects)

    if not full_obs and robot is not None:
        if object_states.ObjectsInFOVOfRobot in robot.states:
            objs_in_fov = robot.states[object_states.ObjectsInFOVOfRobot].get_value()
            objs_to_add &= objs_in_fov

    objs_to_add = {obj for obj in objs_to_add if not isinstance(obj, BaseRobot)}

    for obj in objs_to_add:
        category = getattr(obj, "category", "object")
        obj_id = obj.name

        pos, _ = obj.get_position_orientation()
        position = [float(pos[0]), float(pos[1]), float(pos[2])]

        new_category = category

        entity = Entity(
            ID=obj_id,
            type=EntityType.OBJECT,
            category=new_category,
            attributes=ObjectAttributes(),
            additional_info={
                "asset_name": obj.name,
                "position": position,
                "prim_path": getattr(obj, "prim_path", None),
            },
        )
        graph.add_node(entity)
        cache["og_obj_to_entity_id"][obj] = obj_id
        cache["object_position"][obj_id] = position

        obj_room_id = _get_room_id_for_position(
            scene, pos, room_name_to_entity_id, obj=obj
        )
        if obj_room_id:
            graph.add_edge(
                entity,
                obj_room_id,
                RelationType.INSIDE_ROOM,
                flip_edge(RelationType.INSIDE_ROOM),
            )

    return objs_to_add


def _get_obj_properties_from_abilities(obj) -> set:
    properties = set()

    if not hasattr(obj, "_abilities"):
        return properties

    if not hasattr(obj, "states"):
        return properties

    abilities = obj._abilities

    if Open in obj.states:
        properties.add(PropertyType.OPENABLE)

    if ToggledOn in obj.states:
        properties.add(PropertyType.HAS_POWER)

    if Contains in obj.states or Filled in obj.states:
        properties.add(PropertyType.IS_RECEPTACLE)

    if Cooked in obj.states:
        properties.add(PropertyType.CAN_BE_COOKED)

    if Burnt in obj.states:
        properties.add(PropertyType.CAN_BE_BURNT)

    if Frozen in obj.states:
        properties.add(PropertyType.CAN_BE_FROZEN)

    if Saturated in obj.states:
        properties.add(PropertyType.CAN_BE_STATURATED)

    if Filled in obj.states:
        properties.add(PropertyType.CAN_BE_FILLED_WITH_LIQUID)

    if Heated in obj.states:
        properties.add(PropertyType.CAN_BE_HEATED)

    if Covered in obj.states:
        properties.add(PropertyType.CAN_BE_COVERED)

    if "stainable" in abilities:
        properties.add(PropertyType.CAN_BE_CLEANED)

    if "sliceable" in abilities or "diceable" in abilities:
        properties.add(PropertyType.CAN_BE_SLICED)

    return properties


def _og_state_to_state_type(state_name: str, value: bool) -> Optional[StateType]:

    state_mapping = {
        "Open": StateType.OPEN if value else StateType.CLOSED,
        "ToggledOn": StateType.POWER_ON if value else StateType.POWER_OFF,
        "Cooked": StateType.COOKED if value else StateType.NOT_COOKED,
        "Burnt": StateType.BURNT if value else StateType.NOT_BURNT,
        "Frozen": StateType.COOL if value else StateType.NORMAL_TEMPERATURE,
        "Filled": StateType.FILLED_WITH_LIQUID if value else StateType.NO_LIQUID,
        "Saturated": StateType.SATURATED if value else StateType.NOT_SATURATED,
        "Heated": StateType.HOT if value else StateType.NORMAL_TEMPERATURE,
    }
    return state_mapping.get(state_name)


def _og_relation_to_relation_type(relation_name: str) -> Optional[RelationType]:
    relation_mapping = {
        "IsGrasping": RelationType.GRASPING,
        "OnTop": RelationType.ON_RECEPTACLE,
        "Inside": RelationType.INSIDE_RECEPTACLE,
    }
    return relation_mapping.get(relation_name)


def _set_object_properties_and_initial_states(
    graph: Graph, scene, objs_to_add: set, cache: dict
):
    for obj in objs_to_add:
        if obj not in cache["og_obj_to_entity_id"]:
            continue

        entity_id = cache["og_obj_to_entity_id"][obj]
        entity = graph.get_node_from_ID(entity_id)

        properties = _get_obj_properties_from_abilities(obj)
        for prop in properties:
            entity.set_property(prop)

        if hasattr(obj, "fixed_base") and obj.fixed_base:
            entity.set_property(PropertyType.STATIC)
        else:
            entity.set_property(PropertyType.MOVABLE)

            entity.set_property(PropertyType.GRASPABLE)

        for state_type, state_inst in obj.states.items():
            if not (
                issubclass(state_type, BooleanStateMixin)
                and issubclass(state_type, AbsoluteObjectState)
            ):
                continue

            if state_type in (Filled, Contains):
                continue
            try:
                value = state_inst.get_value()
                state_type_enum = _og_state_to_state_type(
                    get_state_name(state_type), value
                )
                if state_type_enum is not None:
                    required_props = STATE_REQUIREMENTS.get(state_type_enum, set())
                    if required_props.issubset(entity.attributes.properties):
                        entity.add_state(state_type_enum)
            except:
                pass

        if Covered in obj.states:
            covered_system_names = _get_covered_system_names(obj, scene)
            if covered_system_names:
                if PropertyType.CAN_BE_COVERED in entity.attributes.properties:
                    entity.add_state(
                        StateType.COVERED, {"systems": covered_system_names}
                    )
            else:
                if PropertyType.CAN_BE_COVERED in entity.attributes.properties:
                    entity.add_state(StateType.NOT_COVERED)

        filled_system_names = []
        contains_system_names = []
        if Filled in obj.states:
            filled_system_names = _get_filled_system_names(obj, scene)
        if Contains in obj.states:
            contains_system_names = _get_contains_system_names(obj, scene)
        if PropertyType.CAN_BE_FILLED_WITH_LIQUID in entity.attributes.properties:
            if filled_system_names:
                liquid_type = filled_system_names[0] if filled_system_names else None
                entity.add_state(
                    StateType.FILLED_WITH_LIQUID, {"liquid_type": liquid_type}
                )
            elif contains_system_names:
                liquid_type = (
                    contains_system_names[0] if contains_system_names else None
                )
                entity.add_state(StateType.CONTAIN_LIQUID, {"liquid_type": liquid_type})
            else:
                entity.add_state(StateType.NO_LIQUID)

        if PropertyType.CAN_BE_CLEANED in entity.attributes.properties:
            if _is_object_dirty(obj, scene):
                entity.add_state(StateType.DIRTY)
            else:
                entity.add_state(StateType.CLEAN)


def _add_object_relations(graph: Graph, objs_to_add: set, robot, cache: dict):
    entity_id_to_entity = {entity.ID: entity for entity in graph.graph.keys()}
    all_objs = objs_to_add | ({robot} if robot else set())

    for obj1 in all_objs:
        if obj1 not in cache["og_obj_to_entity_id"]:
            continue
        entity1_id = cache["og_obj_to_entity_id"][obj1]
        entity1 = entity_id_to_entity[entity1_id]

        for obj2 in all_objs:
            if obj2 == obj1 or obj2 not in cache["og_obj_to_entity_id"]:
                continue

            entity2_id = cache["og_obj_to_entity_id"][obj2]
            entity2 = entity_id_to_entity[entity2_id]

            for state_type, state_inst in obj1.states.items():
                if not (
                    issubclass(state_type, BooleanStateMixin)
                    and issubclass(state_type, RelativeObjectState)
                ):
                    continue

                state_name = get_state_name(state_type)
                if state_name == "IsGrasping":
                    continue

                try:
                    value = state_inst.get_value(obj2)
                    if value:
                        relation_type = _og_relation_to_relation_type(state_name)
                        if relation_type:
                            graph.add_edge(
                                entity1,
                                entity2,
                                relation_type,
                                flip_edge(relation_type),
                            )
                except:
                    pass


def _update_object_additions_and_removals(
    graph: Graph, scene, robot, cache: dict, full_obs: bool
):
    og_obj_to_entity_id = cache.get("og_obj_to_entity_id", {})
    room_name_to_entity_id = cache.get("room_name_to_entity_id", {})

    current_objs = set(scene.objects)

    current_objs = {obj for obj in current_objs if not isinstance(obj, BaseRobot)}

    if not full_obs and robot is not None:
        if object_states.ObjectsInFOVOfRobot in robot.states:
            objs_in_fov = robot.states[object_states.ObjectsInFOVOfRobot].get_value()
            current_objs &= objs_in_fov

    removed_objs = []
    for obj, entity_id in list(og_obj_to_entity_id.items()):
        if isinstance(obj, BaseRobot):
            continue
        if obj not in current_objs:
            removed_objs.append((obj, entity_id))

    for obj, entity_id in removed_objs:
        try:
            entity = graph.get_node_from_ID(entity_id)
            graph.pop_node(entity)
            del og_obj_to_entity_id[obj]
            if entity_id in cache.get("object_position", {}):
                del cache["object_position"][entity_id]
        except Exception:
            pass

    new_objs = []
    for obj in current_objs:
        if obj not in og_obj_to_entity_id:
            new_objs.append(obj)

    for obj in new_objs:
        category = getattr(obj, "category", "object")
        obj_id = obj.name

        try:
            pos, _ = obj.get_position_orientation()
            position = [float(pos[0]), float(pos[1]), float(pos[2])]
            if "cooked__" in category:
                new_category = category[8:]
            elif "half_" in category:
                new_category = category[5:] + "Sliced"
            elif "sliced_" in category:
                new_category = category[7:] + "Sliced"
            elif "diced__" in category:
                new_category = category[7:] + "Diced"
            else:
                new_category = category
            entity = Entity(
                ID=obj_id,
                type=EntityType.OBJECT,
                category=new_category,
                attributes=ObjectAttributes(),
                additional_info={
                    "asset_name": obj.name,
                    "position": position,
                    "prim_path": getattr(obj, "prim_path", None),
                },
            )
            if "cooked__" in category:
                entity.add_state(StateType.COOKED)
            graph.add_node(entity)
            og_obj_to_entity_id[obj] = obj_id
            cache["object_position"][obj_id] = position

            obj_room_id = _get_room_id_for_position(
                scene, pos, room_name_to_entity_id, obj=obj
            )
            if obj_room_id:
                graph.add_edge(
                    entity,
                    obj_room_id,
                    RelationType.INSIDE_ROOM,
                    flip_edge(RelationType.INSIDE_ROOM),
                )

            properties = _get_obj_properties_from_abilities(obj)
            for prop in properties:
                entity.set_property(prop)

            if hasattr(obj, "fixed_base") and obj.fixed_base:
                entity.set_property(PropertyType.STATIC)
            else:
                entity.set_property(PropertyType.MOVABLE)
                entity.set_property(PropertyType.GRASPABLE)

            for state_type, state_inst in obj.states.items():
                if not (
                    issubclass(state_type, BooleanStateMixin)
                    and issubclass(state_type, AbsoluteObjectState)
                ):
                    continue

                if state_type in (Filled, Contains):
                    continue
                try:
                    value = state_inst.get_value()
                    state_type_enum = _og_state_to_state_type(
                        get_state_name(state_type), value
                    )
                    if state_type_enum is not None:
                        required_props = STATE_REQUIREMENTS.get(state_type_enum, set())
                        if required_props.issubset(entity.attributes.properties):
                            entity.add_state(state_type_enum)
                except:
                    pass

            if Covered in obj.states:
                covered_system_names = _get_covered_system_names(obj, scene)
                if covered_system_names:
                    if PropertyType.CAN_BE_COVERED in entity.attributes.properties:
                        entity.add_state(
                            StateType.COVERED, {"systems": covered_system_names}
                        )
                else:
                    if PropertyType.CAN_BE_COVERED in entity.attributes.properties:
                        entity.add_state(StateType.NOT_COVERED)

            filled_system_names = []
            contains_system_names = []
            if Filled in obj.states:
                filled_system_names = _get_filled_system_names(obj, scene)
            if Contains in obj.states:
                contains_system_names = _get_contains_system_names(obj, scene)
            if PropertyType.CAN_BE_FILLED_WITH_LIQUID in entity.attributes.properties:
                if filled_system_names:
                    liquid_type = (
                        filled_system_names[0] if filled_system_names else None
                    )
                    entity.add_state(
                        StateType.FILLED_WITH_LIQUID, {"liquid_type": liquid_type}
                    )
                elif contains_system_names:
                    liquid_type = (
                        contains_system_names[0] if contains_system_names else None
                    )
                    entity.add_state(
                        StateType.CONTAIN_LIQUID, {"liquid_type": liquid_type}
                    )
                else:
                    entity.add_state(StateType.NO_LIQUID)

            if PropertyType.CAN_BE_CLEANED in entity.attributes.properties:
                if _is_object_dirty(obj, scene):
                    entity.add_state(StateType.DIRTY)
                else:
                    entity.add_state(StateType.CLEAN)

        except Exception:
            pass

    if len(new_objs) > 0:
        _add_object_relations_for_new_objects(graph, new_objs, scene, robot, cache)


def _add_object_relations_for_new_objects(
    graph: Graph, new_objs: list, scene, robot, cache: dict
):
    entity_id_to_entity = {entity.ID: entity for entity in graph.graph.keys()}
    og_obj_to_entity_id = cache.get("og_obj_to_entity_id", {})

    all_objs = set(scene.objects) | ({robot} if robot else set())
    all_objs = {
        obj for obj in all_objs if not isinstance(obj, BaseRobot) or obj == robot
    }

    for new_obj in new_objs:
        if new_obj not in og_obj_to_entity_id:
            continue

        entity1_id = og_obj_to_entity_id[new_obj]
        entity1 = entity_id_to_entity.get(entity1_id)
        if entity1 is None:
            continue

        for obj2 in all_objs:
            if obj2 == new_obj or obj2 not in og_obj_to_entity_id:
                continue

            entity2_id = og_obj_to_entity_id[obj2]
            entity2 = entity_id_to_entity.get(entity2_id)
            if entity2 is None:
                continue

            for state_type, state_inst in new_obj.states.items():
                if not (
                    issubclass(state_type, BooleanStateMixin)
                    and issubclass(state_type, RelativeObjectState)
                ):
                    continue

                state_name = get_state_name(state_type)
                if state_name == "IsGrasping":
                    continue

                try:
                    value = state_inst.get_value(obj2)
                    if value:
                        relation_type = _og_relation_to_relation_type(state_name)
                        if relation_type:
                            graph.add_edge(
                                entity1,
                                entity2,
                                relation_type,
                                flip_edge(relation_type),
                            )
                except:
                    pass

            for state_type, state_inst in obj2.states.items():
                if not (
                    issubclass(state_type, BooleanStateMixin)
                    and issubclass(state_type, RelativeObjectState)
                ):
                    continue

                state_name = get_state_name(state_type)
                if state_name == "IsGrasping":
                    continue

                try:
                    value = state_inst.get_value(new_obj)
                    if value:
                        relation_type = _og_relation_to_relation_type(state_name)
                        if relation_type:
                            graph.add_edge(
                                entity2,
                                entity1,
                                relation_type,
                                flip_edge(relation_type),
                            )
                except:
                    pass


def _update_object_relations(graph: Graph, scene, robot, cache: dict):
    entity_id_to_entity = {entity.ID: entity for entity in graph.graph.keys()}
    og_obj_to_entity_id = cache.get("og_obj_to_entity_id", {})
    room_name_to_entity_id = cache.get("room_name_to_entity_id", {})

    for obj, entity_id in og_obj_to_entity_id.items():
        if isinstance(obj, BaseRobot):
            continue
        try:
            pos, _ = obj.get_position_orientation()
            new_position = [float(pos[0]), float(pos[1]), float(pos[2])]
            old_position = cache["object_position"].get(entity_id)

            if old_position is None or old_position != new_position:
                entity = entity_id_to_entity[entity_id]
                entity.update_info({"position": new_position})
                cache["object_position"][entity_id] = new_position

                new_room_id = _get_room_id_for_position(
                    scene, pos, room_name_to_entity_id, obj=obj
                )
                if new_room_id:
                    old_rooms = graph.get_neighbors_of_type(entity, EntityType.ROOM)
                    for old_room in old_rooms:
                        graph.remove_edge(
                            entity,
                            old_room,
                            RelationType.INSIDE_ROOM,
                            flip_edge(RelationType.INSIDE_ROOM),
                        )

                    graph.add_edge(
                        entity,
                        new_room_id,
                        RelationType.INSIDE_ROOM,
                        flip_edge(RelationType.INSIDE_ROOM),
                    )
        except:
            pass


def _update_object_object_relations(graph: Graph, scene, robot, cache: dict):
    entity_id_to_entity = {entity.ID: entity for entity in graph.graph.keys()}
    og_obj_to_entity_id = cache.get("og_obj_to_entity_id", {})

    all_objs = {
        obj for obj in og_obj_to_entity_id.keys() if not isinstance(obj, BaseRobot)
    }

    for obj1 in all_objs:
        if obj1 not in og_obj_to_entity_id:
            continue

        entity1_id = og_obj_to_entity_id[obj1]
        entity1 = entity_id_to_entity.get(entity1_id)
        if entity1 is None:
            continue

        obj_pos_list = None
        try:
            obj_pos, _ = obj1.get_position_orientation()
            obj_pos_list = [float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2])]
            cached_pos = cache.get("object_position", {}).get(entity1_id, None)

            if cached_pos is not None and cached_pos == obj_pos_list:
                if robot is not None:
                    robot_entity_id = cache.get("og_obj_to_entity_id", {}).get(robot)
                    if robot_entity_id is not None:
                        robot_entity = entity_id_to_entity.get(robot_entity_id)
                        if robot_entity is not None and graph.has_edge(
                            robot_entity, entity1, RelationType.GRASPING
                        ):
                            pass
                        else:
                            continue
                else:
                    continue
        except:
            pass

        current_related_entities = {}
        for neighbor in graph.get_neighbors(entity1):
            if neighbor.type in [EntityType.OBJECT, EntityType.ROBOT]:
                if graph.has_edge(entity1, neighbor, RelationType.INSIDE_RECEPTACLE):
                    current_related_entities[neighbor] = RelationType.INSIDE_RECEPTACLE
                elif graph.has_edge(entity1, neighbor, RelationType.ON_RECEPTACLE):
                    current_related_entities[neighbor] = RelationType.ON_RECEPTACLE

        actual_related_entities = {}
        for obj2 in all_objs:
            if obj2 == obj1 or obj2 not in og_obj_to_entity_id:
                continue
            entity2_id = og_obj_to_entity_id[obj2]
            entity2 = entity_id_to_entity.get(entity2_id)
            if entity2 is None:
                continue

            for state_type, state_inst in obj1.states.items():
                if not (
                    issubclass(state_type, BooleanStateMixin)
                    and issubclass(state_type, RelativeObjectState)
                ):
                    continue
                state_name = get_state_name(state_type)
                if state_name == "IsGrasping":
                    continue
                try:
                    value = state_inst.get_value(obj2)
                    if value:
                        relation_type = _og_relation_to_relation_type(state_name)
                        if relation_type:
                            actual_related_entities[entity2] = relation_type
                except:
                    pass

        for related_entity, relation_type in current_related_entities.items():
            if (
                related_entity not in actual_related_entities
                or actual_related_entities[related_entity] != relation_type
            ):
                graph.remove_edge(
                    entity1, related_entity, relation_type, flip_edge(relation_type)
                )

        for related_entity, relation_type in actual_related_entities.items():
            if (
                related_entity not in current_related_entities
                or current_related_entities[related_entity] != relation_type
            ):
                if related_entity in current_related_entities:
                    old_relation_type = current_related_entities[related_entity]
                    graph.remove_edge(
                        entity1,
                        related_entity,
                        old_relation_type,
                        flip_edge(old_relation_type),
                    )

                graph.add_edge(
                    entity1, related_entity, relation_type, flip_edge(relation_type)
                )

        if obj_pos_list is not None:
            if "object_position" not in cache:
                cache["object_position"] = {}
            cache["object_position"][entity1_id] = obj_pos_list

            entity1.update_info({"position": obj_pos_list})


def _update_robot_room_associations(graph: Graph, scene, robot, cache: dict):
    if robot is None:
        return

    robot_entity_id = cache.get("og_obj_to_entity_id", {}).get(robot)
    if robot_entity_id is None:
        return

    room_name_to_entity_id = cache.get("room_name_to_entity_id", {})
    entity = graph.get_node_from_ID(robot_entity_id)

    robot_pos, _ = robot.get_position_orientation()
    new_position = [float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2])]
    entity.update_info({"position": new_position})
    cache["object_position"][robot_entity_id] = new_position

    new_room_id = _get_room_id_for_position(
        scene, robot_pos, room_name_to_entity_id, obj=robot
    )
    if new_room_id:
        old_rooms = graph.get_neighbors_of_type(entity, EntityType.ROOM)
        for old_room in old_rooms:
            graph.remove_edge(
                entity,
                old_room,
                RelationType.INSIDE_ROOM,
                flip_edge(RelationType.INSIDE_ROOM),
            )

        graph.add_edge(
            entity,
            new_room_id,
            RelationType.INSIDE_ROOM,
            flip_edge(RelationType.INSIDE_ROOM),
        )


def _update_robot_grasping_relations(graph: Graph, robot, grasped_obj, cache: dict):
    if robot is None:
        return

    robot_entity_id = cache.get("og_obj_to_entity_id", {}).get(robot)
    if robot_entity_id is None:
        return

    robot_entity = graph.get_node_from_ID(robot_entity_id)
    og_obj_to_entity_id = cache.get("og_obj_to_entity_id", {})

    if grasped_obj is None:
        grasped_obj = []
    elif not isinstance(grasped_obj, list):
        grasped_obj = [grasped_obj] if grasped_obj else []

    grasped_entity_ids = set()
    for obj in grasped_obj:
        if obj in og_obj_to_entity_id:
            grasped_entity_ids.add(og_obj_to_entity_id[obj])

    current_grasped_entities = []
    for neighbor in graph.get_neighbors(robot_entity):
        if neighbor.type == EntityType.OBJECT:
            if graph.has_edge(robot_entity, neighbor, RelationType.GRASPING):
                current_grasped_entities.append(neighbor)

    for entity in current_grasped_entities:
        if entity.ID not in grasped_entity_ids:
            graph.remove_edge(
                robot_entity,
                entity,
                RelationType.GRASPING,
                flip_edge(RelationType.GRASPING),
            )

    for grasped_entity_id in grasped_entity_ids:
        grasped_entity = graph.get_node_from_ID(grasped_entity_id)
        if grasped_entity is not None:
            if not graph.has_edge(robot_entity, grasped_entity, RelationType.GRASPING):
                graph.add_edge(
                    robot_entity,
                    grasped_entity,
                    RelationType.GRASPING,
                    flip_edge(RelationType.GRASPING),
                )


def _update_object_states(graph: Graph, scene, cache: dict):
    og_obj_to_entity_id = cache.get("og_obj_to_entity_id", {})
    entity_id_to_entity = {entity.ID: entity for entity in graph.graph.keys()}
    for obj, entity_id in og_obj_to_entity_id.items():
        if isinstance(obj, BaseRobot):
            continue
        entity = entity_id_to_entity.get(entity_id)
        if entity is None:
            continue

        for state_type, state_inst in obj.states.items():
            if not (
                issubclass(state_type, BooleanStateMixin)
                and issubclass(state_type, AbsoluteObjectState)
            ):
                continue

            if state_type in (Filled, Contains):
                continue
            try:
                value = state_inst.get_value()
                state_type_enum = _og_state_to_state_type(
                    get_state_name(state_type), value
                )
                if state_type_enum is not None:
                    required_props = STATE_REQUIREMENTS.get(state_type_enum, set())
                    if required_props.issubset(entity.attributes.properties):
                        entity.add_state(state_type_enum)
            except:
                pass

        if Covered in obj.states:
            covered_system_names = _get_covered_system_names(obj, scene)
            if PropertyType.CAN_BE_COVERED in entity.attributes.properties:
                if covered_system_names:
                    entity.add_state(
                        StateType.COVERED, {"systems": covered_system_names}
                    )
                else:
                    entity.add_state(StateType.NOT_COVERED)
        if Saturated in obj.states:
            saturated_system_names = _get_saturated_system_names(obj, scene)
            if PropertyType.CAN_BE_STATURATED in entity.attributes.properties:
                if saturated_system_names and isinstance(saturated_system_names, list):
                    entity.add_state(
                        StateType.SATURATED, {"liquid_type": saturated_system_names[0]}
                    )
                elif saturated_system_names and isinstance(saturated_system_names, str):
                    entity.add_state(
                        StateType.SATURATED, {"liquid_type": saturated_system_names}
                    )
                else:
                    entity.add_state(StateType.NOT_SATURATED)

        filled_system_names = []
        contains_system_names = []

        if Filled in obj.states:
            filled_system_names = _get_filled_system_names(obj, scene)
        if Contains in obj.states:
            contains_system_names = _get_contains_system_names(obj, scene)

        if PropertyType.CAN_BE_FILLED_WITH_LIQUID in entity.attributes.properties:
            current_state = entity.attributes.states.get(StateType.FILLED_WITH_LIQUID)
            current_contain_state = entity.attributes.states.get(
                StateType.CONTAIN_LIQUID
            )
            if filled_system_names:
                liquid_type = filled_system_names[0] if filled_system_names else None

                if current_state is None or current_state.params != {
                    "liquid_type": liquid_type
                }:
                    entity.add_state(
                        StateType.FILLED_WITH_LIQUID, {"liquid_type": liquid_type}
                    )
            elif contains_system_names:
                liquid_type = (
                    contains_system_names[0] if contains_system_names else None
                )
                if current_contain_state is None or current_contain_state.params != {
                    "liquid_type": liquid_type
                }:
                    entity.add_state(
                        StateType.CONTAIN_LIQUID, {"liquid_type": liquid_type}
                    )
            else:
                if current_state is not None or current_contain_state is not None:
                    entity.add_state(StateType.NO_LIQUID)

        if PropertyType.CAN_BE_CLEANED in entity.attributes.properties:
            if _is_object_dirty(obj, scene):
                entity.add_state(StateType.DIRTY)
            else:
                entity.add_state(StateType.CLEAN)
