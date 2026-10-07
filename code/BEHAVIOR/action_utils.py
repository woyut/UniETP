import sys
import os

from typing import List, Optional


_current_dir = os.path.dirname(os.path.abspath(__file__))
_omnigibson_path = os.path.join(_current_dir, "BEHAVIOR-1K", "OmniGibson")
if os.path.exists(_omnigibson_path) and _omnigibson_path not in sys.path:
    sys.path.insert(0, _omnigibson_path)

from omnigibson import object_states
from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError
from omnigibson.envs import Environment
from omnigibson.object_states.object_state_base import (
    BaseObjectState,
    RelativeObjectState,
)
from omnigibson.object_states.particle_modifier import ParticleApplier, ParticleModifier
from omnigibson.objects import BaseObject, StatefulObject
from omnigibson.scenes import Scene
from omnigibson.systems.system_base import BaseSystem
from omnigibson.utils.constants import PrimType


def find_task_related_object(
    env: Environment,
    target_name: str,
    retain_wrapper: bool = False,
) -> Optional[BaseObject]:
    task_related_objects = sorted(
        list(env.templateTask.object_scope.keys()),
        key=lambda name: len(name),
        reverse=True,
    )

    if target_name in task_related_objects:
        ref = env.templateTask.object_scope[target_name]
        target_obj = ref if retain_wrapper else ref.wrapped_obj
        return target_obj

    for name in task_related_objects:
        ref = env.templateTask.object_scope[name]
        if "agent" in name:
            continue

        target_name = target_name.strip()
        candidate_name = name.split(".")[0].strip()
        if target_name in candidate_name:
            target_obj = ref if retain_wrapper else ref.wrapped_obj
            return target_obj

    return None


def is_visual_or_physical_particle_system(scene: Scene, system: BaseSystem) -> bool:
    if scene.is_visual_particle_system(system_name=system.name):
        return True
    if scene.is_physical_particle_system(system_name=system.name):
        return True
    return False


def get_obj_with_state(
    obj: StatefulObject | str, state: BaseObjectState, env: Optional[Environment] = None
) -> Optional[StatefulObject]:
    if isinstance(obj, str):
        assert env is not None
        obj = find_task_related_object(env, obj)

    if obj is None:
        return None
    if not hasattr(obj, "states"):
        return None
    if state not in obj.states:
        return None
    return obj


def get_covered_systems(
    obj: StatefulObject | str, env: Optional[Environment] = None
) -> Optional[List[BaseSystem]]:
    covering_systems = set()
    obj = get_obj_with_state(obj, object_states.Covered, env)
    if obj is None:
        return None

    for system in obj.scene.system_registry.objects:
        if not is_visual_or_physical_particle_system(obj.scene, system):
            continue

        if (
            hasattr(obj, "prim_type")
            and obj.prim_type == PrimType.CLOTH
            and obj.scene.is_physical_particle_system(system_name=system.name)
        ):
            continue
        try:
            if obj.states[object_states.Covered].get_value(system):
                covering_systems.add(system)
        except (ValueError, AssertionError):
            continue

    return list(covering_systems)


def get_contained_systems(
    obj: StatefulObject | str, env: Optional[Environment] = None
) -> Optional[List[BaseSystem]]:
    contained_systems = set()
    obj = get_obj_with_state(obj, object_states.Contains, env)
    if obj is None:
        return None

    for system in obj.scene.system_registry.objects:
        if not is_visual_or_physical_particle_system(obj.scene, system):
            continue
        if obj.states[object_states.Contains].get_value(system):
            contained_systems.add(system)

    return list(contained_systems)


def get_appliable_systems(
    obj: StatefulObject | str, env: Optional[Environment] = None
) -> Optional[List[BaseSystem]]:
    appliable_systems = set()
    obj = get_obj_with_state(obj, ParticleApplier, env)
    if obj is None:
        return None

    particle_applier_state = obj.states[ParticleApplier]
    conditions_keys = list(particle_applier_state.conditions.keys())

    for system in obj.scene.system_registry.objects:
        if not is_visual_or_physical_particle_system(obj.scene, system):
            continue

        system_name = system.name
        if particle_applier_state.check_conditions_for_system(system_name):
            appliable_systems.add(system)
            continue
        for cond_key in conditions_keys:
            if system_name in cond_key or cond_key in system_name:
                appliable_systems.add(system)
    return list(appliable_systems)


def get_produced_systems(
    obj: StatefulObject | str, env: Optional[Environment] = None
) -> Optional[List[BaseSystem]]:
    producing_systems = set()
    obj = get_obj_with_state(obj, object_states.ParticleSource, env)
    if obj is None:
        return None

    for system in obj.scene.system_registry.objects:
        if obj.states[object_states.ParticleSource].check_conditions_for_system(
            system.name
        ):
            producing_systems.add(system)

    return list(producing_systems)


def get_supported_systems(
    tool: StatefulObject,
    systems: List[BaseSystem],
    modifier: ParticleModifier,
) -> List[BaseSystem]:
    supported_systems = set()

    for system in systems:
        if tool.states[modifier].supports_system(system.name):
            supported_systems.add(system)

    return list(supported_systems)


def is_target_object_predicate_with_obj(
    target_obj: StatefulObject, obj: StatefulObject, predicate: RelativeObjectState
) -> bool:
    if not hasattr(target_obj, "states"):
        return False
    if not predicate in target_obj.states:
        return False
    return target_obj.states[predicate].get_value(obj)


def check_open_before_grasp(obj: StatefulObject, env: Environment):
    for parent_obj_name in env.templateTask.object_scope.keys():
        if "agent" in parent_obj_name or "robot" in parent_obj_name:
            continue
        parent_obj = get_obj_with_state(parent_obj_name, object_states.Open, env)
        if parent_obj is None:
            continue

        if (
            is_target_object_predicate_with_obj(obj, parent_obj, object_states.Inside)
            and parent_obj.states[object_states.Open].get_value() is False
        ):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                f"You should open {parent_obj.name} first, because currently the operated object is placed inside {parent_obj.name}.",
                {
                    "operated object": obj.name,
                    "parent object should be opened first": parent_obj.name,
                },
            )
