# Portions of this file are adapted from OmniGibson
# (https://github.com/StanfordVL/BEHAVIOR-1K), in particular from
# omnigibson/action_primitives/symbolic_semantic_action_primitives.py,
# omnigibson/action_primitives/starter_semantic_action_primitives.py and
# omnigibson/systems/macro_particle_system.py.
# Copyright (c) 2023 Stanford Vision and Learning Group. Licensed under the MIT License.
# The adapted code has been modified for this project.

import os
from BEHAVIOR.action_utils import *
from typing import Any
from aenum import IntEnum, auto
from omnigibson import object_states
from omnigibson.object_states.contains import ContainedParticles
from omnigibson.action_primitives.action_primitive_set_base import (
    ActionPrimitiveError,
    ActionPrimitiveErrorGroup,
)
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
)
from omnigibson.objects import DatasetObject
from omnigibson.objects import StatefulObject
from omnigibson.robots import BaseRobot
from omnigibson.systems import BaseSystem
from omnigibson.transition_rules import SlicingRule
from omnigibson.utils.constants import PrimType
from omnigibson.systems.macro_particle_system import (
    MacroVisualParticleSystem,
    MacroParticleSystem,
)
from omnigibson.object_states.filled import m as filled_m
from omnigibson.utils.python_utils import torch_delete
import torch as th
from omnigibson.macros import create_module_macros
import numpy as np
from BEHAVIOR.env_utils import drop_obj, grasp, distance_to_cuboid, move
from omnigibson.object_states.particle_modifier import ParticleApplier
from omnigibson.object_states.saturated import ModifiedParticles
import json
import math
from functools import lru_cache
import omnigibson as og
from omnigibson.utils.bddl_utils import OBJECT_TAXONOMY
from pyquaternion import Quaternion

m = create_module_macros(
    module_path=os.path.join(
        os.path.dirname(og.__file__),
        "action_primitives",
        "modified_semantic_action_primitives.py",
    )
)
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    m as parent_m,
)

m.MAX_STEPS_FOR_SETTLING = getattr(parent_m, "MAX_STEPS_FOR_SETTLING", 10)


@lru_cache(maxsize=1)
def _behavior_furniture_synsets() -> frozenset[str] | None:
    try:
        p = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "task",
            "commonsense_knowledge",
            "behavior_knowledge.json",
        )
        with open(os.path.normpath(p), encoding="utf-8") as f:
            data = json.load(f)
        g = data.get("furniture")
        if not isinstance(g, list):
            return None
        return frozenset(str(x).strip() for x in g if str(x).strip())
    except Exception:
        return None


def _check_pick_target_synset_not_furniture(obj: DatasetObject) -> None:
    """Reject grasp/pick when the target is listed as furniture."""
    furniture_syns = _behavior_furniture_synsets()
    if not furniture_syns:
        return
    category = getattr(obj, "category", None)
    if not category:
        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
            "Cannot validate pick/grasp: target object has no category.",
            {"target object": getattr(obj, "name", repr(obj))},
        )
    try:
        syn = OBJECT_TAXONOMY.get_synset_from_category(category)
    except Exception:
        syn = None
    if not syn:
        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
            "Cannot resolve WordNet synset for target object category.",
            {"target object": getattr(obj, "name", repr(obj)), "category": category},
        )
    if syn in furniture_syns:
        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
            "The target object is not graspable.",
            {
                "synset": syn,
                "category": category,
                "target object": getattr(obj, "name", repr(obj)),
            },
        )


def _patched_remove_particle_by_name(self, name):
    face_id_to_remove = None
    parent_obj = None
    group = None
    if name in self._particles_info:
        parent_obj = self._particles_info[name].get("obj")
        if parent_obj is not None and self._is_cloth_obj(obj=parent_obj):
            group = self.get_group_name(obj=parent_obj)
            face_id_to_remove = self._particles_info[name].get("face_id")
    try:
        super(type(self), self).remove_particle_by_name(name=name)
    except (KeyError, RuntimeError):
        pass
    if name not in self._particles_info:
        if group is not None and group in self._group_particles:
            self._group_particles[group].pop(name, None)
        self._particles_local_mat.pop(name, None)
        return
    if parent_obj is None:
        parent_obj = self._particles_info[name]["obj"]
    if group is None:
        group = self.get_group_name(obj=parent_obj)
    if name not in self._group_particles.get(group, {}):
        self._particles_info.pop(name, None)
        self._particles_local_mat.pop(name, None)
        return
    self._group_particles[group].pop(name, None)
    self._particles_local_mat.pop(name, None)
    particle_info = self._particles_info.pop(name, None)
    if particle_info is None:
        return
    if self._is_cloth_obj(obj=parent_obj):
        if group in self._cloth_face_ids and self._cloth_face_ids[group] is not None:
            face_ids = self._cloth_face_ids[group]
            face_id = (
                face_id_to_remove
                if face_id_to_remove is not None
                else particle_info.get("face_id")
            )
            if face_id is not None:
                try:
                    idx_mapping = {
                        int(face_id): i for i, face_id in enumerate(face_ids)
                    }
                    if int(face_id) in idx_mapping:
                        self._cloth_face_ids[group] = torch_delete(
                            face_ids, idx_mapping[int(face_id)]
                        )
                    else:
                        remaining_face_ids = []
                        for particle_name in self._group_particles[group].keys():
                            if particle_name in self._particles_info:
                                remaining_face_id = self._particles_info[
                                    particle_name
                                ].get("face_id")
                                if remaining_face_id is not None:
                                    remaining_face_ids.append(int(remaining_face_id))
                        if remaining_face_ids:
                            self._cloth_face_ids[group] = th.tensor(
                                remaining_face_ids,
                                dtype=face_ids.dtype,
                                device=face_ids.device,
                            )
                        else:
                            self._cloth_face_ids.pop(group, None)
                except (KeyError, IndexError, RuntimeError) as e:
                    remaining_face_ids = []
                    for particle_name in self._group_particles[group].keys():
                        if particle_name in self._particles_info:
                            remaining_face_id = self._particles_info[particle_name].get(
                                "face_id"
                            )
                            if remaining_face_id is not None:
                                remaining_face_ids.append(int(remaining_face_id))
                    if remaining_face_ids:
                        self._cloth_face_ids[group] = th.tensor(
                            remaining_face_ids,
                            dtype=face_ids.dtype,
                            device=face_ids.device,
                        )
                    else:
                        self._cloth_face_ids.pop(group, None)


try:
    if not hasattr(MacroVisualParticleSystem, "_original_remove_particle_by_name"):
        MacroVisualParticleSystem._original_remove_particle_by_name = (
            MacroVisualParticleSystem.remove_particle_by_name
        )
    MacroVisualParticleSystem.remove_particle_by_name = _patched_remove_particle_by_name
except ImportError:
    pass


class ModifiedSemanticActionPrimitiveSet(IntEnum):
    _init_ = "value __doc__"
    GRASP = auto(), "Grasp an object"
    PLACE_ON_TOP = auto(), "Place the currently grasped object on top of another object"
    PLACE_INSIDE = auto(), "Place the currently grasped object inside another object"
    OPEN = auto(), "Open an object"
    CLOSE = auto(), "Close an object"
    TOGGLE_ON = auto(), "Toggle an object on"
    TOGGLE_OFF = auto(), "Toggle an object off"
    SOAK = auto(), "Soak the currently grasped object."
    CLEAN_OBJ = auto(), "Clean the object."
    CUT = (
        auto(),
        "Cut (slice or dice) the given object with the currently grasped object.",
    )
    NAVIGATE_TO = auto(), "Navigate to an object"
    RELEASE = (
        auto(),
        "Release an object, letting it fall to the ground. You can then grasp it again, as a way of reorienting your grasp of the object.",
    )
    FILL_WITH = (
        auto(),
        "Fill the target_obj with particles produced by the fluid source",
    )
    POUR_INTO = (
        auto(),
        "Pour the particle in the fluid_container into the target_obj (usually a container)",
    )
    SPREAD = (
        auto(),
        "Spread some particles onto some object, make object covered with these particles",
    )
    BURN = auto(), "Make the target object burnt"
    COOK = auto(), "Cook the target object(high level action)"
    WASH = auto(), "Wash the target object(high level action)"
    COOL = auto(), "Cool the target object(high level action)"


class ModifiedSemanticActionPrimitives(StarterSemanticActionPrimitives):
    def __init__(self, env_wrapper, robot: BaseRobot):
        super().__init__(env_wrapper.env, robot, skip_curobo_initilization=True)
        self.env_wrapper = env_wrapper
        self._filled_containers = {}
        self.controller_functions = {
            ModifiedSemanticActionPrimitiveSet.GRASP: self._grasp,
            ModifiedSemanticActionPrimitiveSet.PLACE_ON_TOP: self._place_on_top,
            ModifiedSemanticActionPrimitiveSet.PLACE_INSIDE: self._place_inside,
            ModifiedSemanticActionPrimitiveSet.OPEN: self._open,
            ModifiedSemanticActionPrimitiveSet.CLOSE: self._close,
            ModifiedSemanticActionPrimitiveSet.TOGGLE_ON: self._toggle_on,
            ModifiedSemanticActionPrimitiveSet.TOGGLE_OFF: self._toggle_off,
            ModifiedSemanticActionPrimitiveSet.SOAK: self._soak,
            ModifiedSemanticActionPrimitiveSet.CUT: self._cut,
            ModifiedSemanticActionPrimitiveSet.NAVIGATE_TO: self._navigate_to_obj,
            ModifiedSemanticActionPrimitiveSet.RELEASE: self._release,
            ModifiedSemanticActionPrimitiveSet.FILL_WITH: self._fill_with,
            ModifiedSemanticActionPrimitiveSet.POUR_INTO: self._pour_into,
            ModifiedSemanticActionPrimitiveSet.SPREAD: self._spread,
            ModifiedSemanticActionPrimitiveSet.BURN: self._burn,
            ModifiedSemanticActionPrimitiveSet.COOK: self._cook,
            ModifiedSemanticActionPrimitiveSet.WASH: self._wash,
            ModifiedSemanticActionPrimitiveSet.CLEAN_OBJ: self._clean_obj,
            ModifiedSemanticActionPrimitiveSet.COOL: self._cool,
        }

    def _grasped_or_not(self):
        return self.env_wrapper.grasped_obj

    def apply_ref(self, primitive, *args: Any, attempts=3):
        assert attempts > 0, "Must make at least one attempt"
        ctrl = self.controller_functions[primitive]

        if any(isinstance(arg, BaseRobot) for arg in args):
            raise ActionPrimitiveErrorGroup(
                [
                    ActionPrimitiveError(
                        ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                        "Cannot call a symbolic semantic action primitive with a robot as an argument.",
                    )
                ]
            )

        errors = []
        for attempt_num in range(attempts):
            success = False
            try:
                yield from ctrl(*args)
                success = True
            except ActionPrimitiveError as e:
                errors.append(e)

            try:
                # Settle before returning.
                yield from self._settle_robot()
            except ActionPrimitiveError:
                pass

            # Stop on success
            if success:
                return

        raise ActionPrimitiveErrorGroup(errors)

    def _open_or_close(self, obj, should_open, delta=2.0):
        if object_states.Open not in obj.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is not openable.",
                {"target object": obj.name},
            )
        # Don't do anything if the object is already closed and we're trying to close.
        if should_open == obj.states[object_states.Open].get_value():
            return
        robot_pos = self.env_wrapper.robot.get_position()
        dis = distance_to_cuboid(robot_pos[:2], (obj.aabb[0][:2], obj.aabb[1][:2]))
        if dis > delta:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "The target is too far away.",
                {
                    "target object": obj.name,
                    "is it currently open": obj.states[object_states.Open].get_value(),
                },
            )
        # Set the value
        obj.states[object_states.Open].set_value(should_open)
        # Settle
        yield from self._settle_robot()
        if obj.states[object_states.Open].get_value() != should_open:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                f"The target object did not {'open' if should_open else 'close'}.",
                {
                    "target object": obj.name,
                    "is it currently open": obj.states[object_states.Open].get_value(),
                },
            )

    def _open(self, obj, delta=2.0):
        yield from self._open_or_close(obj, True, delta)

    def _close(self, obj, delta=2.0):
        yield from self._open_or_close(obj, False, delta)

    def _grasp(self, obj: DatasetObject):
        _check_pick_target_synset_not_furniture(obj)
        grasped_obj = self.env_wrapper.grasped_obj
        task_objs = self.env_wrapper.task_objs

        ret, msg = grasp(self.robot, grasped_obj, obj, task_objs, delta=2.0)
        if not ret:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                msg,
                {"target object": obj.name},
            )

        for _ in range(3):
            og.sim.step_physics()
        yield from self._settle_robot()

    def _release(self, drop_position=None):
        grasped_obj = self.env_wrapper.grasped_obj
        task_objs = self.env_wrapper.task_objs
        if len(grasped_obj) == 0:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The agent is not holding an object.",
            )

        if drop_position is not None:
            if isinstance(drop_position, (list, tuple)) and len(drop_position) == 2:
                drop_pos = drop_position[0]
            else:
                drop_pos = drop_position

            if isinstance(drop_pos, th.Tensor):
                drop_pos = drop_pos.cpu().numpy()
            elif not isinstance(drop_pos, np.ndarray):
                drop_pos = np.array(drop_pos)

            if len(drop_pos) < 3:
                drop_pos = np.append(drop_pos, [0.004])
        else:
            drop_pos = None

        ret, msg = drop_obj(self.robot, grasped_obj, task_objs, drop_position=drop_pos)
        if not ret:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                msg,
            )

        yield from self._settle_robot()

    def _toggle(self, obj, value):
        if object_states.ToggledOn not in obj.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is not toggleable.",
                {"target object": obj.name},
            )
        if obj.states[object_states.ToggledOn].get_value() == value:
            return
        # Call the setter
        obj.states[object_states.ToggledOn].set_value(value)
        # Yield some actions
        yield from self._settle_robot()
        if ParticleApplier in obj.states:
            obj.states[object_states.ToggledOn].clear_cache()
        # Check that it actually happened
        if obj.states[object_states.ToggledOn].get_value() != value:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                f"The target object did not turn {'on' if value else 'off'}.",
                {
                    "target object": obj.name,
                    "is it currently toggled on": obj.states[
                        object_states.ToggledOn
                    ].get_value(),
                },
            )

    def _washer_appliance_door_closed(self, appliance_obj: StatefulObject) -> bool:
        if object_states.Open not in appliance_obj.states:
            return True
        return not appliance_obj.states[object_states.Open].get_value()

    def _related_in_or_on(
        self, other_obj: StatefulObject, container: StatefulObject
    ) -> bool:
        if not hasattr(other_obj, "states"):
            return False
        if object_states.Inside in other_obj.states:
            try:
                if other_obj.states[object_states.Inside].get_value(container):
                    return True
            except (ValueError, AssertionError, RuntimeError, AttributeError):
                pass
        if object_states.OnTop in other_obj.states:
            try:
                if other_obj.states[object_states.OnTop].get_value(container):
                    return True
            except (ValueError, AssertionError, RuntimeError, AttributeError):
                pass
        return False

    def _related_inside_only(
        self, other_obj: StatefulObject, container: StatefulObject
    ) -> bool:
        if (
            not hasattr(other_obj, "states")
            or object_states.Inside not in other_obj.states
        ):
            return False
        try:
            return bool(other_obj.states[object_states.Inside].get_value(container))
        except (ValueError, AssertionError, RuntimeError, AttributeError):
            return False

    def _resolve_water_systems_for_applier(self, fluid_src: StatefulObject) -> list:
        systems = get_produced_systems(fluid_src) or []
        water = [
            s for s in systems if "water" in (getattr(s, "name", "") or "").lower()
        ]
        if water:
            return water
        reg = getattr(getattr(fluid_src, "scene", None), "system_registry", None)
        if reg is None:
            return []
        for system in reg.objects:
            if (getattr(system, "name", "") or "").lower() == "water":
                return [system]
        return []

    def _semantic_wash_clear_stains(self, target_obj: StatefulObject) -> None:
        is_cloth = (
            hasattr(target_obj, "prim_type") and target_obj.prim_type == PrimType.CLOTH
        )
        if is_cloth:
            for system in target_obj.scene.system_registry.objects:
                if not is_visual_or_physical_particle_system(target_obj.scene, system):
                    continue
                if object_states.Saturated in target_obj.states and target_obj.states[
                    object_states.Saturated
                ].get_value(system):
                    if (
                        ModifiedParticles in target_obj.states
                        and object_states.Saturated in target_obj.states
                    ):
                        target_obj.states[ModifiedParticles].set_value(system, 0)
                        target_obj.states[object_states.Saturated].clear_cache()
            return
        covered_systems = get_covered_systems(target_obj) or []
        for system in covered_systems:
            try:
                if object_states.Covered in target_obj.states:
                    target_obj.states[object_states.Covered].set_value(system, False)
            except (ValueError, AssertionError, KeyError, RuntimeError) as e:
                pass

    def _toggle_on_try_fill_related_with_water(
        self, sink_obj: StatefulObject, scene_objects: list
    ):
        if not scene_objects:
            return
        water_systems = self._resolve_water_systems_for_applier(sink_obj)
        if not water_systems:
            return
        for other_obj in scene_objects:
            if other_obj == sink_obj or not hasattr(other_obj, "states"):
                continue
            if not self._related_in_or_on(other_obj, sink_obj):
                continue
            if object_states.Filled not in other_obj.states:
                continue
            for ws in water_systems:
                try:
                    yield from self._run_fill_system_on_target(
                        other_obj, ws, log_prefix="[toggle_on fill]"
                    )
                    break
                except (
                    ValueError,
                    AssertionError,
                    KeyError,
                    RuntimeError,
                    AttributeError,
                ):
                    pass

    def _toggle_on_try_wash_related(
        self,
        appliance_obj: StatefulObject,
        scene_objects: list,
        *,
        washer_inside_only: bool,
    ) -> None:
        if not scene_objects:
            return
        for other_obj in scene_objects:
            if other_obj == appliance_obj or not hasattr(other_obj, "states"):
                continue
            if washer_inside_only:
                if not self._related_inside_only(other_obj, appliance_obj):
                    continue
            else:
                if not self._related_in_or_on(other_obj, appliance_obj):
                    continue
            covered = get_covered_systems(other_obj)
            if not covered:
                continue
            self._semantic_wash_clear_stains(other_obj)

    def _apply_soak_particles(
        self, recipient: StatefulObject, fluid_source: StatefulObject
    ) -> int:
        if (
            object_states.ParticleSource not in fluid_source.states
            and object_states.Contains not in fluid_source.states
        ):
            return 0
        if object_states.ParticleSource in fluid_source.states:
            producing = get_produced_systems(fluid_source) or []
        else:
            producing = get_contained_systems(fluid_source) or []
        if not producing:
            return 0
        if (
            object_states.Saturated not in recipient.states
            or object_states.ParticleRemover not in recipient.states
            or ModifiedParticles not in recipient.states
        ):
            return 0
        supported = get_supported_systems(
            recipient,
            producing,
            object_states.ParticleRemover,
        )
        if not supported:
            return 0
        sat = recipient.states[object_states.Saturated]
        mp = recipient.states[ModifiedParticles]
        n = 0
        for system in supported:
            limit = sat.get_limit(system)
            mp.set_value(system, limit)
            sat.clear_cache()
            n += 1
        return n

    def _toggle_on_try_soak_related(
        self, sink_obj: StatefulObject, scene_objects: list
    ) -> None:
        if not scene_objects:
            return
        for other_obj in scene_objects:
            if other_obj == sink_obj or not hasattr(other_obj, "states"):
                continue
            if not self._related_in_or_on(other_obj, sink_obj):
                continue
            if object_states.Saturated not in other_obj.states:
                continue
            self._apply_soak_particles(other_obj, sink_obj)

    def _toggle_on(self, obj):
        yield from self._toggle(obj, True)
        objs_to_cook = []
        scene_objects = (
            obj.scene.objects
            if hasattr(obj, "scene") and hasattr(obj.scene, "objects")
            else []
        )
        check_inside = (
            "pressure_cooker" in obj.category
            or "toaster" in obj.category
            or "oven" in obj.category
        )
        check_ontop = (
            "grill" in obj.category
            or "stove" in obj.category
            or "burner" in obj.category
            or "smoker" in obj.category
        )
        whether_to_burn = "stove" in obj.category or "burner" in obj.category
        whether_to_wash = "washer" in obj.category or "sink" in obj.category
        whether_to_fill = "sink" in obj.category
        is_washer_appliance = "washer" in obj.category
        is_sink_only = "sink" in obj.category and not is_washer_appliance
        if check_inside or check_ontop:
            for other_obj in scene_objects:
                if other_obj == obj:
                    continue
                if not hasattr(other_obj, "states"):
                    continue
                has_relation = False
                if check_inside and object_states.Inside in other_obj.states:
                    if other_obj.states[object_states.Inside].get_value(obj):
                        has_relation = True
                if check_ontop and object_states.OnTop in other_obj.states:
                    if other_obj.states[object_states.OnTop].get_value(obj):
                        has_relation = True
                if has_relation and (
                    object_states.Cooked in other_obj.states
                    or (whether_to_burn and object_states.Burnt in other_obj.states)
                ):
                    objs_to_cook.append(other_obj)
        for cook_obj in objs_to_cook:
            if whether_to_burn and object_states.Burnt in cook_obj.states:
                cook_obj.states[object_states.Burnt].set_value(True)
                if object_states.Temperature in cook_obj.states:
                    burn_temp = cook_obj.states[object_states.Burnt].burn_temperature
                    cook_obj.states[object_states.Temperature].set_value(burn_temp + 15)
                    if object_states.MaxTemperature in cook_obj.states:
                        cook_obj.states[object_states.MaxTemperature].set_value(
                            burn_temp + 15
                        )
            elif object_states.Cooked in cook_obj.states:
                cook_obj.states[object_states.Cooked].set_value(True)
                if object_states.Temperature in cook_obj.states:
                    cook_temp = cook_obj.states[object_states.Cooked].cook_temperature
                    cook_obj.states[object_states.Temperature].set_value(cook_temp + 15)
                    if object_states.MaxTemperature in cook_obj.states:
                        cook_obj.states[object_states.MaxTemperature].set_value(
                            cook_temp + 15
                        )
        if whether_to_wash and is_sink_only:
            self._toggle_on_try_wash_related(
                obj, scene_objects, washer_inside_only=False
            )
            self._toggle_on_try_soak_related(obj, scene_objects)
        if whether_to_fill:
            yield from self._toggle_on_try_fill_related_with_water(obj, scene_objects)
        if (
            whether_to_wash
            and is_washer_appliance
            and self._washer_appliance_door_closed(obj)
        ):
            self._toggle_on_try_wash_related(
                obj, scene_objects, washer_inside_only=True
            )

    def _toggle_off(self, obj):
        yield from self._toggle(obj, False)

    def _place_with_predicate(
        self, obj, predicate, near_poses=None, near_poses_threshold=None
    ):
        grasped_obj = self.env_wrapper.grasped_obj
        if len(grasped_obj) == 0:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The agent is not holding an object.",
            )
        obj_in_hand = grasped_obj[0]
        # Find a spot to put it
        obj_pose = self._sample_pose_with_object_and_predicate(
            predicate,
            obj_in_hand,
            obj,
            near_poses=near_poses,
            near_poses_threshold=near_poses_threshold,
        )
        yield from self._release(drop_position=obj_pose[0])

        obj_in_hand.set_position_orientation(*obj_pose)
        yield from self._settle_robot()
        if not obj_in_hand.states[predicate].get_value(obj):
            yield from self._grasp(obj_in_hand)
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "The object was not placed successfully.",
                {"dropped object": obj_in_hand.name, "target object": obj.name},
            )

        if predicate == object_states.Inside and (
            ("fridge" in obj.name) or ("refrigerator" in obj.name)
        ):

            def _try_set_frozen(x: StatefulObject):
                if hasattr(x, "states") and object_states.Frozen in x.states:
                    x.states[object_states.Frozen].set_value(True)

            def _propagate_frozen_from_container(container: StatefulObject):
                visited = set()
                queue = [container]
                scene_objects = []
                scene_objects = list(container.scene.objects)
                while queue:
                    cur = queue.pop()
                    if cur.name in visited:
                        continue
                    visited.add(cur.name)
                    _try_set_frozen(cur)

                    for other in scene_objects:
                        if not hasattr(other, "states"):
                            continue
                        is_on_top = False
                        if object_states.OnTop in other.states:
                            try:
                                is_on_top = other.states[object_states.OnTop].get_value(
                                    cur
                                )
                            except Exception:
                                is_on_top = False
                        is_inside = False
                        if object_states.Inside in other.states:
                            try:
                                is_inside = other.states[
                                    object_states.Inside
                                ].get_value(cur)
                            except Exception:
                                is_inside = False
                        if is_on_top or is_inside:
                            _try_set_frozen(other)
                            queue.append(other)

            _propagate_frozen_from_container(obj_in_hand)

    def _place_on_top(self, target_obj: StatefulObject, **kwargs):
        yield from self._place_with_predicate(target_obj, object_states.OnTop, **kwargs)

    def _place_inside(self, target_obj: StatefulObject, **kwargs):
        yield from self._place_with_predicate(
            target_obj, object_states.Inside, **kwargs
        )

    def _soak(self, obj):
        grasped = self.env_wrapper.grasped_obj
        obj_in_hand = grasped[0] if grasped else None
        if obj_in_hand is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The agent is not holding an object.",
            )
        if (
            object_states.ParticleSource not in obj.states
            and object_states.Contains not in obj.states
        ):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The selected tool is not suitable for this action.",
                {"target object": obj.name},
            )
        scene = getattr(obj, "scene", None)
        if scene is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object has no associated scene.",
                {"target object": obj.name},
            )
        try:
            producing = [scene.get_system("water", force_init=True)]
        except (AssertionError, KeyError, ValueError, AttributeError) as e:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                f"Particle system 'water' is not available on this scene: {e}",
                {"target object": obj.name},
            ) from e
        if (
            object_states.Saturated not in obj_in_hand.states
            or object_states.ParticleRemover not in obj_in_hand.states
            or ModifiedParticles not in obj_in_hand.states
        ):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The held object cannot be soaked.",
                {"object in hand": obj_in_hand.name},
            )
        supported = get_supported_systems(
            obj_in_hand,
            producing,
            object_states.ParticleRemover,
        )
        if not supported:
            remover = obj_in_hand.states[object_states.ParticleRemover]
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The held object and selected tool are not compatible for this action.",
                {
                    "target object": obj.name,
                    "cleaning tool": obj_in_hand.name,
                    "particles the target object is producing": sorted(
                        x.name for x in producing
                    ),
                    "particles the grasped object can remove": sorted(
                        remover.conditions.keys()
                    ),
                },
            )
        for system in supported:
            if (
                ModifiedParticles in obj_in_hand.states
                and object_states.Saturated in obj_in_hand.states
            ):
                limit = obj_in_hand.states[object_states.Saturated].get_limit(system)
                obj_in_hand.states[ModifiedParticles].set_value(system, limit)
                obj_in_hand.states[object_states.Saturated].clear_cache()
        yield from self._settle_robot()

    def _clean_obj(self, target_obj):
        obj_in_hand = self.env_wrapper.grasped_obj
        if len(obj_in_hand) > 0:
            obj_in_hand = obj_in_hand[0]
        else:
            obj_in_hand = None
        if obj_in_hand is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The agent is not holding an object.",
            )
        if object_states.Covered not in target_obj.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is not coverable by any particles, so there is no need to wipe it.",
                {"target object": target_obj.name},
            )
        covering_systems = {
            ps
            for ps in target_obj.scene.system_registry.objects
            if target_obj.states[object_states.Covered].get_value(ps)
        }
        if not covering_systems:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object is not covered by any particles.",
                {"target object": target_obj.name},
            )
        # Check that the current object can remove those particles
        if object_states.ParticleRemover not in obj_in_hand.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The held object is not a valid cleaning tool.",
                {"tool": obj_in_hand.name},
            )
        supported_systems = {
            x
            for x in covering_systems
            if obj_in_hand.states[object_states.ParticleRemover].supports_system(x.name)
        }
        if not supported_systems:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The held object and target are not compatible for this action.",
                {
                    "target object": target_obj.name,
                    "cleaning tool": obj_in_hand.name,
                    "particles the target object is covered by": sorted(
                        x.name for x in covering_systems
                    ),
                    "particles the grasped object can remove": sorted(
                        [
                            x
                            for x in obj_in_hand.states[
                                object_states.ParticleRemover
                            ].conditions.keys()
                        ]
                    ),
                },
            )
        for system in covering_systems:
            target_obj.states[object_states.Covered].set_value(system, False)
        yield from self._settle_robot()

    def _cut(self, obj):
        # Check that the currently held object is a slicer.
        obj_in_hand = self.env_wrapper.grasped_obj
        if len(obj_in_hand) > 0:
            obj_in_hand = obj_in_hand[0]
        else:
            obj_in_hand = None
        if obj_in_hand is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The agent is not holding a valid slicing tool.",
            )
        if "slicer" not in obj_in_hand._abilities:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The agent is not holding a valid slicing tool.",
                {"object in hand": obj_in_hand.name},
            )
        # Check that the target object is sliceable
        if "sliceable" not in obj._abilities and "diceable" not in obj._abilities:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target object cannot be sliced.",
                {"target object": obj.name},
            )
        # Get close

        added_obj_attrs = []
        removed_objs = []
        (slicing_rule,) = [
            rule
            for rule in obj.scene.transition_rule_api.active_rules
            if isinstance(rule, SlicingRule)
        ]
        output = slicing_rule.transition({"sliceable": [obj]})
        added_obj_attrs += output.add
        removed_objs += output.remove
        obj.scene.transition_rule_api.execute_transition(
            added_obj_attrs=added_obj_attrs, removed_objs=removed_objs
        )
        for obj_attr in added_obj_attrs:
            self.env_wrapper.task_objs.append(obj_attr.obj)
            self.env_wrapper.task_obj_to_name[obj_attr.obj] = obj_attr.obj.name
        yield from self._settle_robot()

    def _navigate_to_pose(self, pose_2d):

        robot_pose = self._get_robot_pose_from_2d_pose(pose_2d)
        pos, _ = robot_pose

        if isinstance(pos, th.Tensor):
            pos = pos.cpu().numpy()
        elif not isinstance(pos, np.ndarray):
            pos = np.array(pos)

        current_pos, current_orn = self.robot.get_position_orientation()
        if isinstance(current_pos, th.Tensor):
            current_pos = current_pos.cpu().numpy()
        elif not isinstance(current_pos, np.ndarray):
            current_pos = np.array(current_pos)

        pos[2] = current_pos[2]
        grasped_obj = self.env_wrapper.grasped_obj
        task_objs = self.env_wrapper.task_objs

        if isinstance(current_orn, th.Tensor):
            current_orn = current_orn.cpu().numpy()
        elif not isinstance(current_orn, np.ndarray):
            current_orn = np.array(current_orn)

        if np.any(np.isnan(current_orn)) or np.any(np.isinf(current_orn)):
            current_orn = np.array([0, 0, 0, 1.0])

        ret, msg = move(
            self.robot, None, pos, grasped_obj, task_objs, orientation=current_orn
        )
        if not ret:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                f"Failed to navigate to pose: {msg}",
            )

        yield from self._settle_robot()

    def _settle_robot(self):

        no_op = {self.robot.name: np.zeros(self.robot.action_dim)}

        for _ in range(5):
            self.env_wrapper.env.step(no_op)

        for _ in range(m.MAX_STEPS_FOR_SETTLING):
            if th.norm(self.robot.get_linear_velocity()) < 0.01:
                break
            self.env_wrapper.env.step(no_op)

        yield None

    def _navigate_to_obj(self, obj, eef_pose=None, skip_obstacle_update=False):
        is_pose_2d = False
        if isinstance(obj, (list, tuple)) and len(obj) in [2, 3]:
            try:
                float(obj[0])
                float(obj[1])
                if len(obj) == 3:
                    float(obj[2])
                is_pose_2d = True
            except (ValueError, TypeError, IndexError):
                pass
        if is_pose_2d:
            yield from self._navigate_to_pose(obj)
        else:
            pos_3d = self._sample_nav_pose_near_object(obj)
            yield from self._navigate_to_pose(pos_3d)

    def _run_fill_system_on_target(
        self,
        target_obj: StatefulObject,
        system: BaseSystem,
        *,
        log_prefix: str = "[FILL]",
    ):
        target_obj.states[object_states.Contains].clear_cache()
        target_obj.states[ContainedParticles].clear_cache()
        initial_n_particles = system.n_particles
        initial_contained = (
            target_obj.states[object_states.ContainedParticles]
            .get_value(system)
            .n_in_volume
        )
        target_obj.states[object_states.Filled].set_value(system, True)
        target_obj.states[object_states.Contains].clear_cache()
        target_obj.states[ContainedParticles].clear_cache()
        after_set_n_particles = system.n_particles
        after_set_contained = (
            target_obj.states[object_states.ContainedParticles]
            .get_value(system)
            .n_in_volume
        )
        if after_set_n_particles == initial_n_particles:
            contained_particles_state = target_obj.states[
                object_states.ContainedParticles
            ]
            try:
                system.generate_particles_from_link(
                    obj=target_obj,
                    link=contained_particles_state.link,
                    check_contact=False,
                    max_samples=(
                        filled_m.N_MAX_MACRO_PARTICLE_SAMPLES
                        if isinstance(system, MacroParticleSystem)
                        else filled_m.N_MAX_MICRO_PARTICLE_SAMPLES
                    ),
                )
            except Exception:
                pass
        max_wait_steps = 100
        link_volume = target_obj.states[object_states.ContainedParticles].link.volume
        particle_volume = (
            (system.particle_radius * 2) ** 3 if system.n_particles > 0 else 0
        )
        for step in range(max_wait_steps):
            og.sim.step_physics()
            target_obj.states[object_states.Contains].clear_cache()
            target_obj.states[ContainedParticles].clear_cache()
            n_particles = system.n_particles
            contained_data = target_obj.states[
                object_states.ContainedParticles
            ].get_value(system)
            n_in_volume = contained_data.n_in_volume
            if target_obj.states[object_states.Filled].get_value(system):
                break
        target_obj.states[object_states.Contains].clear_cache()
        target_obj.states[ContainedParticles].clear_cache()
        filled_status = target_obj.states[object_states.Filled].get_value(system)
        final_n_particles = system.n_particles
        final_contained = (
            target_obj.states[object_states.ContainedParticles]
            .get_value(system)
            .n_in_volume
        )
        if filled_status:
            obj_id = target_obj.name
            if obj_id not in self._filled_containers:
                self._filled_containers[obj_id] = []
            if system not in self._filled_containers[obj_id]:
                self._filled_containers[obj_id].append(system)
        yield from self._settle_robot()

    def _fill_with(self, fluid_source: StatefulObject):

        target_obj = self.env_wrapper.grasped_obj
        if len(target_obj) > 0:
            target_obj = target_obj[0]
        else:
            target_obj = None
        if target_obj is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The agent is not holding an object.",
            )
        contained_systems = get_contained_systems(target_obj)
        if contained_systems is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The held object cannot be filled with the selected tool.",
                {"target object": target_obj.name},
            )

        check_open_before_grasp(target_obj, self.env_wrapper.env)

        if "sink" not in fluid_source.name:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The held object cannot be filled with the selected tool.",
            )

        scene = getattr(target_obj, "scene", None)
        produced_systems = [scene.get_system("water", force_init=True)]
        if produced_systems is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                f"Default contained system 'water' is not available on this scene",
                {"target object": target_obj.name},
            )

        for system in produced_systems:
            yield from self._run_fill_system_on_target(
                target_obj, system, log_prefix="[FILL]"
            )
        target_obj.states[object_states.Contains].clear_cache()
        target_obj.states[ContainedParticles].clear_cache()
        for system in produced_systems:
            if not target_obj.states[object_states.Filled].get_value(system):
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                    "The held object was not filled successfully.",
                    {
                        "target object": target_obj.name,
                        "system": getattr(system, "name", None),
                    },
                )

    def _refill_container(self, obj: StatefulObject):
        obj_id = obj.name
        if obj_id not in self._filled_containers:
            return
        systems = self._filled_containers[obj_id]
        if not systems:
            return
        for system in systems:
            try:
                obj.states[object_states.Contains].clear_cache()
                obj.states[ContainedParticles].clear_cache()

                if obj.states[object_states.Filled].get_value(system):
                    continue

                initial_n_particles = system.n_particles
                initial_contained = (
                    obj.states[object_states.ContainedParticles]
                    .get_value(system)
                    .n_in_volume
                )

                obj.states[object_states.Filled].set_value(system, True)

                obj.states[object_states.Contains].clear_cache()
                obj.states[ContainedParticles].clear_cache()
                after_set_n_particles = system.n_particles
                after_set_contained = (
                    obj.states[object_states.ContainedParticles]
                    .get_value(system)
                    .n_in_volume
                )

                if after_set_n_particles == initial_n_particles:

                    contained_particles_state = obj.states[
                        object_states.ContainedParticles
                    ]
                    try:
                        system.generate_particles_from_link(
                            obj=obj,
                            link=contained_particles_state.link,
                            check_contact=False,
                            max_samples=(
                                filled_m.N_MAX_MACRO_PARTICLE_SAMPLES
                                if isinstance(system, MacroParticleSystem)
                                else filled_m.N_MAX_MICRO_PARTICLE_SAMPLES
                            ),
                        )
                    except Exception:
                        pass

                max_wait_steps = 50
                link_volume = obj.states[object_states.ContainedParticles].link.volume
                particle_volume = (
                    (system.particle_radius * 2) ** 3 if system.n_particles > 0 else 0
                )
                for step in range(max_wait_steps):
                    og.sim.step_physics()
                    obj.states[object_states.Contains].clear_cache()
                    obj.states[ContainedParticles].clear_cache()
                    n_particles = system.n_particles
                    contained_data = obj.states[
                        object_states.ContainedParticles
                    ].get_value(system)
                    n_in_volume = contained_data.n_in_volume
                    if obj.states[object_states.Filled].get_value(system):
                        break

                obj.states[object_states.Contains].clear_cache()
                obj.states[ContainedParticles].clear_cache()
                filled_status = obj.states[object_states.Filled].get_value(system)
                if not filled_status:
                    filled_status = obj.states[object_states.Filled].set_value(
                        system, True
                    )
                final_n_particles = system.n_particles
                final_contained = (
                    obj.states[object_states.ContainedParticles]
                    .get_value(system)
                    .n_in_volume
                )
            except Exception:
                pass

    def _pour_into(self, target_obj: StatefulObject):
        obj_in_hand = self.env_wrapper.grasped_obj
        if len(obj_in_hand) > 0:
            obj_in_hand = obj_in_hand[0]
        else:
            obj_in_hand = None
        if obj_in_hand is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The agent is not holding an object.",
            )
        contained_systems = get_contained_systems(target_obj)
        if contained_systems is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The target is not a valid receptacle.",
                {"target object": target_obj.name},
            )

        check_open_before_grasp(target_obj, self.env_wrapper.env)
        check_open_before_grasp(obj_in_hand, self.env_wrapper.env)

        obj_in_hand.states[object_states.Contains].clear_cache()
        obj_in_hand.states[ContainedParticles].clear_cache()

        for _ in range(5):
            og.sim.step_physics()

        obj_in_hand.states[object_states.Contains].clear_cache()
        obj_in_hand.states[ContainedParticles].clear_cache()
        contained_systems = get_contained_systems(obj_in_hand)
        if contained_systems is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The held object is not a liquid container.",
                {"fluid container": obj_in_hand.name},
            )
        if not contained_systems:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The held container is empty.",
                {"fluid container": obj_in_hand.name},
            )

        for system in contained_systems:
            target_obj.states[object_states.Contains].clear_cache()
            target_obj.states[ContainedParticles].clear_cache()
            initial_n_particles = system.n_particles
            initial_contained = (
                target_obj.states[object_states.ContainedParticles]
                .get_value(system)
                .n_in_volume
            )

            target_obj.states[object_states.Filled].set_value(system, True)
            target_obj.states[object_states.Contains].clear_cache()
            target_obj.states[ContainedParticles].clear_cache()
            after_set_n_particles = system.n_particles
            after_set_contained = (
                target_obj.states[object_states.ContainedParticles]
                .get_value(system)
                .n_in_volume
            )

            if after_set_n_particles == initial_n_particles:

                contained_particles_state = target_obj.states[
                    object_states.ContainedParticles
                ]
                try:
                    system.generate_particles_from_link(
                        obj=target_obj,
                        link=contained_particles_state.link,
                        check_contact=False,
                        max_samples=(
                            filled_m.N_MAX_MACRO_PARTICLE_SAMPLES
                            if isinstance(system, MacroParticleSystem)
                            else filled_m.N_MAX_MICRO_PARTICLE_SAMPLES
                        ),
                    )
                except Exception:
                    pass
            max_wait_steps = 100
            link_volume = target_obj.states[
                object_states.ContainedParticles
            ].link.volume
            particle_volume = (
                (system.particle_radius * 2) ** 3 if system.n_particles > 0 else 0
            )
            for step in range(max_wait_steps):
                og.sim.step_physics()
                target_obj.states[object_states.Contains].clear_cache()
                target_obj.states[ContainedParticles].clear_cache()
                n_particles = system.n_particles
                contained_data = target_obj.states[
                    object_states.ContainedParticles
                ].get_value(system)
                n_in_volume = contained_data.n_in_volume
                if target_obj.states[object_states.Filled].get_value(system):
                    break
            target_obj.states[object_states.Contains].clear_cache()
            target_obj.states[ContainedParticles].clear_cache()
            filled_status = target_obj.states[object_states.Filled].get_value(system)
            final_n_particles = system.n_particles
            final_contained = (
                target_obj.states[object_states.ContainedParticles]
                .get_value(system)
                .n_in_volume
            )

            if filled_status:
                obj_id = target_obj.name
                if obj_id not in self._filled_containers:
                    self._filled_containers[obj_id] = []
                if system not in self._filled_containers[obj_id]:
                    self._filled_containers[obj_id].append(system)
            yield from self._settle_robot()

    def _spread(self, target_obj: StatefulObject):
        obj_in_hand = self.env_wrapper.grasped_obj
        if len(obj_in_hand) > 0:
            obj_in_hand = obj_in_hand[0]
        else:
            obj_in_hand = None
        if obj_in_hand is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The agent is not holding an object.",
            )
        appliable_systems = get_appliable_systems(obj_in_hand, self.env_wrapper.env)
        if appliable_systems is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The held object is not a valid spreading tool.",
            )
        if not appliable_systems:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The held object cannot spread anything in its current state.",
            )
        systems_to_apply = appliable_systems

        check_open_before_grasp(obj_in_hand, self.env_wrapper.env)
        check_open_before_grasp(target_obj, self.env_wrapper.env)

        for system in systems_to_apply:
            if target_obj.prim_type != PrimType.CLOTH:
                target_obj.states[object_states.Covered].set_value(system, True)
            else:
                if (
                    ModifiedParticles in target_obj.states
                    and object_states.Saturated in target_obj.states
                ):
                    limit = target_obj.states[object_states.Saturated].get_limit(system)
                    target_obj.states[ModifiedParticles].set_value(system, limit)
                    target_obj.states[object_states.Saturated].clear_cache()
            yield from self._settle_robot()

    def _sample_nav_pose_near_object(self, obj, sampling_attempts=200):

        scene = self.robot.scene
        if not hasattr(scene, "trav_map") or scene.trav_map is None:
            return self._find_robot_place_for_object(obj)
        trav_map = scene.trav_map
        obj_pos = obj.get_position()

        if isinstance(obj_pos, th.Tensor):
            obj_pos = obj_pos.cpu().numpy()
        obj_pos_2d = obj_pos[:2]

        obj_room = None
        if hasattr(scene, "_seg_map") and scene._seg_map is not None:
            obj_room = scene._seg_map.get_room_instance_by_point(obj_pos_2d)

        floor = 0
        trav_map_data = th.clone(trav_map.floor_map[floor])
        trav_map_data = trav_map._erode_trav_map(trav_map_data, robot=self.robot)

        dist_lo, dist_hi = 1.2, 1.5
        attempt = 0
        while attempt < sampling_attempts:
            dist = (th.rand(1) * (dist_hi - dist_lo) + dist_lo).item()
            yaw = (th.rand(1) * 2 * math.pi - math.pi).item()

            candidate_x = obj_pos[0] + dist * math.cos(yaw)
            candidate_y = obj_pos[1] + dist * math.sin(yaw)
            candidate_pos_2d = np.array([candidate_x, candidate_y])

            candidate_pos_map = trav_map.world_to_map(candidate_pos_2d)

            if (
                candidate_pos_map[0] < 0
                or candidate_pos_map[0] >= trav_map.map_size
                or candidate_pos_map[1] < 0
                or candidate_pos_map[1] >= trav_map.map_size
            ):
                attempt += 1
                continue

            y_idx = candidate_pos_map[0].item()
            x_idx = candidate_pos_map[1].item()

            if trav_map_data[y_idx, x_idx] != 255:
                attempt += 1
                continue

            if (
                obj_room is not None
                and hasattr(scene, "_seg_map")
                and scene._seg_map is not None
            ):
                candidate_room = scene._seg_map.get_room_instance_by_point(
                    candidate_pos_2d
                )
                if candidate_room != obj_room:
                    attempt += 1
                    continue

            candidate_3d_pose = th.tensor([candidate_x, candidate_y, 0.004])
            return candidate_3d_pose

        return self._find_robot_place_for_object(obj)

    def _is_navigable_position(
        self, pos_2d, trav_map, trav_map_data, scene, obj_room=None
    ):
        candidate_pos_map = trav_map.world_to_map(pos_2d)
        if (
            candidate_pos_map[0] < 0
            or candidate_pos_map[0] >= trav_map.map_size
            or candidate_pos_map[1] < 0
            or candidate_pos_map[1] >= trav_map.map_size
        ):
            return False
        y_idx = candidate_pos_map[0].item()
        x_idx = candidate_pos_map[1].item()
        if trav_map_data[y_idx, x_idx] != 255:
            return False
        if (
            obj_room is not None
            and hasattr(scene, "_seg_map")
            and scene._seg_map is not None
        ):
            candidate_room = scene._seg_map.get_room_instance_by_point(pos_2d)
            if candidate_room != obj_room:
                return False
        return True

    def _find_robot_place_for_object(self, obj):
        obj_pos, obj_ori = obj.get_position_orientation()
        obj_pos = np.array(
            obj_pos.cpu().numpy() if isinstance(obj_pos, th.Tensor) else obj_pos,
            dtype=np.float64,
        )
        obj_ori = np.array(
            obj_ori.cpu().numpy() if isinstance(obj_ori, th.Tensor) else obj_ori
        )
        robot_pos = self.robot.get_position()
        robot_pos = np.array(
            robot_pos.cpu().numpy() if isinstance(robot_pos, th.Tensor) else robot_pos,
            dtype=np.float64,
        )
        vec_to_robot = robot_pos[:2] - obj_pos[:2]
        vec_norm = np.linalg.norm(vec_to_robot)
        if vec_norm > 0.1:
            forward_dir = vec_to_robot / vec_norm
        else:
            quat_array = np.array([obj_ori[3], obj_ori[0], obj_ori[1], obj_ori[2]])
            forward_vec = Quaternion(quat_array).rotate(np.array([1, 0, 0]))
            forward_dir = forward_vec[:2] / (np.linalg.norm(forward_vec[:2]) + 1e-6)
        low, high = obj.states[object_states.AABB].get_value()
        low = low.cpu().numpy() if isinstance(low, th.Tensor) else np.array(low)
        high = high.cpu().numpy() if isinstance(high, th.Tensor) else np.array(high)
        safe_distance = np.max(np.abs(high - low)) * 0.5 + 0.5

        scene = self.robot.scene
        if hasattr(scene, "trav_map") and scene.trav_map is not None:
            trav_map = scene.trav_map
            floor = 0
            trav_map_data = th.clone(trav_map.floor_map[floor])
            trav_map_data = trav_map._erode_trav_map(trav_map_data, robot=self.robot)

            obj_room = None
            if hasattr(scene, "_seg_map") and scene._seg_map is not None:
                obj_room = scene._seg_map.get_room_instance_by_point(obj_pos[:2])
            robot_pos_2d = obj_pos[:2] + forward_dir * safe_distance
            if self._is_navigable_position(
                robot_pos_2d, trav_map, trav_map_data, scene, obj_room
            ):
                return np.array(
                    [robot_pos_2d[0], robot_pos_2d[1], 0.004], dtype=np.float64
                )
            for dist_mult in [0.8, 0.6, 0.4]:
                for angle in range(0, 360, 30):
                    angle_rad = math.radians(angle)
                    test_pos_2d = (
                        obj_pos[:2]
                        + np.array([math.cos(angle_rad), math.sin(angle_rad)])
                        * safe_distance
                        * dist_mult
                    )
                    if self._is_navigable_position(
                        test_pos_2d, trav_map, trav_map_data, scene, obj_room
                    ):
                        return np.array(
                            [test_pos_2d[0], test_pos_2d[1], 0.004], dtype=np.float64
                        )
        robot_pos_2d = obj_pos[:2] + forward_dir * safe_distance
        return np.array([robot_pos_2d[0], robot_pos_2d[1], 0.004], dtype=np.float64)

    def _burn(self, tool_obj):
        if "stove" not in tool_obj.name and "burner" not in tool_obj.name:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The held object cannot be burned with the selected tool.",
            )
        target_obj = self.env_wrapper.grasped_obj
        if len(target_obj) > 0:
            target_obj = target_obj[0]
        else:
            target_obj = None
        if target_obj is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The agent is not holding an object.",
            )
        heating_temperature = -1.0
        if object_states.Burnt in target_obj.states:
            heating_temperature = max(
                heating_temperature,
                target_obj.states[object_states.Burnt].burn_temperature,
            )
        else:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The held object cannot be burned with the selected tool.",
                {"target object": target_obj.name},
            )
        heating_temperature = heating_temperature + 15
        target_obj.states[object_states.Temperature].set_value(heating_temperature)
        target_obj.states[object_states.MaxTemperature].set_value(heating_temperature)
        yield from self._settle_robot()

        if not (
            object_states.Burnt in target_obj.states
            and target_obj.states[object_states.Burnt].get_value()
        ):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "The held object was not burned successfully.",
                {"target object": target_obj.name},
            )

    def _cook(self, tool_obj):
        target_obj = self.env_wrapper.grasped_obj
        if len(target_obj) > 0:
            target_obj = target_obj[0]
        else:
            target_obj = None
        if target_obj is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The agent is not holding an object.",
            )
        cook_heat_source_name = [
            "oven",
            "charcoal_grill",
            "flat_top_grill",
            "stove",
            "burner",
            "smoker",
            "electric_cauldron",
            "microwave",
            "pressure_cooker",
        ]
        bread_heat_source_name = ["toaster", "toaster_oven"]
        heat_source = None
        if "bread" not in target_obj.name and "toast" not in target_obj.name:
            for name in cook_heat_source_name:
                if name in tool_obj.name:
                    heat_source = tool_obj
                    break
        else:
            for name in bread_heat_source_name + cook_heat_source_name:
                if name in tool_obj.name:
                    heat_source = tool_obj
                    break
        if heat_source is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The held object cannot be cooked with the selected tool.",
            )

        heating_temperature = -1.0
        if object_states.Cooked in target_obj.states:
            heating_temperature = max(
                heating_temperature,
                target_obj.states[object_states.Cooked].cook_temperature,
            )
        elif object_states.Heated in target_obj.states:
            heating_temperature = max(
                heating_temperature,
                target_obj.states[object_states.Heated].heat_temperature,
            )

        if heating_temperature < 0:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The held object cannot be cooked with the selected tool.",
                {"target object": target_obj.name},
            )
        heating_temperature = heating_temperature + 15
        target_obj.states[object_states.Temperature].set_value(heating_temperature)
        target_obj.states[object_states.MaxTemperature].set_value(heating_temperature)
        yield from self._settle_robot()

        if not (
            (
                object_states.Cooked in target_obj.states
                and target_obj.states[object_states.Cooked].get_value()
            )
            or (
                object_states.Heated in target_obj.states
                and target_obj.states[object_states.Heated].get_value()
            )
        ):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "The held object was not cooked successfully.",
                {"target object": target_obj.name},
            )

    def _wash(self, tool_obj):
        if (
            not tool_obj.name.split("_")[0] == "washer"
            and not tool_obj.name.split("_")[0] == "dishwasher"
            and "sink" not in tool_obj.name
        ):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The held object cannot be washed with the selected tool.",
            )
        target_obj = self.env_wrapper.grasped_obj
        if len(target_obj) > 0:
            target_obj = target_obj[0]
        else:
            target_obj = None
        if target_obj is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The agent is not holding an object.",
            )
        is_cloth = (
            hasattr(target_obj, "prim_type") and target_obj.prim_type == PrimType.CLOTH
        )
        if is_cloth:
            saturated_systems = []
            for system in target_obj.scene.system_registry.objects:
                if not is_visual_or_physical_particle_system(target_obj.scene, system):
                    continue
                if object_states.Saturated in target_obj.states and target_obj.states[
                    object_states.Saturated
                ].get_value(system):
                    saturated_systems.append(system)
            for system in saturated_systems:
                if (
                    ModifiedParticles in target_obj.states
                    and object_states.Saturated in target_obj.states
                ):
                    target_obj.states[ModifiedParticles].set_value(system, 0)
                    target_obj.states[object_states.Saturated].clear_cache()
        covered_systems = get_covered_systems(target_obj) or []
        for system in covered_systems:
            try:
                if object_states.Covered in target_obj.states:
                    target_obj.states[object_states.Covered].set_value(system, False)
                yield from self._settle_robot()
            except (ValueError, AssertionError, KeyError, RuntimeError) as e:
                continue

    def _cool(self, tool_obj):
        target_obj = self.env_wrapper.grasped_obj
        if len(target_obj) > 0:
            target_obj = target_obj[0]
        else:
            target_obj = None
        if target_obj is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The agent is not holding an object.",
            )
        if (
            "freezer" not in tool_obj.name
            and "fridge" not in tool_obj.name
            and "refrigerator" not in tool_obj.name
        ):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The held object cannot be cooled with the selected tool.",
            )
        cool_temperature = -50.0
        if object_states.Temperature not in target_obj.states:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "The held object cannot be cooled with the selected tool.",
                {"target object": target_obj.name},
            )
        target_obj.states[object_states.Temperature].set_value(cool_temperature)
        yield from self._settle_robot()
