from __future__ import annotations

import copy
from typing import Any, Iterable

from scene_graph.graph_schema.attribute import PropertyType
from scene_graph.graph_schema.entity import EntityType


class Environment:
    def __init__(
        self,
    ):
        return

    def _init_common(self, max_steps: int, max_failed_steps: int, useSeg: bool = False):
        self.num_steps = 0
        self.failed_steps = 0
        self.max_steps = max_steps
        self.max_failed_steps = max_failed_steps
        self.useSeg = useSeg
        return

    def _init_episode_common(
        self,
    ):
        self.num_steps = 0
        self.failed_steps = 0
        return

    def step(self, action_cls: str, action_args: dict, usg):
        raise NotImplementedError

    def prepare_observation_for_saving(self, observation):
        """Convert a backend observation to the image convention used by agents."""
        return observation

    def build_scene_context(self, entities: Iterable[Any]) -> dict[str, Any]:
        """Build the object and room lists supplied to the agent."""
        entities = tuple(entities)
        return {
            "usg_id2cls": {entity.ID: entity.category for entity in entities},
            "usg_rooms_list": [
                entity.ID
                for entity in entities
                if entity.type == EntityType.ROOM
                and "unknown" not in entity.category
            ],
            "usg_big_objects_list": [
                entity.ID
                for entity in entities
                if entity.type == EntityType.OBJECT
                and not entity.has_property(PropertyType.GRASPABLE)
                and "unknown" not in entity.category
            ],
            "usg_small_objects_list": [
                entity.ID
                for entity in entities
                if entity.type == EntityType.OBJECT
                and entity.has_property(PropertyType.GRASPABLE)
                and "unknown" not in entity.category
            ],
        }

    def expand_dynamic_object_references(
        self,
        object_reference: dict[str, list[Any]],
        current_object_ids: Iterable[Any],
        goal_categories: set[str] | None = None,
    ) -> dict[str, list[Any]]:
        """Return evaluation references, extended for objects created at runtime.

        Most backends do not need special handling.  Backends whose actions
        create new object IDs can override this hook without adding simulator
        branches to the shared runner.
        """
        return copy.deepcopy(object_reference)

    def close(self):
        raise NotImplementedError
