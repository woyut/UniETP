from typing import Dict, Any, Optional
import copy
from enum import Enum

from scene_graph.graph_schema.attribute import (
    ObjectAttributes,
    PropertyType,
    StateType,
)


class EntityType(str, Enum):
    HOUSE = "House"
    ROOM = "Room"

    OBJECT = "Object"
    ROBOT = "Robot"
    HUMAN = "Human"


class Entity:
    """A scene-graph node: a house, room, object or agent, with its attributes."""

    def __init__(
        self,
        ID,
        type: EntityType,
        category: str,
        attributes: ObjectAttributes,
        additional_info: Optional[dict] = None,
    ):
        self.ID = ID
        self.type = type
        self.category = category
        self.attributes = attributes
        self.additional_info = {} if additional_info is None else additional_info

    def __deepcopy__(self, memo):
        new_obj = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_obj

        new_obj.ID = self.ID
        new_obj.type = self.type
        new_obj.category = self.category
        new_obj.attributes = copy.deepcopy(self.attributes, memo)
        new_obj.additional_info = copy.deepcopy(self.additional_info, memo)

        return new_obj

    def __copy__(self):
        cls = self.__class__.__new__(self.__class__)
        new_obj = cls
        new_obj.ID = self.ID
        new_obj.type = self.type
        new_obj.category = self.category
        new_obj.attributes = copy.copy(self.attributes)
        new_obj.additional_info = self.additional_info.copy()
        return new_obj

    def __hash__(self):
        return hash(self.ID)

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False

        return self.ID == other.ID

    def __str__(self):
        out = f"{self.__class__.__name__}[name={self.ID}, type={self.type}, category={self.category}]"
        return out

    def __lt__(self, other):
        return self.ID < other.ID

    def set_property(self, property: PropertyType):
        self.attributes.set_property(property)

    def add_state(self, state: StateType, state_params: dict = {}):
        self.attributes.add_state(state, state_params)

    def update_info(self, info: dict):
        for key, value in info.items():
            self.additional_info[key] = value

    def get_info(self, key: str):
        return self.additional_info.get(key)

    def has_property(self, property: PropertyType):
        return self.attributes.has_property(property)

    def has_state(
        self, state: StateType, state_params: Optional[Dict[str, Any]] = None
    ):
        return self.attributes.has_state(state, state_params)
