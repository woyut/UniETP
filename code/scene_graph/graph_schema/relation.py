from enum import Enum
import copy
from typing import Optional


class RelationType(str, Enum):
    ON_ROOM_FLOOR = "on_room_floor"
    ON_RECEPTACLE = "on_receptacle"
    INSIDE_HOUSE = "inside_house"
    INSIDE_ROOM = "inside_room"
    INSIDE_RECEPTACLE = "inside_receptacle"
    GRASPING = "grasping"


RELATION_GROUPS = {
    "object-receptacle": {
        RelationType.ON_RECEPTACLE,
        RelationType.INSIDE_RECEPTACLE,
    },
}


class Relations:
    def __init__(self, relations: set[RelationType], details: Optional[dict] = None):
        self.relations = relations
        self.details = details if details is not None else {}

    def __deepcopy__(self, memo):
        new_obj = self.__class__(
            relations=copy.deepcopy(self.relations, memo),
            details=copy.deepcopy(self.details, memo),
        )
        memo[id(self)] = new_obj
        return new_obj

    def __copy__(self):
        new_obj = self.__class__(
            relations=self.relations.copy(), details=self.details.copy()
        )
        return new_obj

    def add(self, rel: RelationType):
        for group, group_rels in RELATION_GROUPS.items():
            if rel in group_rels:
                conflict = self.relations & group_rels
                if conflict:
                    self.relations -= conflict

        self.relations.add(rel)

    def update_details(self, key, value):
        self.details[key] = value

    def get_details(self, key):
        return self.details.get(key)

    def remove(self, rel: RelationType):
        self.relations.discard(rel)

    def has(self, rel: RelationType) -> bool:
        return rel in self.relations

    def empty(self) -> bool:
        return len(self.relations) == 0

    def __str__(self) -> str:
        return ", ".join(sorted(self.relations))


def flip_edge(edge: str) -> str:
    return {
        "on_receptacle": "receptacle_supports",
        "on_room_floor": "room_floor_supports",
        "inside_house": "house_contains",
        "inside_room": "room_contains",
        "inside_receptacle": "receptacle_contains",
        "grasping": "grasped_by",
    }[edge]
