import numpy as np
from typing import Any, List


from task.definition.core import (
    ValueNode,
    Trace,
    LogicEnv,
    LogicNode,
    serialize_arg,
    deserialize_arg,
)
from scene_graph.unified_scene_graph import UnifiedSceneGraph


class EntityDistance(ValueNode):
    def __init__(self, obj1: Any, obj2: Any):
        self.obj1 = obj1
        self.obj2 = obj2

    def compute(self, trace: Trace, t_idx: int, env: LogicEnv) -> float:
        current_usg: UnifiedSceneGraph = trace[t_idx]

        id1 = self._resolve(self.obj1, env)
        id2 = self._resolve(self.obj2, env)

        pos1 = current_usg.graph.get_node_from_ID(id1).get_info("position")
        pos2 = current_usg.graph.get_node_from_ID(id2).get_info("position")

        return float(np.linalg.norm(np.array(pos1) - np.array(pos2)))

    def to_dict(self):
        return {
            "type": "EntityDistance",
            "obj1": serialize_arg(self.obj1),
            "obj2": serialize_arg(self.obj2),
        }

    @staticmethod
    def from_dict(data):
        return EntityDistance(
            obj1=deserialize_arg(data["obj1"]),
            obj2=deserialize_arg(data["obj2"]),
        )


class CountSatisfying(ValueNode):
    def __init__(
        self, var_name: str, var_categories: List[str], child_condition: LogicNode
    ):
        self.var_name = var_name
        self.var_categories = var_categories
        self.condition = child_condition

    def compute(
        self,
        trace: Trace,
        t_idx: int,
        env: LogicEnv = None,
        reference_constraint: dict[str, list[str]] = {},
    ) -> int:
        if env is None:
            env = {}
        current_usg = trace[t_idx]

        candidates = [
            x.ID
            for x in current_usg.graph.get_all_nodes_of_categories(self.var_categories)
        ]

        count = 0
        for obj_id in candidates:
            new_env = env.copy()
            new_env[self.var_name] = obj_id

            if (
                self.condition.evaluate(trace, t_idx, new_env, reference_constraint)
                > 0.5
            ):
                count += 1

        return count

    def to_dict(self):
        return {
            "type": "CountSatisfying",
            "var_name": self.var_name,
            "var_categories": self.var_categories,
            "condition": self.condition.to_dict(),
        }

    @staticmethod
    def from_dict(data):
        from task.definition.registry import logic_from_dict

        return CountSatisfying(
            var_name=data["var_name"],
            var_categories=data["var_categories"],
            child_condition=logic_from_dict(data["condition"]),
        )
