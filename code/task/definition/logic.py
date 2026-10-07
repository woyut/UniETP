from typing import List, Any, Union, Dict

from scene_graph.unified_scene_graph import UnifiedSceneGraph
from scene_graph.graph_schema import (
    StateType,
    RelationType,
    Relations,
)
from task.definition.core import (
    LogicNode,
    ValueNode,
    Trace,
    LogicEnv,
    serialize_arg,
    deserialize_arg,
)


class And(LogicNode):
    def __init__(self, *children):
        self.children: List[LogicNode] = children

    def evaluate(
        self,
        trace,
        t_idx=-1,
        env: Dict[str, Any] = None,
        reference_constraint: dict[str, list[str]] = {},
    ):

        return min(
            c.evaluate(trace, t_idx, env, reference_constraint) for c in self.children
        )

    def to_dict(self):
        return {"type": "And", "children": [c.to_dict() for c in self.children]}

    @staticmethod
    def from_dict(data):
        from task.definition.registry import logic_from_dict

        return And(*[logic_from_dict(c) for c in data["children"]])


class Or(LogicNode):
    def __init__(self, *children):
        self.children: List[LogicNode] = children

    def evaluate(
        self,
        trace,
        t_idx=-1,
        env: Dict[str, Any] = None,
        reference_constraint: dict[str, list[str]] = {},
    ):
        return max(
            c.evaluate(trace, t_idx, env, reference_constraint) for c in self.children
        )

    def to_dict(self):
        return {"type": "Or", "children": [c.to_dict() for c in self.children]}

    @staticmethod
    def from_dict(data):
        from task.definition.registry import logic_from_dict

        return Or(*[logic_from_dict(c) for c in data["children"]])


class Not(LogicNode):
    def __init__(self, child):
        self.child: LogicNode = child

    def evaluate(
        self,
        trace,
        t_idx=-1,
        env: Dict[str, Any] = None,
        reference_constraint: dict[str, list[str]] = {},
    ):
        return 1.0 - self.child.evaluate(trace, t_idx, env, reference_constraint)

    def to_dict(self):
        return {
            "type": "Not",
            "child": self.child.to_dict(),
        }

    @staticmethod
    def from_dict(data):
        from task.definition.registry import logic_from_dict

        return Not(logic_from_dict(data["child"]))


class Implies(LogicNode):
    def __init__(self, premise: LogicNode, conclusion: LogicNode):
        self.premise = premise
        self.conclusion = conclusion

    def evaluate(
        self,
        trace: Trace,
        t_idx: int = -1,
        env: LogicEnv = None,
        reference_constraint: dict[str, list[str]] = {},
    ) -> float:
        p_val = self.premise.evaluate(trace, t_idx, env, reference_constraint)
        c_val = self.conclusion.evaluate(trace, t_idx, env, reference_constraint)
        # Soft Logic: max(1 - p, c)
        return max(1.0 - p_val, c_val)

    def to_dict(self):
        return {
            "type": "Implies",
            "premise": self.premise.to_dict(),
            "conclusion": self.conclusion.to_dict(),
        }

    @staticmethod
    def from_dict(data):
        from task.definition.registry import logic_from_dict

        return Implies(
            premise=logic_from_dict(data["premise"]),
            conclusion=logic_from_dict(data["conclusion"]),
        )


class ForAll(LogicNode):
    def __init__(self, var_name: str, var_categories: List[str], child: LogicNode):
        self.var_name = var_name
        self.var_categories = var_categories
        self.child = child

    def evaluate(
        self,
        trace: Trace,
        t_idx: int = -1,
        env: Dict[str, Any] = None,
        reference_constraint: dict[str, list[str]] = {},
    ) -> float:
        assert reference_constraint == {}
        if env is None:
            env = {}
        current_usg = trace[t_idx]

        if self.var_categories is None:
            candidates = current_usg.graph.get_all_node_IDs()
        else:
            candidates = [
                x.ID
                for x in current_usg.graph.get_all_nodes_of_categories(
                    self.var_categories
                )
            ]
        if not candidates:
            return 1.0

        results = []
        for obj_id in candidates:
            new_env = env.copy()
            new_env[self.var_name] = obj_id

            res = self.child.evaluate(trace, t_idx, new_env, reference_constraint)

            if res == 0.0:
                return 0.0
        return 1.0

    def to_dict(self):
        return {
            "type": "ForAll",
            "var_name": self.var_name,
            "var_categories": self.var_categories,
            "child": self.child.to_dict(),
        }

    @staticmethod
    def from_dict(data):
        from task.definition.registry import logic_from_dict

        return ForAll(
            var_name=data["var_name"],
            var_categories=data["var_categories"],
            child=logic_from_dict(data["child"]),
        )


class Exists(LogicNode):
    def __init__(self, var_name: str, var_categories: List[str], child: LogicNode):
        self.var_name = var_name
        self.var_categories = var_categories
        self.child = child

    def evaluate(
        self,
        trace: Trace,
        t_idx: int = -1,
        env: Dict[str, Any] = None,
        reference_constraint: dict[str, list[str]] = {},
    ) -> float:
        if env is None:
            env = {}

        if self.var_categories is None:
            assert reference_constraint == {}
            candidates = set()
            for t in range(len(trace)):
                for x in trace[t].graph.get_all_node_IDs():
                    candidates.add(x)
        else:
            candidates = set()
            for t in range(len(trace)):
                for x in trace[t].graph.get_all_nodes_of_categories(
                    self.var_categories
                ):
                    assert len(self.var_categories) == 1, self.var_categories
                    if self.var_categories[0] in reference_constraint:
                        if x.ID not in reference_constraint[self.var_categories[0]]:
                            continue
                    candidates.add(x.ID)
        if not candidates:
            return 0.0

        results = []
        for obj_id in candidates:
            new_env = env.copy()
            new_env[self.var_name] = obj_id

            res = self.child.evaluate(trace, t_idx, new_env, reference_constraint)

            if res == 1.0:
                return 1.0
        return 0.0

    def to_dict(self):
        return {
            "type": "Exists",
            "var_name": self.var_name,
            "var_categories": self.var_categories,
            "child": self.child.to_dict(),
        }

    @staticmethod
    def from_dict(data):
        from task.definition.registry import logic_from_dict

        return Exists(
            var_name=data["var_name"],
            var_categories=data["var_categories"],
            child=logic_from_dict(data["child"]),
        )


class Compare(LogicNode):
    def __init__(self, left: ValueNode, op: str, right: Union[ValueNode, float]):
        self.left = left
        self.op = op  # '==', '>', '<', '!=', 'in'
        self.right = right

    def evaluate(
        self, trace, t_idx, env, reference_constraint: dict[str, list[str]] = {}
    ):
        assert reference_constraint == {}

        v_l = self.left.compute(trace, t_idx, env)

        v_r = (
            self.right.compute(trace, t_idx, env)
            if isinstance(self.right, ValueNode)
            else self.right
        )

        if self.op == "==":
            return float(v_l == v_r)
        if self.op == ">":
            return float(v_l > v_r)
        if self.op == "<":
            return float(v_l < v_r)
        if self.op == "<=":
            return float(v_l <= v_r)
        if self.op == ">=":
            return float(v_l >= v_r)
        assert False, self.op

    def to_dict(self):
        return {
            "type": "Compare",
            "left": self.left.to_dict(),
            "op": self.op,
            "right": (
                self.right.to_dict()
                if isinstance(self.right, ValueNode)
                else self.right
            ),
        }

    @staticmethod
    def from_dict(data):
        from task.definition.registry import (
            value_from_dict,
        )

        left = value_from_dict(data["left"])

        right_raw = data["right"]
        right = (
            value_from_dict(right_raw)
            if isinstance(right_raw, dict) and "type" in right_raw
            else right_raw
        )

        return Compare(left=left, op=data["op"], right=right)


class Predicate(LogicNode):
    def __init__(self, check_fn, args, predicate_type: str, predicate_kwargs: dict):
        self.fn = check_fn
        self.args = args
        self.predicate_type = predicate_type
        self.predicate_kwargs = predicate_kwargs

    def evaluate(
        self,
        trace: Trace,
        t_idx: int = -1,
        env: LogicEnv = None,
        reference_constraint: dict[str, list[str]] = {},
    ) -> float:
        current_usg = trace[t_idx]
        resolved_args = self._resolve(self.args, env)

        if isinstance(resolved_args, list) or isinstance(resolved_args, tuple):
            return float(self.fn(current_usg, *resolved_args))

        return float(self.fn(current_usg, resolved_args))

    def to_dict(self):
        return {
            "type": "Predicate",
            "predicate_type": self.predicate_type,
            "predicate_kwargs": self.predicate_kwargs,
            "args": serialize_arg(self.args),
        }

    @staticmethod
    def from_dict(data):
        ptype = data["predicate_type"]
        kwargs = data["predicate_kwargs"]
        args = deserialize_arg(data["args"])

        if ptype == "category":
            fn = category_predicate(**kwargs)
        elif ptype == "state":
            fn = state_predicate(**kwargs)
        elif ptype == "relation":
            fn = relation_predicate(**kwargs)
        elif ptype == "empty":
            fn = empty_predicate()
        else:
            raise ValueError(f"Unknown predicate_type: {ptype}")

        return fn(*args) if isinstance(args, (list, tuple)) else fn(args)


def category_predicate(category: str):
    def func(usg: UnifiedSceneGraph, obj_id):
        node = usg.graph.get_node_from_ID(obj_id)
        if node is None:
            return 0.0
        return node.category == category

    def wrapper(obj_id):
        return Predicate(
            func,
            obj_id,
            predicate_type="category",
            predicate_kwargs={"category": category},
        )

    return wrapper


def state_predicate(state: str, state_params: Dict[str, Any] = None):
    def func(usg: UnifiedSceneGraph, obj_id):

        node = usg.graph.get_node_from_ID(obj_id)
        if node is None:
            return 0.0
        return node.has_state(StateType(state), state_params)

    def wrapper(obj_id):
        return Predicate(
            func,
            obj_id,
            predicate_type="state",
            predicate_kwargs={
                "state": state,
                "state_params": state_params,
            },
        )

    return wrapper


def relation_predicate(relation: str, relation_details: Dict[str, Any] = None):
    def func(usg: UnifiedSceneGraph, obj1_id, obj2_id):

        if not usg.graph.has_edge(obj1_id, obj2_id, RelationType(relation)):
            return 0.0
        if relation_details is None:
            return 1.0
        edges: Relations = usg.graph.graph[usg.graph.get_node_from_ID(obj1_id)][
            usg.graph.get_node_from_ID(obj2_id)
        ]

        for key in relation_details:
            if not edges.get_details(key) == relation_details[key]:
                return 0.0
        return 1.0

    def wrapper(obj1_id, obj2_id):
        return Predicate(
            func,
            (obj1_id, obj2_id),
            predicate_type="relation",
            predicate_kwargs={
                "relation": relation,
                "relation_details": relation_details,
            },
        )

    return wrapper


def empty_predicate():
    def func(usg: UnifiedSceneGraph, obj_id):
        return usg.graph.get_node_from_ID(obj_id) is not None

    def wrapper(obj_id):
        return Predicate(
            func,
            obj_id,
            predicate_type="empty",
            predicate_kwargs={},
        )

    return wrapper
