from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional

from scene_graph.unified_scene_graph import UnifiedSceneGraph

Trace = List[UnifiedSceneGraph]
LogicEnv = Dict[str, Any]


class Var:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"?{self.name}"


def V(name):
    return Var(name)


def serialize_arg(arg):
    if isinstance(arg, Var):
        return f"?{arg.name}"
    if isinstance(arg, (list, tuple)):
        return [serialize_arg(x) for x in arg]
    return arg


def deserialize_arg(arg):
    if isinstance(arg, str) and arg.startswith("?"):
        return Var(arg[1:])
    if isinstance(arg, list):
        return [deserialize_arg(x) for x in arg]
    return arg


class BaseLogicNode(ABC):
    def _resolve(self, args: Any, env: Optional[LogicEnv] = None) -> Any:
        if env is None:
            env = {}

        if isinstance(args, Var):
            assert env is not None
            if args.name not in env:
                raise ValueError(
                    f"Variable '{args.name}' is not bound in environment: {env.keys()}"
                )
            return env[args.name]

        if isinstance(args, list):
            return [self._resolve(x, env) for x in args]
        if isinstance(args, tuple):
            return tuple(self._resolve(x, env) for x in args)

        return args

    @abstractmethod
    def to_dict(self) -> dict:
        pass
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def from_dict(data: dict) -> "BaseLogicNode":
        pass
        raise NotImplementedError


class LogicNode(BaseLogicNode):
    @abstractmethod
    def evaluate(
        self,
        trace: Trace,
        t_idx: int = -1,
        env: Optional[LogicEnv] = None,
        reference_constraint: dict[str, list[str]] = {},
    ) -> float:
        pass

    def __and__(self, other):

        from task.definition.logic import And

        return And(self, other)

    def __or__(self, other):
        from task.definition.logic import Or

        return Or(self, other)

    def __invert__(self):
        from task.definition.logic import Not

        return Not(self)


class ValueNode(BaseLogicNode):
    @abstractmethod
    def compute(self, trace, t_idx, env):
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

    @staticmethod
    @abstractmethod
    def from_dict(data: dict) -> "ValueNode":
        pass
