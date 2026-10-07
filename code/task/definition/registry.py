import sys

sys.path.append(".")
import json

from task.definition.core import LogicNode, Var, ValueNode
from task.definition.logic import (
    And,
    Or,
    Not,
    Implies,
    Compare,
    ForAll,
    Exists,
    Predicate,
)
from task.definition.temporal import (
    Always,
    Sometime,
    Until,
    Next,
    AtEnd,
    SometimeBefore,
)
from task.definition.value import EntityDistance, CountSatisfying

LOGIC_NODE_REGISTRY = {
    "And": And,
    "Or": Or,
    "Not": Not,
    "Implies": Implies,
    "Compare": Compare,
    "ForAll": ForAll,
    "Exists": Exists,
    "Predicate": Predicate,
    "Always": Always,
    "Sometime": Sometime,
    "Until": Until,
    "Next": Next,
    "AtEnd": AtEnd,
    "SometimeBefore": SometimeBefore,
}

VALUE_NODE_REGISTRY = {
    "EntityDistance": EntityDistance,
    "CountSatisfying": CountSatisfying,
}


def logic_from_dict(data: dict):
    cls = LOGIC_NODE_REGISTRY[data["type"]]
    return cls.from_dict(data)


def value_from_dict(data: dict):
    cls = VALUE_NODE_REGISTRY[data["type"]]
    return cls.from_dict(data)


def _fmt_arg(arg):
    if isinstance(arg, Var):
        return f"?{arg.name}"
    if isinstance(arg, list):
        return "[" + ", ".join(_fmt_arg(a) for a in arg) + "]"
    if isinstance(arg, tuple):
        return "(" + ", ".join(_fmt_arg(a) for a in arg) + ")"
    return str(arg)


def pretty_print(node, indent=0) -> str:
    pad = "  " * indent

    # ========= Quantifiers =========
    if isinstance(node, Exists):
        s = f"{pad}Exists {node.var_name} ∈ {node.var_categories}\n"
        s += pretty_print(node.child, indent + 1)
        return s

    if isinstance(node, ForAll):
        s = f"{pad}ForAll {node.var_name} ∈ {node.var_categories}\n"
        s += pretty_print(node.child, indent + 1)
        return s

    # ========= Boolean Logic =========
    if isinstance(node, And):
        s = f"{pad}And\n"
        for c in node.children:
            s += pretty_print(c, indent + 1)
        return s

    if isinstance(node, Or):
        s = f"{pad}Or\n"
        for c in node.children:
            s += pretty_print(c, indent + 1)
        return s

    if isinstance(node, Not):
        s = f"{pad}Not\n"
        s += pretty_print(node.child, indent + 1)
        return s

    if isinstance(node, Implies):
        s = f"{pad}Implies\n"
        s += f"{pad}  Premise:\n"
        s += pretty_print(node.premise, indent + 2)
        s += f"{pad}  Conclusion:\n"
        s += pretty_print(node.conclusion, indent + 2)
        return s

    # ========= Compare =========
    if isinstance(node, Compare):
        s = f"{pad}Compare\n"
        s += f"{pad}  Left:\n"
        s += pretty_print(node.left, indent + 2)

        s += f"{pad}  Op: {node.op}\n"

        s += f"{pad}  Right:\n"
        if isinstance(node.right, ValueNode):
            s += pretty_print(node.right, indent + 2)
        else:
            s += f"{pad}    {node.right}\n"

        return s

    # ========= Predicate =========
    if isinstance(node, Predicate):
        return (
            f"{pad}Predicate[{node.predicate_type}]"
            f"{node.predicate_kwargs} "
            f"args={_fmt_arg(node.args)}\n"
        )

    # ========= Temporal =========
    if isinstance(node, Always):
        return f"{pad}Always\n" + pretty_print(node.child, indent + 1)

    if isinstance(node, Sometime):
        return f"{pad}Sometime\n" + pretty_print(node.child, indent + 1)

    if isinstance(node, Next):
        return f"{pad}Next\n" + pretty_print(node.child, indent + 1)

    if isinstance(node, AtEnd):
        return f"{pad}AtEnd\n" + pretty_print(node.child, indent + 1)

    if isinstance(node, Until):
        s = f"{pad}Until\n"
        s += f"{pad}  Condition A:\n"
        s += pretty_print(node.cond_a, indent + 2)
        s += f"{pad}  Condition B:\n"
        s += pretty_print(node.cond_b, indent + 2)
        return s

    if isinstance(node, SometimeBefore):
        s = f"{pad}SometimeBefore\n"
        s += f"{pad}  Predecessor:\n"
        s += pretty_print(node.pre, indent + 2)
        s += f"{pad}  Successor:\n"
        s += pretty_print(node.succ, indent + 2)
        return s

    # ========= Value Nodes =========
    if isinstance(node, EntityDistance):
        return f"{pad}EntityDistance({_fmt_arg(node.obj1)}, {_fmt_arg(node.obj2)})\n"

    if isinstance(node, CountSatisfying):
        s = f"{pad}CountSatisfying {node.var_name} ∈ {node.var_categories}\n"
        s += f"{pad}  Condition:\n"
        s += pretty_print(node.condition, indent + 2)
        return s

    # ========= Fallback =========
    return f"{pad}{node.__class__.__name__}\n"


def save_logic(logic: LogicNode, path: str):
    with open(path, "w") as f:
        json.dump(logic.to_dict(), f, indent=2)


def load_logic(path: str) -> LogicNode:
    with open(path) as f:
        data = json.load(f)
    return logic_from_dict(data)
