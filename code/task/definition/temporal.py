from task.definition.core import LogicNode, Trace, LogicEnv


class TemporalNode(LogicNode):
    pass


class Always(TemporalNode):
    def __init__(self, child: LogicNode):
        self.child = child

    def evaluate(
        self,
        trace: Trace,
        t_idx: int = -1,
        env: LogicEnv = None,
        reference_constraint: dict[str, list[str]] = {},
    ) -> float:

        start_t = t_idx if t_idx >= 0 else len(trace) + t_idx

        for t in range(start_t, len(trace)):
            if self.child.evaluate(trace, t, env, reference_constraint) < 0.5:
                return 0.0
        return 1.0

    def to_dict(self):
        return {
            "type": "Always",
            "child": self.child.to_dict(),
        }

    @staticmethod
    def from_dict(data):
        from task.definition.registry import logic_from_dict

        return Always(logic_from_dict(data["child"]))


class Sometime(TemporalNode):
    def __init__(self, child: LogicNode):
        self.child = child

    def evaluate(
        self,
        trace: Trace,
        t_idx: int = -1,
        env: LogicEnv = None,
        reference_constraint: dict[str, list[str]] = {},
    ) -> float:
        start_t = t_idx if t_idx >= 0 else len(trace) + t_idx

        for t in range(start_t, len(trace)):
            if self.child.evaluate(trace, t, env, reference_constraint) > 0.5:
                return 1.0
        return 0.0

    def to_dict(self):
        return {
            "type": "Sometime",
            "child": self.child.to_dict(),
        }

    @staticmethod
    def from_dict(data):
        from task.definition.registry import logic_from_dict

        return Sometime(logic_from_dict(data["child"]))


class Until(TemporalNode):
    def __init__(self, condition_a: LogicNode, condition_b: LogicNode):
        self.cond_a = condition_a
        self.cond_b = condition_b

    def evaluate(
        self,
        trace: Trace,
        t_idx: int = -1,
        env: LogicEnv = None,
        reference_constraint: dict[str, list[str]] = {},
    ) -> float:
        start_t = t_idx if t_idx >= 0 else len(trace) + t_idx

        for tc in range(start_t, len(trace)):
            if self.cond_b.evaluate(trace, tc, env, reference_constraint) > 0.5:
                all_a = True
                for ta in range(start_t, tc):
                    if self.cond_a.evaluate(trace, ta, env, reference_constraint) < 0.5:
                        all_a = False
                        break

                if all_a:
                    return 1.0

                continue

        return 0.0

    def to_dict(self):
        return {
            "type": "Until",
            "cond_a": self.cond_a.to_dict(),
            "cond_b": self.cond_b.to_dict(),
        }

    @staticmethod
    def from_dict(data):
        from task.definition.registry import logic_from_dict

        return Until(
            condition_a=logic_from_dict(data["cond_a"]),
            condition_b=logic_from_dict(data["cond_b"]),
        )


class Next(TemporalNode):
    def __init__(self, child: LogicNode):
        self.child = child

    def evaluate(
        self,
        trace: Trace,
        t_idx: int = -1,
        env: LogicEnv = None,
        reference_constraint: dict[str, list[str]] = {},
    ) -> float:
        start_t = t_idx if t_idx >= 0 else len(trace) + t_idx
        next_t = start_t + 1

        if next_t >= len(trace):
            return 0.0

        return self.child.evaluate(trace, next_t, env, reference_constraint)

    def to_dict(self):
        return {
            "type": "Next",
            "child": self.child.to_dict(),
        }

    @staticmethod
    def from_dict(data):
        from task.definition.registry import logic_from_dict

        return Next(logic_from_dict(data["child"]))


class SometimeBefore(TemporalNode):
    def __init__(self, predecessor: LogicNode, successor: LogicNode):
        self.pre = predecessor
        self.succ = successor

    def evaluate(
        self,
        trace: Trace,
        t_idx: int = -1,
        env: LogicEnv = None,
        reference_constraint: dict[str, list[str]] = {},
    ) -> float:

        start_t = t_idx if t_idx >= 0 else len(trace) + t_idx

        succ_indices = []
        for t in range(start_t, len(trace)):
            if self.succ.evaluate(trace, t, env, reference_constraint) > 0.5:
                succ_indices.append(t)

        if not succ_indices:
            return 1.0

        for tb in succ_indices:
            pre_found = False
            for ta in range(start_t, tb):
                if self.pre.evaluate(trace, ta, env, reference_constraint) > 0.5:
                    pre_found = True
                    break

            if pre_found:
                return 1.0

        return 0.0

    def to_dict(self):
        return {
            "type": "SometimeBefore",
            "pre": self.pre.to_dict(),
            "succ": self.succ.to_dict(),
        }

    @staticmethod
    def from_dict(data):
        from task.definition.registry import logic_from_dict

        return SometimeBefore(
            predecessor=logic_from_dict(data["pre"]),
            successor=logic_from_dict(data["succ"]),
        )


class AtEnd(TemporalNode):
    """
    Check if the condition holds at the very last step of the trace.
    Standard Goal Condition in Planning.
    """

    def __init__(self, child: LogicNode):
        self.child = child

    def evaluate(
        self,
        trace: Trace,
        t_idx: int = -1,
        env: LogicEnv = None,
        reference_constraint: dict[str, list[str]] = {},
    ) -> float:

        if not trace:
            return 0.0

        return self.child.evaluate(trace, -1, env, reference_constraint)

    def to_dict(self):
        return {
            "type": "AtEnd",
            "child": self.child.to_dict(),
        }

    @staticmethod
    def from_dict(data):
        from task.definition.registry import logic_from_dict

        return AtEnd(logic_from_dict(data["child"]))
