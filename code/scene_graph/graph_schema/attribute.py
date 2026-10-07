from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
import copy


class StateType(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    POWER_ON = "power_on"
    POWER_OFF = "power_off"
    CLEAN = "clean"
    DIRTY = "dirty"
    FILLED_WITH_LIQUID = "filled_with_liquid"
    CONTAIN_LIQUID = "contain_liquid"
    NO_LIQUID = "no_liquid"
    BROKEN = "broken"
    NOT_BROKEN = "not_broken"
    USED_UP = "used_up"
    NOT_USED_UP = "not_used_up"

    COOKED = "cooked"
    NOT_COOKED = "not_cooked"
    BURNT = "burnt"
    NOT_BURNT = "not_burnt"
    FROZEN = "frozen"
    UNFROZEN = "unfrozen"
    SATURATED = "saturated"
    NOT_SATURATED = "not_saturated"
    HOT = "hot"
    NORMAL_TEMPERATURE = "normal_temperature"
    COOL = "cool"
    COVERED = "covered"
    NOT_COVERED = "not_covered"


@dataclass(frozen=True)
class State:
    type: StateType
    params: Optional[Dict[str, Any]] = None

    def __eq__(self, other):
        if isinstance(other, StateType):
            return self.type == other

        if isinstance(other, State):
            return self.type == other.type

        return NotImplemented

    def exact_equal(self, other: "State") -> bool:
        return self.type == other.type and self.params == other.params

    def __hash__(self):
        return hash(self.type)


STATE_GROUPS = {
    "open_state": {StateType.OPEN, StateType.CLOSED},
    "power_state": {StateType.POWER_ON, StateType.POWER_OFF},
    "clean_state": {StateType.CLEAN, StateType.DIRTY},
    "liquid_state": {
        StateType.FILLED_WITH_LIQUID,
        StateType.CONTAIN_LIQUID,
        StateType.NO_LIQUID,
    },
    "broken_state": {StateType.BROKEN, StateType.NOT_BROKEN},
    "use_state": {StateType.USED_UP, StateType.NOT_USED_UP},
    "cook_state": {StateType.COOKED, StateType.NOT_COOKED},
    "burn_state": {StateType.BURNT, StateType.NOT_BURNT},
    "frozen_state": {StateType.FROZEN, StateType.UNFROZEN},
    "saturated_state": {StateType.SATURATED, StateType.NOT_SATURATED},
    "heated_state": {StateType.HOT, StateType.NORMAL_TEMPERATURE, StateType.COOL},
    "covered_state": {StateType.COVERED, StateType.NOT_COVERED},
}


class PropertyType(str, Enum):
    OPENABLE = "openable"
    HAS_POWER = "has_power"
    CAN_BE_CLEANED = "can_be_cleaned"
    CAN_BE_FILLED_WITH_LIQUID = "can_be_filled_with_liquid"
    GRASPABLE = "graspable"
    MOVABLE = "movable"
    STATIC = "static"
    IS_RECEPTACLE = "is_receptacle"
    CAN_BE_BROKEN = "can_be_broken"
    CAN_BE_USED_UP = "can_be_used_up"

    CAN_BE_COOKED = "can_be_cooked"
    CAN_BE_BURNT = "can_be_burnt"
    CAN_BE_FROZEN = "can_be_frozen"
    CAN_BE_STATURATED = "can_be_staturated"
    CAN_BE_SLICED = "can_be_sliced"
    CAN_BE_HEATED = "can_be_heated"
    CAN_BE_COVERED = "can_be_covered"


STATE_REQUIREMENTS = {
    StateType.OPEN: {PropertyType.OPENABLE},
    StateType.CLOSED: {PropertyType.OPENABLE},
    StateType.POWER_ON: {PropertyType.HAS_POWER},
    StateType.POWER_OFF: {PropertyType.HAS_POWER},
    StateType.CLEAN: {PropertyType.CAN_BE_CLEANED},
    StateType.DIRTY: {PropertyType.CAN_BE_CLEANED},
    StateType.FILLED_WITH_LIQUID: {PropertyType.CAN_BE_FILLED_WITH_LIQUID},
    StateType.CONTAIN_LIQUID: {PropertyType.CAN_BE_FILLED_WITH_LIQUID},
    StateType.NO_LIQUID: {PropertyType.CAN_BE_FILLED_WITH_LIQUID},
    StateType.BROKEN: {PropertyType.CAN_BE_BROKEN},
    StateType.NOT_BROKEN: {PropertyType.CAN_BE_BROKEN},
    StateType.USED_UP: {PropertyType.CAN_BE_USED_UP},
    StateType.NOT_USED_UP: {PropertyType.CAN_BE_USED_UP},
    StateType.COOKED: {PropertyType.CAN_BE_COOKED},
    StateType.NOT_COOKED: {PropertyType.CAN_BE_COOKED},
    StateType.BURNT: {PropertyType.CAN_BE_BURNT},
    StateType.NOT_BURNT: {PropertyType.CAN_BE_BURNT},
    StateType.FROZEN: {PropertyType.CAN_BE_FROZEN},
    StateType.UNFROZEN: {PropertyType.CAN_BE_FROZEN},
    StateType.SATURATED: {PropertyType.CAN_BE_STATURATED},
    StateType.NOT_SATURATED: {PropertyType.CAN_BE_STATURATED},
    StateType.HOT: {PropertyType.CAN_BE_HEATED},
    StateType.NORMAL_TEMPERATURE: {PropertyType.CAN_BE_HEATED},
    StateType.COOL: {PropertyType.CAN_BE_HEATED},
    StateType.COVERED: {PropertyType.CAN_BE_COVERED},
    StateType.NOT_COVERED: {PropertyType.CAN_BE_COVERED},
}


PROPERTY_DEFAULT_STATE = {
    PropertyType.OPENABLE: StateType.CLOSED,
    PropertyType.HAS_POWER: StateType.POWER_OFF,
    PropertyType.CAN_BE_CLEANED: StateType.CLEAN,
    PropertyType.CAN_BE_FILLED_WITH_LIQUID: StateType.NO_LIQUID,
    PropertyType.CAN_BE_BROKEN: StateType.NOT_BROKEN,
    PropertyType.CAN_BE_USED_UP: StateType.NOT_USED_UP,
    PropertyType.CAN_BE_COOKED: StateType.NOT_COOKED,
    PropertyType.CAN_BE_BURNT: StateType.NOT_BURNT,
    PropertyType.CAN_BE_FROZEN: StateType.UNFROZEN,
    PropertyType.CAN_BE_STATURATED: StateType.NOT_SATURATED,
    PropertyType.CAN_BE_HEATED: StateType.NORMAL_TEMPERATURE,
    PropertyType.CAN_BE_COVERED: StateType.NOT_COVERED,
}


class ObjectAttributes:
    def __init__(
        self,
        properties: Optional[set[PropertyType]] = None,
        states: Optional[set] = None,
    ):
        if properties is None:
            properties = set()
        if states is None:
            states = set()
        self.properties = properties

        self.states: Dict[StateType, State] = {}
        if states:
            for s in states:
                if isinstance(s, State):
                    self.states[s.type] = s
                elif isinstance(s, StateType):
                    self.states[s] = State(s, None)
                else:
                    raise TypeError(f"Unsupported state type in init: {type(s)}")

    def __deepcopy__(self, memo):

        new_obj = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_obj
        new_obj.properties = copy.deepcopy(self.properties, memo)
        new_obj.states = copy.deepcopy(self.states, memo)
        return new_obj

    def __copy__(self):
        new_obj = self.__class__(
            properties=self.properties.copy(), states=self.states.copy()
        )
        return new_obj

    def set_property(self, property: PropertyType):
        self.properties.add(property)
        if property in PROPERTY_DEFAULT_STATE:
            default_state = PROPERTY_DEFAULT_STATE[property]
            assert default_state not in self.states
            self.add_state(default_state)

    def add_state(
        self, state: StateType, state_params: Optional[Dict[str, Any]] = None
    ):

        required = STATE_REQUIREMENTS.get(state, set())
        if not required.issubset(self.properties):
            raise ValueError(f"State {state} requires properties {required}")

        for group, states in STATE_GROUPS.items():
            if state in states:
                for s in states:
                    self.states.pop(s, None)

        self.states[state] = State(state, state_params)

    def remove_state(self, state: StateType):
        self.states.pop(state, None)

    def has_state(
        self, state: StateType, state_params: Optional[Dict[str, Any]] = None
    ) -> bool:
        s = self.states.get(state)
        if s is None:
            return False
        if state_params is None:
            return True
        return s.exact_equal(State(state, state_params))

    def has_property(self, property: PropertyType) -> bool:
        return property in self.properties
