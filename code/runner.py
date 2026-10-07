from __future__ import annotations

import copy
import importlib
import os
import sys
import time
from collections import defaultdict
from typing import Any, Iterable

from Agent.general_agent import Agent
from general_env import Environment
from scene_graph.unified_scene_graph import UnifiedSceneGraph
from scene_graph.usg_builder import create_init_USG, update_USG
from task.definition.registry import logic_from_dict


_BACKEND_SPECS = {
    "THOR": ("THOR.thor_env", "THOR_Environment"),
    "VH": ("VH.vh_env", "VH_Environment"),
    "Hab": ("Hab.hab_env", "Hab_Environment"),
    "BEHAVIOR": ("BEHAVIOR.og_env", "BehaviorEnv"),
}
_BACKEND_CLASSES: dict[str, type[Environment]] = {}
_BACKEND_IMPORT_ERRORS: dict[str, Exception] = {}


def _goal_variable_categories(logic_dict: dict[str, Any]) -> set[str]:
    categories: set[str] = set()
    pending: list[Any] = [logic_dict]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            var_categories = value.get("var_categories")
            if isinstance(var_categories, list):
                categories.update(
                    category
                    for category in var_categories
                    if isinstance(category, str)
                )
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return categories


def _trace_usg_node_ids(
    scene_graphs: Iterable[UnifiedSceneGraph],
) -> tuple[Any, ...]:
    return tuple(
        dict.fromkeys(
            entity.ID
            for usg in scene_graphs
            for entity in usg.graph.graph
        )
    )


def _current_simulator_id_to_usg_node_id(
    usg: UnifiedSceneGraph | None,
    fallback: dict[Any, Any] | None,
) -> dict[Any, Any] | None:
    """Return the latest simulator-to-USG mapping, falling back to initialization."""
    if usg is not None:
        object_id_to_node_id = usg.cache.get("object_id_to_node_id")
        if object_id_to_node_id:
            return dict(object_id_to_node_id)

        node_id_to_object_id = usg.cache.get("node_id_to_object_id")
        if node_id_to_object_id:
            return {
                object_id: node_id
                for node_id, object_id in node_id_to_object_id.items()
            }
    return fallback


def _load_backend(simulator_name: str) -> type[Environment]:
    if simulator_name in _BACKEND_CLASSES:
        return _BACKEND_CLASSES[simulator_name]
    module_name, class_name = _BACKEND_SPECS[simulator_name]
    try:
        backend = getattr(importlib.import_module(module_name), class_name)
    except Exception as exc:
        _BACKEND_IMPORT_ERRORS[simulator_name] = exc
        raise ImportError(
            f"The {simulator_name} backend is unavailable. "
            "Install its optional dependencies."
        ) from exc
    _BACKEND_CLASSES[simulator_name] = backend
    return backend


def create_env(env_metadata: dict[str, Any]) -> Environment:
    simulator_name = env_metadata["simulator_name"]
    if simulator_name not in _BACKEND_SPECS:
        raise ValueError(f"Unknown simulator: {simulator_name}")
    backend = _load_backend(simulator_name)
    env = backend(**env_metadata)
    env.sim_name = simulator_name
    return env


class Runner:
    def __init__(
        self, task: dict[str, Any], agent: Agent, break_when_action_fail: bool = True
    ):
        self.task = task
        self.env = create_env(task["env_metadata"])
        self.agent = agent
        self.break_when_action_fail = break_when_action_fail
        self.usg: UnifiedSceneGraph | None = None
        self.simulator_id_to_usg_node_id: dict[Any, Any] | None = None
        self.episode_times: list[float] = []
        self.step_times: dict[str, list[float]] = defaultdict(list)
        self.step_idx = 0

    def _save_observation(self, observation, additional_info: dict[str, Any]) -> None:
        image = self.env.prepare_observation_for_saving(observation)
        image_height, image_width = image.shape[:2]
        image_path = os.path.join(self.agent.save_dir, f"{self.step_idx}.png")
        self.agent.save_fig(image_path, image)
        self.agent.update_img_path(image_path, image_height, image_width)
        print(f"[Runner] Saved observation: {image_path}")
        self.step_idx += 1

    def _add_scene_context(self, additional_info: dict[str, Any]) -> None:
        additional_info.update(self.env.build_scene_context(self.usg.graph.graph))

    def run(self):
        episode_start = time.time()

        start = time.time()
        self.agent.init_episode(
            self.task["instruction"], **self.task["init_info_for_agent"]
        )
        after_agent_init = time.time()
        observation, additional_info = self.env.init_episode()
        after_env_init = time.time()
        self.usg = create_init_USG(self.task["env_metadata"], self.env)
        after_usg_init = time.time()
        self.step_times["agent_init"].append(after_agent_init - start)
        self.step_times["env_init"].append(after_env_init - after_agent_init)
        self.step_times["usg_init"].append(after_usg_init - after_env_init)

        node_id_to_object_id = self.usg.cache.get("node_id_to_object_id")
        if node_id_to_object_id:
            self.simulator_id_to_usg_node_id = {
                object_id: node_id
                for node_id, object_id in node_id_to_object_id.items()
            }

        scene_graph_history = [copy.deepcopy(self.usg.graph)]
        initial_score = self.evaluate(self.task["goal_info"], scene_graph_history)
        if initial_score == 1.0:
            raise RuntimeError(
                "Invalid benchmark episode: the goal is already satisfied in the "
                "initial state."
            )

        action_successes = [True]
        feedback = [""]
        observations = [observation]
        additional_info_items = [additional_info]
        additional_info_items[-1]["usg_cache"] = self.usg.cache
        done = False
        self.step_idx = 0
        self._save_observation(observation, additional_info)

        while not done:
            plan_start = time.time()
            self._add_scene_context(additional_info_items[-1])
            actions = self.agent.plan(
                observations,
                action_successes,
                feedback,
                additional_info_items,
            )
            self.step_times["agent_plan"].append(time.time() - plan_start)

            observations = []
            action_successes = []
            feedback = []
            additional_info_items = []

            for action_name, action_args in actions:
                step_start = time.time()
                observation, action_success, message, done, additional_info = (
                    self.env.step(action_name, action_args, self.usg)
                )
                self.step_times["env_step"].append(time.time() - step_start)
                observations.append(observation)
                action_successes.append(action_success)
                feedback.append(message)
                additional_info_items.append(additional_info)

                usg_start = time.time()
                self.usg = update_USG(self.env, self.usg)
                scene_graph_history.append(copy.deepcopy(self.usg.graph))
                additional_info_items[-1]["usg_cache"] = self.usg.cache
                self.step_times["usg_update"].append(time.time() - usg_start)
                self._save_observation(observation, additional_info)

                printable_args = {
                    key: value
                    for key, value in action_args.items()
                    if key != "possible_locations"
                }
                print(
                    f"[Runner] step={self.env.num_steps} action={action_name} "
                    f"args={printable_args} success={action_success} feedback={message}"
                )
                sys.stdout.flush()

                if done or self.step_idx > self.env.max_steps * 2:
                    break
                if not action_success and self.break_when_action_fail:
                    break

            if done or self.step_idx > self.env.max_steps * 2:
                break

        evaluation_start = time.time()
        score = self.evaluate(self.task["goal_info"], scene_graph_history)
        self.step_times["evaluate"].append(time.time() - evaluation_start)
        self.episode_times.append(time.time() - episode_start)
        stats = self.agent.get_stats() if hasattr(self.agent, "get_stats") else {}
        return score, self.env.num_steps, stats

    def evaluate(self, goal_info: dict[str, Any], scene_graph_history) -> float | None:
        if "logic_dict" not in goal_info:
            return None

        evaluation_logic = logic_from_dict(goal_info["logic_dict"])
        scene_graphs = [
            UnifiedSceneGraph(self.task["env_metadata"], graph)
            for graph in scene_graph_history
        ]

        simulator_id_to_usg_node_id = _current_simulator_id_to_usg_node_id(
            self.usg, self.simulator_id_to_usg_node_id
        )
        current_object_ids = (
            simulator_id_to_usg_node_id
            if simulator_id_to_usg_node_id is not None
            else _trace_usg_node_ids(scene_graphs)
        )
        object_reference = self.env.expand_dynamic_object_references(
            self.task["env_metadata"].get("object_reference", {}),
            current_object_ids,
            _goal_variable_categories(goal_info["logic_dict"]),
        )
        if simulator_id_to_usg_node_id is not None:
            mapped_reference = {}
            for key, object_ids in object_reference.items():
                missing_ids = [
                    object_id
                    for object_id in object_ids
                    if object_id not in simulator_id_to_usg_node_id
                ]
                if missing_ids:
                    raise ValueError(
                        f"Object reference {key!r} contains simulator IDs that are "
                        f"missing from the current USG: {missing_ids!r}"
                    )
                mapped_reference[key] = [
                    simulator_id_to_usg_node_id[object_id]
                    for object_id in object_ids
                ]
            object_reference = mapped_reference
        return evaluation_logic.evaluate(
            scene_graphs,
            t_idx=0,
            env=None,
            reference_constraint=object_reference,
        )

    def show_timing_stats(self):
        print("=== Timing Stats Across Episodes ===")
        if not self.episode_times:
            print("No episodes recorded yet.")
            return

        average_total = sum(self.episode_times) / len(self.episode_times)
        print(f"Average total episode time: {average_total:.4f} s")

        for stage, values in self.step_times.items():
            average_stage = sum(values) / len(values)
            print(f"  {stage}: avg {average_stage:.4f} s over {len(values)} steps")

        return self.step_times, self.episode_times

    def close(self, close_agent: bool = True):
        if close_agent:
            self.agent.close()
        self.env.close()
