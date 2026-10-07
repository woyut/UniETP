from __future__ import annotations

import importlib
from typing import Any

from general_env import Environment
from scene_graph.unified_scene_graph import UnifiedSceneGraph


_BACKEND_SPECS = {
    "THOR": ("THOR.thor_SG_utils", ("thor_event_to_SG",)),
    "VH": ("VH.vh_SG_utils", ("vh_event_to_SG",)),
    "Hab": ("Hab.hab_SG_utils", ("hab_sim_to_init_SG", "hab_update_USG")),
    "BEHAVIOR": (
        "BEHAVIOR.modified_graph_builder",
        ("UnifiedSceneGraphBuilder", "step"),
    ),
}
_BACKEND_SYMBOLS: dict[str, tuple] = {}


def _load_backend(simulator_name: str) -> tuple:
    if simulator_name in _BACKEND_SYMBOLS:
        return _BACKEND_SYMBOLS[simulator_name]
    module_name, symbol_names = _BACKEND_SPECS[simulator_name]
    try:
        module = importlib.import_module(module_name)
        symbols = tuple(getattr(module, name) for name in symbol_names)
    except Exception as exc:
        raise ImportError(
            f"The {simulator_name} scene-graph backend is unavailable."
        ) from exc
    _BACKEND_SYMBOLS[simulator_name] = symbols
    return symbols


def create_init_USG(
    env_metadata: dict[str, Any], env: Environment
) -> UnifiedSceneGraph:
    simulator_name = env_metadata["simulator_name"]
    if simulator_name == "THOR":
        (thor_event_to_SG,) = _load_backend(simulator_name)
        graph, cache = thor_event_to_SG(env)
    elif simulator_name == "VH":
        (vh_event_to_SG,) = _load_backend(simulator_name)
        graph, cache = vh_event_to_SG(env)
    elif simulator_name == "Hab":
        hab_sim_to_init_SG, _ = _load_backend(simulator_name)
        if env_metadata.get("pre-built_SG_path"):
            raise NotImplementedError("Habitat pre-built scene graphs are not supported.")
        graph, cache = hab_sim_to_init_SG(env)
    elif simulator_name == "BEHAVIOR":
        UnifiedSceneGraphBuilder, _ = _load_backend(simulator_name)
        builder = UnifiedSceneGraphBuilder(full_obs=True)
        return builder.start(
            scene=env.env.scene,
            robot=env.robot,
            env_metadata=env_metadata,
            grasped_obj=[],
        )
    else:
        raise ValueError(f"Unknown simulator: {simulator_name}")
    return UnifiedSceneGraph(env_metadata, graph, cache)


def update_USG(env: Environment, usg: UnifiedSceneGraph) -> UnifiedSceneGraph:
    simulator_name = env.sim_name
    if simulator_name == "THOR":
        (thor_event_to_SG,) = _load_backend(simulator_name)
        graph, cache = thor_event_to_SG(env, usg.cache)
        return UnifiedSceneGraph(usg.env_metadata, graph, cache)
    if simulator_name == "VH":
        (vh_event_to_SG,) = _load_backend(simulator_name)
        graph, cache = vh_event_to_SG(env)
        return UnifiedSceneGraph(usg.env_metadata, graph, cache)
    if simulator_name == "Hab":
        _, hab_update_USG = _load_backend(simulator_name)
        return hab_update_USG(env, usg)
    if simulator_name == "BEHAVIOR":
        _, step = _load_backend(simulator_name)
        return step(usg, env.env.scene, env.robot, env.grasped_obj)
    raise ValueError(f"Unknown simulator: {simulator_name}")
