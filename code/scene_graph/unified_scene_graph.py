from typing import Optional

from scene_graph.graph_schema import Graph


class UnifiedSceneGraph:
    def __init__(
        self,
        env_metadata: Optional[dict[str]],
        graph: Graph,
        cache: Optional[dict] = None,
        full_obs: bool = True,
        debug: bool = True,
    ) -> None:
        self.env_metadata = env_metadata
        self.sim_name = self.env_metadata["simulator_name"]
        self.graph = graph
        self.cache = cache
        self.full_obs = full_obs
        self.debug = debug

    def check_action_validity(self, action_cls: str, action_args: dict):
        return True

    def apply_action(self, action_cls: str, action_args: dict, overwrite: bool = True):
        return
