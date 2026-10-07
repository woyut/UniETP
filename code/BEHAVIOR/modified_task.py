import os
import sys
from typing import Dict, Optional

_current_dir = os.path.dirname(os.path.abspath(__file__))
_omnigibson_path = os.path.join(_current_dir, "BEHAVIOR-1K", "OmniGibson")
if os.path.exists(_omnigibson_path) and _omnigibson_path not in sys.path:
    sys.path.insert(0, _omnigibson_path)
from omnigibson.utils.bddl_utils import OBJECT_TAXONOMY, BDDLEntity

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from task.commonsense_knowledge.behavior import (
    OBJ_SLICING,
    OBJ_CLEANING,
    OBJ_COOKING,
    OBJ_COOLING,
    OBJ_HEATING,
    OBJECT_WASHING_TOOL,
)

TOOL_RELATIONS = {
    "object_slicing_tool": OBJ_SLICING,
    "object_cleaning_tool": OBJ_CLEANING,
    "object_cooking_tool": OBJ_COOKING,
    "object_cooling_tool": OBJ_COOLING,
    "object_heating_tool": OBJ_HEATING,
    "object_washing_tool": OBJECT_WASHING_TOOL,
}


class TemplateTask:
    def __init__(self, scene=None):
        self.scene = scene
        self.object_scope: Dict[str, Optional[BDDLEntity]] = {}
        self._build_object_scope()

    def _build_object_scope(self):
        self.object_scope["agent.n.01_1"] = None
        if self.scene is not None:
            synset_to_objs: Dict[str, list] = {}
            for obj in self.scene.objects:
                if not hasattr(obj, "category"):
                    continue
                synset = OBJECT_TAXONOMY.get_synset_from_category(obj.category)
                if synset:
                    if synset not in synset_to_objs:
                        synset_to_objs[synset] = []
                    synset_to_objs[synset].append(obj)
            for synset, objs in synset_to_objs.items():
                for idx, obj in enumerate(objs, start=1):
                    bddl_inst = f"{synset}_{idx}"
                    self.object_scope[bddl_inst] = BDDLEntity(
                        bddl_inst=bddl_inst, entity=obj
                    )
