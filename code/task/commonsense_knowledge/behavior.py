import ast
import json
from pathlib import Path


with (Path(__file__).parent / "behavior_knowledge.json").open(encoding="utf-8") as file:
    _data = {key: set(value) for key, value in json.load(file).items()}

IN_RECEPTACLE = _data["in_recep"]
ON_RECEPTACLE = _data["on_recep"]

OBJ_SLICING = {ast.literal_eval(item) for item in _data["object_slicing_tool"]}
OBJ_CLEANING = {ast.literal_eval(item) for item in _data["object_cleaning_tool"]}
OBJ_COOKING = {ast.literal_eval(item) for item in _data["object_cooking_tool"]}
OBJ_COOLING = {(obj, "electric_refrigerator.n.01") for obj in _data["freezable"]}
OBJ_HEATING = {
    (obj, heat_source)
    for obj in _data["heatable"]
    for heat_source in _data["heatSource"]
}

_cleanable = _data["stainable"] | _data["dustyable"]
OBJECT_WASHING_TOOL = {(obj, "washer.n.03") for obj in _data["object_washing_tool"]} | {
    (obj, "sink.n.01") for obj in _cleanable
}
