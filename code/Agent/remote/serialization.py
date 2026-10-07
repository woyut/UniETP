from __future__ import annotations

from typing import Any, Dict, List
from enum import Enum
from collections.abc import Mapping
import numpy as np

from scene_graph.unified_scene_graph import UnifiedSceneGraph
from scene_graph.usg_serializer import serialize_usg


NON_TRANSFERABLE_INFO_KEYS = {"sim"}


def _is_ai2thor_object(value: Any) -> bool:
    module = getattr(value.__class__, "__module__", "")
    return module.startswith("ai2thor")


def _to_transport_safe_key(key: Any) -> Any:
    if isinstance(key, (str, int, float, bool)) or key is None:
        return key
    if isinstance(key, np.generic):
        return key.item()
    if isinstance(key, Enum):
        return key.value
    if isinstance(key, tuple):
        return tuple(_to_transport_safe_key(x) for x in key)
    return repr(key)


def _to_transport_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            _to_transport_safe_key(k): _to_transport_safe(v) for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_to_transport_safe(v) for v in value]
    if _is_ai2thor_object(value):
        return repr(value)
    return repr(value)


def sanitize_additional_info(info: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in info.items():
        if key in NON_TRANSFERABLE_INFO_KEYS:
            continue
        if key == "usg":
            if isinstance(value, UnifiedSceneGraph):
                result[key] = _to_transport_safe(serialize_usg(value))
            else:
                result[key] = _to_transport_safe(value)
            continue
        result[key] = _to_transport_safe(value)
    return result


def sanitize_additional_info_list(
    additional_info_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    sanitized = []
    for item in additional_info_list:
        sanitized.append(sanitize_additional_info(item))
    return sanitized
