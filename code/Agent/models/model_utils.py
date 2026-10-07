import base64
import json
import os
import re
from typing import Any, Optional
from PIL import Image


def build_actions_prompt(action_space: dict, pixel_type: str = "normalized_1") -> str:
    """Convert action_space to a table + examples format that's LLM-friendly."""

    # ---- 1. Parse all actions into (action_name, arg_type) pairs ----
    def parse_action(action: str):
        """Returns (action_name, arg_name_or_None)"""
        match = re.match(r"^([a-zA-Z ]+?) ?<(.*?)>$", action.strip())
        if match:
            return match.group(1).strip(), match.group(2).strip()
        else:
            return action.strip(), None

    # ---- 2. Build table ----
    table_lines = []
    table_lines.append(f"| {'action_name':<40} | {'required args':<20} |")
    table_lines.append(f"|{'-' * 42}|{'-' * 22}|")

    if "nav_object_visual" in action_space:
        table_lines.append("| **Object Navigation** | |")
        for raw in action_space["nav_object_visual"]:
            action_name, arg = parse_action(raw)
            arg_display = arg if arg else "(none)"
            table_lines.append(f"| {action_name:<40} | {arg_display:<20} |")
    if "nav_object_ID" in action_space:
        table_lines.append("| **Object / Room Navigation** | |")
        for raw in action_space["nav_object_ID"]:
            action_name, arg = parse_action(raw)
            arg_display = arg if arg else "(none)"
            table_lines.append(f"| {action_name:<40} | {arg_display:<20} |")
    if "nav_basic" in action_space:
        table_lines.append("| **Fine-grained Navigation** | |")
        for raw in action_space["nav_basic"]:
            action_name, arg = parse_action(raw)
            arg_display = arg if arg else "(none)"
            table_lines.append(f"| {action_name:<40} | {arg_display:<20} |")
    table_lines.append("| **Manipulation** | |")
    for key in action_space:
        if key.split("_")[0] == "manip":
            for raw in action_space[key]:
                action_name, arg = parse_action(raw)
                arg_display = arg if arg else "(none)"
                table_lines.append(f"| {action_name:<40} | {arg_display:<20} |")
        else:
            assert key in [
                "nav_object_visual",
                "nav_object_ID",
                "nav_basic",
                "finish_episode",
            ], f"Unknown action space key: {key}"

    table_lines.append("| **Finish Episode** | |")
    table_lines.append(f"| {'done':<40} | {'(none)':<20} |")
    table = "\n".join(table_lines)

    # ---- 3. Build examples ----
    def make_example(action_name: str, args: dict) -> str:
        return json.dumps(
            {"action_name": action_name, "args": args}, ensure_ascii=False
        )

    example_lines = []
    example_lines.append(
        "// move forward:\n"
        + make_example(
            "move forward",
            {
                "object_id": None,
                "room_id": None,
                "tool_id": None,
                "pixel": None,
                "object_bbox": None,
            },
        )
    )
    if "object_id" in table:
        example_lines.append(
            "// go to:\n"
            + make_example(
                "go to",
                {
                    "object_id": "Apple_0",
                    "room_id": None,
                    "tool_id": None,
                    "pixel": None,
                    "object_bbox": None,
                },
            )
        )
    if "room_id" in table:
        example_lines.append(
            "// go to:\n"
            + make_example(
                "go to",
                {
                    "object_id": None,
                    "room_id": "Kitchen_15",
                    "tool_id": None,
                    "pixel": None,
                    "object_bbox": None,
                },
            )
        )
    if "tool_id" in table:
        example_lines.append(
            "// wash the object in hand with:\n"
            + make_example(
                "wash the object in hand with",
                {
                    "object_id": None,
                    "room_id": None,
                    "tool_id": "Faucet_asdhyx",
                    "pixel": None,
                    "object_bbox": None,
                },
            )
        )

    if "object_bbox" in table:
        if pixel_type == "normalized_1":
            example_box = [0.15, 0.20, 0.52, 0.66]
        elif pixel_type == "normalized_1000":
            example_box = [151, 200, 528, 669]
        elif pixel_type == "absolute":
            example_box = [151, 200, 227, 258]
        else:
            raise ValueError(f"Invalid pixel type: {pixel_type}")
        example_lines.append(
            "// pick up:\n"
            + make_example(
                "pick up",
                {
                    "object_id": None,
                    "room_id": None,
                    "tool_id": None,
                    "pixel": None,
                    "object_bbox": {"bbox": example_box, "category": "apple"},
                },
            )
        )
    if "pixel" in table:
        if pixel_type == "normalized_1":
            example_pixel = [0.32, 0.55]
        elif pixel_type == "normalized_1000":
            example_pixel = [320, 550]
        elif pixel_type == "absolute":
            example_pixel = [320, 476]
        else:
            raise ValueError(f"Invalid pixel type: {pixel_type}")
        example_lines.append(
            "// turn to point:\n"
            + make_example(
                "turn to point",
                {
                    "object_id": None,
                    "room_id": None,
                    "tool_id": None,
                    "pixel": example_pixel,
                    "object_bbox": None,
                },
            )
        )
    examples = "\n\n".join(example_lines)

    return f"""\
### Action Table

The `action_name` must be exactly one of the strings in the left column below.


{table}

### Output Examples

{examples}"""


def _load_optional_json_config(path: Optional[str]) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "temperature": 0.2,
        "max_tokens": 2048,
        "timeout": 120.0,
    }
    if not path or not os.path.isfile(path):
        return defaults
    with open(path, encoding="utf-8") as f:
        user = json.load(f)
    if not isinstance(user, dict):
        return defaults
    defaults.update(user)
    return defaults


def _probe_image_size(image_path: str) -> Optional[tuple[int, int]]:
    """Return (width, height) of the observation image, or None."""
    if not image_path or not os.path.isfile(image_path):
        return None
    try:
        with Image.open(image_path) as im:
            w, h = im.size
            return int(w), int(h)
    except OSError:
        return None


def _image_to_data(image_path: str) -> tuple[Optional[str], Optional[str]]:
    if not image_path or not os.path.isfile(image_path):
        return None, None
    ext = os.path.splitext(image_path)[1].lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(ext, "image/png")
    with open(image_path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("ascii")
    return mime, b64
