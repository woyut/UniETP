from Agent.general_agent import Agent
from typing import List, Any
import copy
import json
import os
import numpy as np
from Agent.models.model_registry import get_model


def _sanitize_llm_user_content_for_log(obj: Any, image_path: str) -> Any:
    """Replace inline data: image URLs with a pointer to the saved frame (avoids huge JSON files)."""
    if isinstance(obj, list):
        return [_sanitize_llm_user_content_for_log(x, image_path) for x in obj]
    if isinstance(obj, dict):
        if obj.get("type") == "image_url":
            iu = obj.get("image_url")
            if (
                isinstance(iu, dict)
                and isinstance(iu.get("url"), str)
                and iu["url"].startswith("data:")
            ):
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"<omitted base64; saved frame: {image_path}>"
                    },
                }
        return {
            k: _sanitize_llm_user_content_for_log(v, image_path) for k, v in obj.items()
        }
    return obj


def _llm_io_record(llm_in: Any, llm_out: str, image_path: str) -> dict[str, Any]:
    if isinstance(llm_in, dict) and "user" in llm_in:
        rec = {
            **llm_in,
            "user": _sanitize_llm_user_content_for_log(llm_in["user"], image_path),
        }
    else:
        rec = {"user": _sanitize_llm_user_content_for_log(llm_in, image_path)}

    return {"llm_in": rec, "llm_out": llm_out}


class LLMAgent(Agent):
    def __init__(
        self, save_dir: str, model_type: str, model_name: str, model_config_file: str
    ):
        super().__init__(save_dir)
        self.model = get_model(model_type, model_name, model_config_file)

    def init_episode(
        self,
        instruction: str,
        simulator_name: str,
        nav_action_mode,
        manip_action_mode,
        use_feedback: bool = True,
        use_bbox_coord: bool = True,
        use_action_success: bool = True,
        **kwargs,
    ):
        super().init_episode(instruction)
        os.makedirs(self.save_dir, exist_ok=True)

        self.simulator_name = simulator_name
        self.model.use_feedback = use_feedback
        self.model.use_action_success = use_action_success
        self.use_bbox_coord = use_bbox_coord
        self.nav_action_mode = nav_action_mode
        self.manip_action_mode = manip_action_mode
        if self.nav_action_mode == 1:
            assert self.manip_action_mode == 1, "Mode1 requires mode1 manip"
            self.model.set_difficulty_mode(1)
        elif self.nav_action_mode == 2:
            assert self.manip_action_mode == 2, "Mode2 requires mode2 manip"
            self.model.set_difficulty_mode(2)
        elif self.nav_action_mode == 3:
            assert self.manip_action_mode == 3, "Mode3 requires mode3 manip"
            self.model.set_difficulty_mode(3)
        else:
            raise ValueError(f"Unknown nav action mode: {self.nav_action_mode}")

        self.action_space = {
            "finish_episode": ["done"],
            "nav_basic": [
                "move forward",
                "turn left",
                "turn right",
                "look up",
                "look down",
            ],
        }
        if self.simulator_name in ["THOR", "Hab", "BEHAVIOR"]:
            self.action_space["nav_basic"].extend(
                [
                    "move backward",
                    "move left",
                    "move right",
                ]
            )
        if self.simulator_name == "Hab":
            self.action_space["nav_basic"].extend(
                ["turn base to looking direction", "reset head direction"]
            )

        if self.nav_action_mode == 1:
            self.action_space["nav_object_visual"] = [
                "go to <object_bbox>",
                "turn to point <pixel>",
                "turn to <object_bbox>",
            ]
            self.action_space["nav_object_ID"] = ["go to <room_id>"]
        if self.nav_action_mode >= 2:
            self.action_space["nav_object_ID"] = [
                "go to <object_id>",
                "turn to <object_id>",
            ]

        visual_actions1 = [
            "pick up <object_bbox>",
            "place on <object_bbox>",
            "place in <object_bbox>",
            "place to point <pixel>",
            "open <object_bbox>",
            "close <object_bbox>",
        ]
        optional_visual_actions = {
            "THOR": [
                "pour to <object_bbox>",
                "slice <object_bbox>",
                "turn on <object_bbox>",
                "turn off <object_bbox>",
                "use up <object_bbox>",
            ],
            "VH": [
                "turn on <object_bbox>",
                "turn off <object_bbox>",
            ],
            "Hab": [],
            "BEHAVIOR": [
                "pour to <object_bbox>",
                "slice <object_bbox>",
                "turn on <object_bbox>",
                "turn off <object_bbox>",
                "wipe <object_bbox>",
                "spread to <object_bbox>",
            ],
        }
        visual_actions2 = optional_visual_actions[self.simulator_name]
        if self.simulator_name == "VH":
            visual_actions1.remove("place to point <pixel>")
        if self.manip_action_mode == 1:
            self.action_space["manip_basic_atomic_visual"] = visual_actions1
            self.action_space["manip_optional_atomic_visual"] = visual_actions2
        if self.manip_action_mode >= 2:
            self.action_space["manip_basic_atomic_ID"] = [
                x.replace("<object_bbox>", "<object_id>")
                for x in visual_actions1
                if "<object_bbox>" in x
            ]
            self.action_space["manip_optional_atomic_ID"] = [
                x.replace("<object_bbox>", "<object_id>")
                for x in visual_actions2
                if "<object_bbox>" in x
            ]
        if self.manip_action_mode >= 3:
            if self.simulator_name == "THOR":
                self.action_space["manip_highlevel_ID"] = [
                    "heat the object in hand with <tool_id>",
                    "cool the object in hand with <tool_id>",
                    "wash the object in hand with <tool_id>",
                    "cook the object in hand with <tool_id>",
                    "fill the object in hand with <tool_id>",
                ]
            elif self.simulator_name == "BEHAVIOR":
                self.action_space["manip_highlevel_ID"] = [
                    "cook the object in hand with <tool_id>",
                    "burn the object in hand with <tool_id>",
                    "cool the object in hand with <tool_id>",
                    "wash the object in hand with <tool_id>",
                    "fill the object in hand with <tool_id>",
                    "soak the object in hand with <tool_id>",
                ]

        if self.simulator_name == "THOR":
            self.action_space["manip_basic"] = [
                "move hand left",
                "move hand right",
                "rotate hand",
                "drop",
            ]
        elif self.simulator_name == "VH":
            self.action_space["nav_basic"].remove("look up")
            self.action_space["nav_basic"].remove("look down")
            if "nav_object_visual" in self.action_space:
                self.action_space["nav_object_visual"].remove("turn to point <pixel>")
        elif self.simulator_name in ["Hab", "BEHAVIOR"]:
            self.action_space["manip_basic"] = ["drop"]

        self.action_name_alias = {
            "done": "done",
            "move forward": "moveAhead",
            "move backward": "moveBack",
            "move left": "moveLeft",
            "move right": "moveRight",
            "turn left": "turnLeft",
            "turn right": "turnRight",
            "look up": "lookUp",
            "look down": "lookDown",
            "turn base to looking direction": "turnToLook",
            "reset head direction": "resetHead",
            "go to": "goTo_obj",
            "turn to point": "turnTo_point",
            "turn to": "turnTo_obj",
            "pick up": "pick_obj",
            "place on": "placeTo_recep"
            if self.simulator_name == "THOR"
            else "placeOn_recep",
            "place in": "placeTo_recep"
            if self.simulator_name == "THOR"
            else "placeIn_recep",
            "place to point": "placeTo_point",
            "open": "open_recep",
            "close": "close_recep",
            "pour to": "pourTo_recep",
            "slice": "slice_obj",
            "turn on": "turnOn_obj",
            "turn off": "turnOff_obj",
            "use up": "useUp_obj",
            "wipe": "clean_obj",
            "spread to": "spreadTo_obj",
            "soak the object in hand with": "soakWith_tool",
            "heat the object in hand with": "heatWith_tool",
            "cool the object in hand with": "coolWith_tool",
            "wash the object in hand with": "washWith_tool",
            "cook the object in hand with": "cookWith_tool",
            "burn the object in hand with": "burnWith_tool",
            "fill the object in hand with": "fillWith_tool",
            "move hand forward": "moveHandAhead",
            "move hand backward": "moveHandBack",
            "move hand left": "moveHandLeft",
            "move hand right": "moveHandRight",
            "rotate hand": "rotateHand",
            "drop": "drop",
        }
        if self.simulator_name == "BEHAVIOR":
            self.action_name_alias.pop("use up", None)
        elif self.simulator_name == "Hab":
            self.action_name_alias.pop("turn on", None)
            self.action_name_alias.pop("turn off", None)
        elif self.simulator_name == "VH":
            self.action_name_alias.pop("place to point", None)

        self._llm_call_seq = 0
        self.action_history = ["[episode start]"]
        self.action_success_history = []
        self.feedback_history = []
        self.cur_img_path = None

    def plan(
        self,
        obs_list: List[np.ndarray],
        action_success_list: List[bool],
        feedback_list: List[str],
        additional_info_list: List[dict[str]],
    ):
        for f in feedback_list:
            self.feedback_history.append(f)
        for s in action_success_list:
            self.action_success_history.append(s)
        assert len(self.action_history) == len(self.feedback_history)
        assert len(self.action_history) == len(self.action_success_history)

        id2cls = additional_info_list[-1]["usg_id2cls"]

        scene_rooms_list = additional_info_list[-1]["usg_rooms_list"]
        scene_big_objects_list = additional_info_list[-1]["usg_big_objects_list"]
        scene_small_objects_list = additional_info_list[-1]["usg_small_objects_list"]

        in_view_objects_bbox = self.objects_bbox_in_view(
            additional_info_list[-1], id2cls
        )
        for i in range(len(in_view_objects_bbox)):
            x1, y1, x2, y2 = in_view_objects_bbox[i]["bbox_xyxy"]
            in_view_objects_bbox[i]["bbox_xyxy"] = [
                round(x1 / self.img_w, 3),
                round(y1 / self.img_h, 3),
                round(x2 / self.img_w, 3),
                round(y2 / self.img_h, 3),
            ]
            if not self.use_bbox_coord:
                del in_view_objects_bbox[i]["bbox_xyxy"]

        scene_info = {}
        if self.nav_action_mode >= 1:
            scene_info["scene_rooms_list"] = scene_rooms_list
        if self.nav_action_mode >= 2:
            scene_info["scene_big_objects_list"] = scene_big_objects_list
        if self.nav_action_mode >= 3:
            scene_info["scene_small_objects_list"] = scene_small_objects_list
        if self.manip_action_mode == 1:
            pass
        elif self.manip_action_mode >= 2:
            scene_info["in_view_objects_bbox"] = in_view_objects_bbox
        actions, llm_in, llm_out = self.model.generate_response(
            current_obs_img_path=self.cur_img_path,
            action_history=self.action_history,
            feedback_history=self.feedback_history,
            action_success_history=self.action_success_history,
            scene_info=scene_info,
            action_space=self.action_space,
            instruction=self.instruction,
        )
        llm_path = os.path.join(self.save_dir, f"llm_io_{self._llm_call_seq:05d}.json")
        self._llm_call_seq += 1
        with open(llm_path, "w", encoding="utf-8") as f:
            json.dump(
                _llm_io_record(llm_in, llm_out or "", self.cur_img_path),
                f,
                ensure_ascii=False,
                indent=2,
            )

        if actions is None:
            actions = []
        elif (
            isinstance(actions, dict)
            and "actions" in actions
            and isinstance(actions["actions"], list)
        ):
            actions = actions["actions"]
        elif isinstance(actions, dict):
            actions = [actions]
        elif not isinstance(actions, (list, tuple)):
            actions = [actions]

        actions_to_execute = []
        if len(actions) == 0:
            actions = [["empty_action", {}]]

        for a in actions:
            try:
                if isinstance(a, dict):
                    normalized_action = a
                elif isinstance(a, list) and len(a) >= 1 and isinstance(a[0], str):
                    normalized_action = {
                        "action": a[0],
                        "args": a[1] if len(a) >= 2 and isinstance(a[1], dict) else {},
                    }
                else:
                    raise TypeError(f"unexpected action payload: {a!r}")

                action_type = copy.deepcopy(normalized_action.get("action"))
                action_args = copy.deepcopy(normalized_action.get("args", {}))
                if not isinstance(action_type, str) or not isinstance(
                    action_args, dict
                ):
                    raise TypeError(f"malformed action payload: {normalized_action!r}")

                if (
                    "args" in normalized_action
                    and "object_bbox" in normalized_action["args"]
                ):
                    bb = copy.deepcopy(
                        normalized_action["args"]["object_bbox"]["bbox_xyxy"]
                    )
                    del normalized_action["args"]["object_bbox"]["bbox_xyxy"]
                    if "bbox_xyxy_real" in normalized_action["args"]["object_bbox"]:
                        del normalized_action["args"]["object_bbox"]["bbox_xyxy_real"]
                    normalized_action["args"]["object_bbox"]["bbox"] = bb
                self.action_history.append(str(normalized_action))
                if action_type not in self.action_name_alias:
                    print(f"[Agent] Invalid action type: {action_type}")
                    actions_to_execute.append(["invalid_action", {}])
                    continue

                action_name = self.action_name_alias[action_type]
                action_to_exec = [action_name, {}]
                for k, v in action_args.items():
                    if k in ["object_id", "room_id"]:
                        action_to_exec[1]["object_ref"] = v
                    elif k in ["object_bbox"]:
                        if isinstance(v, dict):
                            if "bbox_xyxy_real" not in v:
                                raise ValueError(f"Invalid object_bbox payload: {v!r}")
                            action_to_exec[1]["object_ref"] = [
                                vv / 1.0 for vv in v["bbox_xyxy_real"]
                            ]
                        elif isinstance(v, (list, tuple)) and len(v) == 4:
                            raise ValueError(
                                f"Unscaled object bbox is not allowed: {v!r}"
                            )
                        else:
                            print(f"[Agent] Invalid object_bbox payload: {v!r}")
                    elif k in ["tool_bbox"]:
                        if isinstance(v, dict):
                            if "bbox_xyxy_real" not in v:
                                raise ValueError(f"Invalid tool_bbox payload: {v!r}")
                            action_to_exec[1]["tool_ref"] = [
                                vv / 1.0 for vv in v["bbox_xyxy_real"]
                            ]
                        elif isinstance(v, (list, tuple)) and len(v) == 4:
                            raise ValueError(
                                f"Unscaled tool bbox is not allowed: {v!r}"
                            )
                        else:
                            print(f"[Agent] Invalid tool_bbox payload: {v!r}")
                    elif k in ["tool_id"]:
                        action_to_exec[1]["tool_ref"] = v
                    elif k == "pixel" or k == "pixel_real":
                        if k == "pixel_real":
                            action_to_exec[1]["target_position_pixel"] = [
                                vv / 1.0 for vv in v
                            ]
                    else:
                        print(f"[Agent] Invalid action argument: {k}:{v}")
                params_valid = self.check_action_args(action_type, action_to_exec[1])
                if not params_valid:
                    print(
                        f"[Agent] Invalid action arguments: {action_type} {action_to_exec[1]}"
                    )
                    actions_to_execute.append(["invalid_action", {}])
                    continue
                actions_to_execute.append(action_to_exec)
            except Exception as e:
                print(f"[Agent] Failed to normalize action {a!r}: {e}")
                actions_to_execute.append(["invalid_action", {}])
        if len(actions_to_execute) != len(actions):
            print(
                f"[Agent] Warning: action normalization size mismatch "
                f"({len(actions_to_execute)} vs {len(actions)})"
            )
        return actions_to_execute

    def get_stats(self):
        stats = {}
        model_stats = getattr(self.model, "stats", None)
        if isinstance(model_stats, dict):
            stats["model_stats"] = model_stats
        return stats

    def check_action_args(self, action_name: str, action_args: dict[str]) -> bool:
        if action_name in [
            "done",
            "move forward",
            "move backward",
            "move left",
            "move right",
            "turn left",
            "turn right",
            "look up",
            "look down",
            "turn base to looking direction",
            "reset head direction",
            "move hand forward",
            "move hand backward",
            "move hand left",
            "move hand right",
            "rotate hand",
            "drop",
        ]:
            if len(action_args) != 0:
                return False
            return True
        elif action_name in [
            "go to",
            "turn to",
            "pick up",
            "place on",
            "place in",
            "open",
            "close",
            "pour to",
            "slice",
            "turn on",
            "turn off",
            "use up",
            "wipe",
            "spread to",
        ]:
            if "object_ref" not in action_args:
                return False
            if len(action_args) != 1:
                return False
            return True
        elif action_name in [
            "turn to point",
            "place to point",
        ]:
            if "target_position_pixel" not in action_args:
                return False
            if len(action_args) != 1:
                return False
            return True
        elif action_name in [
            "soak the object in hand with",
            "heat the object in hand with",
            "cool the object in hand with",
            "wash the object in hand with",
            "cook the object in hand with",
            "burn the object in hand with",
            "fill the object in hand with",
        ]:
            if "tool_ref" not in action_args:
                return False
            if len(action_args) != 1:
                return False
            return True
        else:
            assert False, action_name
            return False
