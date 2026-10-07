import math
import numpy as np
from PIL import Image


def save_fig(pth, image_nparray):
    if image_nparray.shape[-1] == 4:
        img = Image.fromarray(image_nparray, mode="RGBA")
    else:
        img = Image.fromarray(image_nparray, mode="RGB")
    img.save(pth)


class Agent:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.simulator_name = None

    def set_save_dir(self, save_dir: str):
        self.save_dir = save_dir

    def init_episode(self, instruction: str, **kwargs):
        self.instruction = instruction

    def plan(self, obs_list, action_success_list, feedback_list, additional_info_list):
        return []

    def get_stats(self):
        return {}

    def close(self):
        return

    def update_img_path(self, img_path: str, img_h: int, img_w: int):
        self.cur_img_path = img_path
        self.img_h = img_h
        self.img_w = img_w

    def objects_bbox_in_view(self, additional_info: dict[str], id2cls: dict[str, str]):
        results = []

        if self.simulator_name == "THOR":
            instance_bboxs = additional_info.get("instance_bboxs", {}) or {}
            object_id_to_node_id = additional_info["usg_cache"]["object_id_to_node_id"]
            for object_id, bbox in instance_bboxs.items():
                if bbox is None or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                if object_id not in object_id_to_node_id:
                    continue
                mapped_object_id = object_id_to_node_id[object_id]
                results.append(
                    {
                        "object_id": mapped_object_id,
                        "bbox_xyxy": [x1, y1, x2, y2],
                    }
                )
        elif self.simulator_name == "BEHAVIOR":
            _BBOX_DISTANCE_THRESHOLD_M = 3.0
            instance_bboxs = additional_info.get("instance_bboxs", {}) or {}
            usg_cache = additional_info.get("usg_cache") or {}
            object_position = usg_cache.get("object_position") or {}
            robot_position = object_position.get("robot_agent")
            for object_id, bbox in instance_bboxs.items():
                if bbox is None or len(bbox) != 4:
                    continue
                if robot_position is not None and len(robot_position) >= 3:
                    pos = object_position.get(object_id)
                    if pos is None or len(pos) < 3:
                        continue
                    dx = float(pos[0]) - float(robot_position[0])
                    dy = float(pos[1]) - float(robot_position[1])
                    dz = float(pos[2]) - float(robot_position[2])
                    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if dist >= _BBOX_DISTANCE_THRESHOLD_M:
                        continue
                results.append(
                    {
                        "object_id": object_id,
                        "bbox_xyxy": [int(v) for v in bbox],
                    }
                )
        elif self.simulator_name == "Hab":
            seg = additional_info["instance_seg_frame"]
            for _obj_id in np.unique(seg):
                if _obj_id not in additional_info["nearby_object_and_link_ids"]:
                    continue
                if _obj_id in additional_info["object_id_to_class"]:
                    semantic = additional_info["object_id_to_class"][_obj_id]
                    obj_id = f"{semantic}_{_obj_id}"
                elif _obj_id in additional_info["link_object_id_to_class"]:
                    semantic = additional_info["link_object_id_to_class"][_obj_id]

                    obj_id = f"{semantic.split('..')[0]}_{semantic.split('..')[1]}_part_{semantic.split('..')[2]}"
                elif _obj_id in [10000, 0]:
                    continue
                else:
                    assert False, _obj_id

                if "unknown" in obj_id:
                    continue

                ys, xs = np.where(seg == _obj_id)
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())

                results.append(
                    {
                        "object_id": obj_id,
                        "bbox_xyxy": [x1, y1, x2, y2],
                    }
                )
        elif self.simulator_name == "VH":
            vis_objs_colors = additional_info["refined_visible_objects"]
            seg = additional_info["instance_seg_frame"]
            for oid, color in vis_objs_colors.items():
                if oid.split("_")[-1] == "1":
                    continue

                mask = np.all(
                    seg[..., ::-1] == (np.array(color) * 255).astype(np.uint8), axis=-1
                )
                ys, xs = np.where(mask)
                if len(xs) == 0 or len(ys) == 0:
                    assert False, f"object {oid} has no pixel"
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                results.append(
                    {
                        "object_id": f"{oid}",
                        "bbox_xyxy": [x1, y1, x2, y2],
                    }
                )
        else:
            raise NotImplementedError

        return results

    def save_fig(self, pth, image_nparray):
        save_fig(pth, image_nparray)
