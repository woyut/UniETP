from general_env import Environment
import base64
import cv2
import numpy as np
from typing import Any, Iterable, List, Optional

# Apply the headless Unity launcher compatibility patch before creating a connection.
import VH.unity_launcher_patch  # noqa: F401
from virtualhome.simulation.unity_simulator import comm_unity

# Compatibility aliases required by older VirtualHome releases.
import collections
import collections.abc

collections.Iterable = collections.abc.Iterable


def _decode_virtualhome_image(image_string):
    """Decode VirtualHome image bytes without changing NumPy's public API."""
    image_bytes = base64.b64decode(image_string)
    if image_bytes[1:4] == b"PNG":
        flags = cv2.IMREAD_COLOR
    else:
        flags = cv2.IMREAD_ANYDEPTH + cv2.IMREAD_ANYCOLOR
    return cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), flags)


comm_unity._decode_image = _decode_virtualhome_image

from scene_graph.unified_scene_graph import UnifiedSceneGraph
from scene_graph.graph_schema.entity import EntityType
from task.commonsense_knowledge.vh import BIG_OBJS
from VH.vh_actions import (
    refine_visible_objects,
    configure_vh_image_size,
    configure_vh_camera_idx,
    vh_camera_image,
    vh_instance_colors,
    turnLeft,
    turnRight,
    moveAhead,
    goTo_obj,
    turnTo_obj,
    open_obj,
    close_obj,
    turnOn_obj,
    turnOff_obj,
    pick_obj,
    placeLeftIn_recep,
    placeRightIn_recep,
    placeLeftOn_recep,
    placeRightOn_recep,
    find_objCls,
)

DEFAULT_TIMEOUT = 300


class VH_Environment(Environment):
    def prepare_observation_for_saving(self, observation):
        """Convert VirtualHome's BGR images to RGB before saving."""
        return observation[:, :, ::-1]

    def build_scene_context(self, entities: Iterable[Any]) -> dict[str, Any]:
        entities = tuple(entities)
        context = super().build_scene_context(entities)
        context["usg_big_objects_list"] = [
            entity.ID
            for entity in entities
            if entity.type == EntityType.OBJECT and entity.category in BIG_OBJS
        ]
        context["usg_small_objects_list"] = [
            entity.ID
            for entity in entities
            if entity.type == EntityType.OBJECT and entity.category not in BIG_OBJS
        ]
        return context

    def __init__(
        self,
        comm: comm_unity.UnityCommunication,
        scene_graph,
        sceneID: int,
        max_steps: int,
        max_failed_steps: int,
        initial_position: Optional[List] = None,
        initial_room: Optional[str] = None,
        camera_relative_position: Optional[List[float]] = [0, 1.5, 0],
        camera_relative_rotation: Optional[List[float]] = [0, 0, 0],
        character_name: Optional[str] = "Chars/Male1",
        unity_port: str = "8080",
        x_display: str = "localhost:10.0",
        timeout_wait: int = DEFAULT_TIMEOUT,
        object_randomize: bool = False,
        object_random_seed: int = -1,
        object_prefabs_map: Optional[dict] = None,
        object_exact_position: bool = True,
        time_hours: int = 12,
        time_minutes: int = 0,
        time_seconds: int = 0,
        useDepth: bool = False,
        useSeg: bool = False,
        img_w: int = 640,
        img_h: int = 480,
        FoV: int = 60,
        **kwargs,
    ):
        self.comm = comm
        self.init_params = {
            "scene_graph": scene_graph,
            "scene_ID": sceneID,
            "object_randomize": object_randomize,
            "object_random_seed": object_random_seed,
            "object_prefabs_map": object_prefabs_map,
            "object_exact_position": object_exact_position,
            "time_hours": time_hours,
            "time_minutes": time_minutes,
            "time_seconds": time_seconds,
            "camera_relative_position": camera_relative_position,
            "camera_relative_rotation": camera_relative_rotation,
            "character_name": character_name,
            "initial_position": initial_position,
            "initial_room": initial_room,
        }

        self.camera_relative_position = camera_relative_position
        self.camera_relative_rotation = camera_relative_rotation

        self._init_common(max_steps, max_failed_steps, useSeg)
        self.img_h = img_h
        self.img_w = img_w
        self.FoV = FoV
        configure_vh_image_size(self.comm, self.img_w, self.img_h)

    def init_episode(
        self,
    ):

        self._init_episode_common()

        self.comm.reset(self.init_params["scene_ID"])
        if self.init_params["scene_graph"] is not None:
            ret, msg = self.comm.expand_scene(
                self.init_params["scene_graph"],
                randomize=False,
                transfer_transform=True,
                random_seed=0,
            )

        ret, msg = self.comm.add_character_camera(
            position=self.init_params["camera_relative_position"],
            rotation=self.init_params["camera_relative_rotation"],
            field_view=self.FoV,
            name="new",
        )
        assert ret
        if self.init_params["character_name"] is not None:
            ret = self.comm.add_character(
                character_resource=self.init_params["character_name"],
                position=self.init_params["initial_position"],
            )
            assert ret
        else:
            assert False

        self.comm.set_time(
            hours=self.init_params["time_hours"],
            minutes=self.init_params["time_minutes"],
            seconds=self.init_params["time_seconds"],
        )
        self.comm.activate_physics()

        self.camera_idx = self.comm.camera_count()[1] - 1
        configure_vh_camera_idx(self.comm, self.camera_idx)

        rgb = vh_camera_image(self.comm, mode="normal", camera_idx=self.camera_idx)[1][
            0
        ]
        seg = vh_camera_image(self.comm, mode="seg_inst", camera_idx=self.camera_idx)[
            1
        ][0]
        depth = vh_camera_image(self.comm, mode="depth", camera_idx=self.camera_idx)[1][
            0
        ]
        instance_colors = vh_instance_colors(self.comm)

        additional_info = {
            "depth": depth,
            "instance_seg_frame": seg,
            "instance_colors": instance_colors,
            "refined_visible_objects": self.refine_colors(seg, instance_colors),
        }

        return rgb, additional_info

    def step(self, action_cls: str, action_args: dict, usg: UnifiedSceneGraph):
        self.num_steps += 1
        done = self.num_steps >= self.max_steps

        if action_cls == "":
            assert False

        elif action_cls == "moveAhead":
            action_success, feedback = moveAhead(self.comm)
        elif action_cls == "turnLeft":
            action_success, feedback = turnLeft(self.comm)
        elif action_cls == "turnRight":
            action_success, feedback = turnRight(self.comm)

        elif action_cls == "turnTo_obj":
            action_success, feedback = turnTo_obj(self.comm, **action_args)
        elif action_cls == "goTo_obj":
            action_success, feedback = goTo_obj(self.comm, **action_args)

        elif action_cls == "pick_obj":
            action_success, feedback = pick_obj(self.comm, **action_args)
        elif action_cls == "close_recep":
            action_success, feedback = close_obj(self.comm, **action_args)
        elif action_cls == "open_recep":
            action_success, feedback = open_obj(self.comm, **action_args)
        elif action_cls == "turnOn_obj":
            action_success, feedback = turnOn_obj(self.comm, **action_args)
        elif action_cls == "turnOff_obj":
            action_success, feedback = turnOff_obj(self.comm, **action_args)
        elif action_cls == "placeLeftIn_recep":
            action_success, feedback = placeLeftIn_recep(self.comm, **action_args)
        elif action_cls == "placeLeftOn_recep":
            action_success, feedback = placeLeftOn_recep(self.comm, **action_args)
        elif action_cls == "placeRightIn_recep":
            action_success, feedback = placeRightIn_recep(self.comm, **action_args)
        elif action_cls == "placeRightOn_recep":
            action_success, feedback = placeRightOn_recep(self.comm, **action_args)

        elif action_cls == "placeOn_recep":
            action_success, feedback = placeRightOn_recep(self.comm, **action_args)
            if feedback == "The agent is not holding an object.":
                action_success, feedback = placeLeftOn_recep(self.comm, **action_args)
        elif action_cls == "placeIn_recep":
            action_success, feedback = placeRightIn_recep(self.comm, **action_args)
            if feedback == "The agent is not holding an object.":
                action_success, feedback = placeLeftIn_recep(self.comm, **action_args)

        elif action_cls == "find_objCls":
            action_success, feedback = find_objCls(self.comm, **action_args)

        elif action_cls == "done":
            done = True
            action_success, feedback = True, "done"
        else:
            action_success, feedback = False, f"Invalid action"
            self.num_steps -= 1
            self.failed_steps -= 1

        if isinstance(feedback, dict):
            feedback = feedback["0"]["message"]
        if feedback == "Success":
            feedback = ""

        self.failed_steps += 0 if action_success else 1
        done = done or (self.failed_steps >= self.max_failed_steps)

        rgb = vh_camera_image(self.comm, mode="normal", camera_idx=self.camera_idx)[1][
            0
        ]
        seg = vh_camera_image(self.comm, mode="seg_inst", camera_idx=self.camera_idx)[
            1
        ][0]
        depth = vh_camera_image(self.comm, mode="depth", camera_idx=self.camera_idx)[1][
            0
        ]
        instance_colors = vh_instance_colors(self.comm)

        additional_info = {
            "depth": depth,
            "instance_seg_frame": seg,
            "instance_colors": instance_colors,
            "refined_visible_objects": self.refine_colors(seg, instance_colors),
        }

        return rgb, action_success, feedback, done, additional_info

    def refine_colors(self, seg, instance_colors):
        return refine_visible_objects(self.comm, self.camera_idx, seg, instance_colors)

    def close(
        self,
    ):
        return
