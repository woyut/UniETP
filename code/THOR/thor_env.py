from general_env import Environment
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from typing import List, Optional, Union, Any, Iterable

from scene_graph.unified_scene_graph import UnifiedSceneGraph
from task.commonsense_knowledge.thor import FOR_SLICING
from THOR.thor_actions import (
    moveAhead,
    moveBack,
    moveLeft,
    moveRight,
    turnLeft,
    turnRight,
    lookUp,
    lookDown,
    moveHandAhead,
    moveHandBack,
    moveHandLeft,
    moveHandRight,
    rotateHand,
    turnTo_point,
    turnTo_obj,
    goTo_obj,
    goTo_loc,
    pick_obj,
    placeTo_recep,
    placeTo_point,
    drop,
    throw_force,
    pourTo_recep,
    close_recep,
    open_recep,
    slice_obj,
    turnOn_obj,
    turnOff_obj,
    useUp_obj,
    find_objCls,
    heatWith_tool,
    coolWith_tool,
    washWith_tool,
    cookWith_tool,
    fillWith_tool,
)


class THOR_Environment(Environment):
    def __init__(
        self,
        sceneID,
        max_steps: int,
        max_failed_steps: int,
        proc_data,
        material_randomization_args: dict[str],
        lighting_randomization_args: dict[str],
        placement_randomization_args: dict[str],
        init_steps: List[dict[str]] = [],
        robotType: str = "default",
        visibilityDistance: float = 1.5,
        gridSize: float = 0.25,
        snapToGrid: bool = False,
        rotateStepDegrees: int = 30,
        useDepth: bool = False,
        useSeg: bool = False,
        img_w: int = 500,
        img_h: int = 500,
        FoV: int = 90,
        **kwargs,
    ):

        if isinstance(sceneID, int):
            if sceneID < 10000:
                sceneID = proc_data["train"][sceneID]
            elif sceneID < 11000:
                sceneID = proc_data["val"][sceneID - 10000]
            else:
                sceneID = proc_data["test"][sceneID - 11000]

        self.controller = Controller(
            agentMode=robotType,
            visibilityDistance=visibilityDistance,
            scene=sceneID,
            # step sizes
            gridSize=gridSize,
            snapToGrid=snapToGrid,
            rotateStepDegrees=rotateStepDegrees,
            # image modalities
            renderDepthImage=useDepth,
            renderInstanceSegmentation=True,
            # camera properties
            width=img_w,
            height=img_h,
            fieldOfView=FoV,
            platform=CloudRendering,  # headless running
            server_timeout=300,
        )

        self.material_randomization_args = material_randomization_args
        self.lighting_randomization_args = lighting_randomization_args
        self.placement_randomization_args = placement_randomization_args
        self.init_steps = init_steps
        # Keep the same mutable dictionary that Runner stores in env_metadata so
        # stale IDs can be rebound before Runner builds its reference constraint.
        self.object_reference = kwargs.get("object_reference", {})
        self.object_reference_info = kwargs.get("object_reference_info", {})
        self.bbox_distance_threshold = kwargs.get("bbox_distance_threshold", None)

        if isinstance(sceneID, str):
            self.room_id_to_category = None
            self.room_id_to_floor_polygon = {}
            self.door_to_rooms = None
        else:
            self.room_id_to_category = {}
            self.room_id_to_floor_polygon = {}
            self.door_to_rooms = {}
            for x in sceneID["rooms"]:
                self.room_id_to_category[x["id"]] = x["roomType"]
                self.room_id_to_floor_polygon[x["id"]] = x["floorPolygon"]
            for x in sceneID["doors"]:
                self.door_to_rooms[x["id"]] = [
                    (x["room0"], self.room_id_to_category[x["room0"]]),
                    (x["room1"], self.room_id_to_category[x["room1"]]),
                ]

        self._init_common(max_steps, max_failed_steps, useSeg)

    @staticmethod
    def _init_action_error(phase: str, action: dict[str, Any], event) -> RuntimeError:
        error = event.metadata.get("errorMessage") or "no simulator error message"
        return RuntimeError(
            f"THOR episode initialization failed during {phase}: "
            f"action={action!r}; error={error}"
        )

    def _step_init_checked(self, phase: str, action: dict[str, Any]):
        event = self.controller.step(**action)
        if not event.metadata.get("lastActionSuccess", False):
            raise self._init_action_error(phase, action, event)
        return event

    @staticmethod
    def _id_type(object_id: str) -> str:
        return object_id.split("|", 1)[0]

    def _resolve_init_step_object_ids(self) -> dict[str, str]:
        """Rebind only object IDs whose replacement is unambiguous.

        Exact IDs are kept.  A stale ID is rebound only when exactly one unused
        current object of the same type remains; all ambiguous cases fail early.
        """
        objects = self.controller.last_event.metadata.get("objects", [])
        current_ids = {str(obj["objectId"]) for obj in objects}
        by_type: dict[str, list[str]] = {}
        for obj in objects:
            by_type.setdefault(str(obj.get("objectType")), []).append(str(obj["objectId"]))

        indexed: dict[str, list[dict[str, Any]]] = {}
        for step in self.init_steps:
            object_id = step.get("objectId")
            if isinstance(object_id, str):
                indexed.setdefault(self._id_type(object_id), []).append(step)

        mapping: dict[str, str] = {}
        for object_type, entries in indexed.items():
            expected_ids = list(dict.fromkeys(str(step["objectId"]) for step in entries))
            candidates = by_type.get(object_type, [])
            exact = {x: x for x in expected_ids if x in current_ids}
            missing = [x for x in expected_ids if x not in exact]
            available = [x for x in candidates if x not in exact]
            mapping.update(exact)
            if not missing:
                continue
            if len(missing) == len(available) == 1:
                mapping[missing[0]] = available[0]
                continue
            raise RuntimeError(
                "THOR episode initialization cannot unambiguously rebind stale object IDs: "
                f"type={object_type!r}, missing={missing!r}, "
                f"available={available!r}, candidate_count={len(candidates)}. "
                "Use benchmark assets whose object IDs match the selected scene."
            )

        for step in self.init_steps:
            object_id = step.get("objectId")
            if object_id in mapping:
                step["objectId"] = mapping[object_id]
        return mapping

    def _objects_with_parent_type(self, category: str, parent_type: str) -> list[str]:
        objects = self.controller.last_event.metadata.get("objects", [])
        object_type_by_id = {
            str(obj["objectId"]): str(obj.get("objectType")) for obj in objects
        }
        matches = []
        for obj in objects:
            if obj.get("objectType") != category:
                continue
            parents = obj.get("parentReceptacles") or []
            if any(object_type_by_id.get(str(parent_id)) == parent_type for parent_id in parents):
                matches.append(str(obj["objectId"]))
        return sorted(matches)

    def _resolve_object_references(self, init_mapping: dict[str, str]) -> None:
        """Rebind reference IDs from their declared descriptor, or fail early."""
        objects = self.controller.last_event.metadata.get("objects", [])
        current_ids = {str(obj["objectId"]) for obj in objects}
        by_type: dict[str, list[str]] = {}
        for obj in objects:
            by_type.setdefault(str(obj.get("objectType")), []).append(str(obj["objectId"]))

        for category, old_ids_value in list(self.object_reference.items()):
            old_ids = [str(x) for x in old_ids_value]
            if all(x in current_ids for x in old_ids):
                continue

            info = self.object_reference_info.get(category)
            rebound = None
            if (
                isinstance(info, list)
                and len(info) == 2
                and info[0] == "by_parent"
                and isinstance(info[1], dict)
                and isinstance(info[1].get("parent"), str)
            ):
                matches = self._objects_with_parent_type(category, info[1]["parent"])
                if len(matches) == len(old_ids):
                    rebound = matches

            if rebound is None:
                mapped = [init_mapping.get(x, x) for x in old_ids]
                if all(x in current_ids for x in mapped) and len(set(mapped)) == len(mapped):
                    rebound = mapped

            if rebound is None and len(old_ids) == 1 and len(by_type.get(category, [])) == 1:
                rebound = list(by_type[category])

            if rebound is None:
                missing = [x for x in old_ids if x not in current_ids]
                raise RuntimeError(
                    "THOR episode initialization cannot safely rebind object_reference: "
                    f"category={category!r}, missing={missing!r}, "
                    f"reference_info={info!r}, candidates={by_type.get(category, [])!r}"
                )
            self.object_reference[category][:] = rebound

    def expand_dynamic_object_references(
        self,
        object_reference: dict[str, list[Any]],
        current_object_ids: Iterable[Any],
        goal_categories: set[str] | None = None,
    ) -> dict[str, list[Any]]:
        """Add sliced descendants of referenced whole objects.

        AI2-THOR retains a sliced whole object and prefixes every generated
        slice ID with the whole object's ID.  Slice IDs do not exist in the
        initial USG, so bind them from the latest simulator-to-USG cache.
        """
        expanded = super().expand_dynamic_object_references(
            object_reference, current_object_ids, goal_categories
        )
        simulator_ids = tuple(current_object_ids)
        for whole_category in sorted(FOR_SLICING):
            if whole_category not in object_reference:
                continue
            sliced_category = f"{whole_category}Sliced"
            if goal_categories is not None and sliced_category not in goal_categories:
                continue

            derived_ids = list(expanded.get(sliced_category, []))
            seen_ids = set(derived_ids)
            for whole_id in object_reference[whole_category]:
                descendant_prefix = f"{whole_id}|{sliced_category}_"
                for simulator_id in simulator_ids:
                    if (
                        isinstance(simulator_id, str)
                        and simulator_id.startswith(descendant_prefix)
                        and simulator_id not in seen_ids
                    ):
                        derived_ids.append(simulator_id)
                        seen_ids.add(simulator_id)
            # Keeping an empty key prevents category-only goals from matching
            # slices generated from an unreferenced source object.
            expanded[sliced_category] = derived_ids
        return expanded

    def _build_additional_info(self):
        event = self.controller.last_event
        instance_masks = {str(k): v for k, v in dict(event.instance_masks).items()}
        instance_bboxs = {}
        object_distances = {
            str(obj["objectId"]): obj.get("distance")
            for obj in event.metadata.get("objects", [])
        }
        for k, bbox in dict(event.instance_detections2D).items():
            obj_id = str(k)
            obj_distance = object_distances.get(obj_id)
            if obj_distance is None or obj_distance >= 3.0:
                continue

            if bbox is None:
                instance_bboxs[obj_id] = None
            else:
                instance_bboxs[obj_id] = [int(x) for x in bbox]
        color_to_object_id = {}
        for color, obj_id in dict(event.color_to_object_id).items():
            color_to_object_id[tuple(int(x) for x in color)] = str(obj_id)
        object_id_to_color = {}
        for obj_id, color in dict(event.object_id_to_color).items():
            object_id_to_color[str(obj_id)] = [int(x) for x in color]

        return {
            "instance_seg_frame": event.instance_segmentation_frame,
            "instance_masks": instance_masks,
            "instance_bboxs": instance_bboxs,
            "color_to_object_id": color_to_object_id,
            "object_id_to_color": object_id_to_color,
            "reachable_positions": self.positions,
        }

    def init_episode(
        self,
    ):
        self.controller.reset()
        self._init_episode_common()

        if self.placement_randomization_args is not None:
            self._step_init_checked(
                "InitialRandomSpawn",
                {"action": "InitialRandomSpawn", **self.placement_randomization_args},
            )
        if self.material_randomization_args is not None:
            self._step_init_checked(
                "RandomizeMaterials",
                {"action": "RandomizeMaterials", **self.material_randomization_args},
            )
        if self.lighting_randomization_args is not None:
            self._step_init_checked(
                "RandomizeLighting",
                {"action": "RandomizeLighting", **self.lighting_randomization_args},
            )

        init_mapping = self._resolve_init_step_object_ids()
        if len(self.init_steps):
            for index, step in enumerate(self.init_steps):
                self._step_init_checked(f"init_steps[{index}]", step)

        self._resolve_object_references(init_mapping)

        reachable = self._step_init_checked(
            "GetReachablePositions", {"action": "GetReachablePositions"}
        )
        self.positions = reachable.metadata["actionReturn"]

        self._step_init_checked(
            "SetTemperatureDecayTime",
            {"action": "SetTemperatureDecayTime", "decayTime": 300.0},
        )

        additional_info = self._build_additional_info()
        return self.controller.last_event.frame, additional_info

    def _convert_ref_to_thor_object_id(
        self,
        ref_value: Union[str, List[float]],
        scene_graph: Optional[UnifiedSceneGraph],
    ):
        """
        Convert SG node id (e.g. category_idx) back to THOR objectId when needed.
        Values without a mapping are returned unchanged.
        """
        if not isinstance(ref_value, str) or scene_graph is None:
            return ref_value
        cache = getattr(scene_graph, "cache", None)
        node_id_to_object_id = cache["node_id_to_object_id"]
        if ref_value not in node_id_to_object_id:
            return ref_value
        return node_id_to_object_id[ref_value]

    def _normalize_action_args_refs(
        self,
        action_args: Optional[dict[str, Any]],
        scene_graph: Optional[UnifiedSceneGraph],
    ) -> dict[str, Any]:
        """
        Normalize all action refs (object_ref/tool_ref/...) to THOR objectId.
        """
        normalized_action_args = dict(action_args)
        for key, value in normalized_action_args.items():
            if key.endswith("_ref"):
                normalized_action_args[key] = self._convert_ref_to_thor_object_id(
                    value, scene_graph
                )
        return normalized_action_args

    def step(
        self, action_cls: str, action_args: dict, scene_graph: UnifiedSceneGraph
    ):

        self.num_steps += 1
        done = self.num_steps >= self.max_steps
        action_args = self._normalize_action_args_refs(action_args, scene_graph)

        if action_cls == "moveAhead":
            action_success, feedback = moveAhead(self.controller)
        elif action_cls == "moveBack":
            action_success, feedback = moveBack(self.controller)
        elif action_cls == "moveLeft":
            action_success, feedback = moveLeft(self.controller)
        elif action_cls == "moveRight":
            action_success, feedback = moveRight(self.controller)
        elif action_cls == "turnLeft":
            action_success, feedback = turnLeft(self.controller)
        elif action_cls == "turnRight":
            action_success, feedback = turnRight(self.controller)
        elif action_cls == "lookUp":
            action_success, feedback = lookUp(self.controller)
        elif action_cls == "lookDown":
            action_success, feedback = lookDown(self.controller)
        elif action_cls == "moveHandAhead":
            action_success, feedback = moveHandAhead(self.controller)
        elif action_cls == "moveHandBack":
            action_success, feedback = moveHandBack(self.controller)
        elif action_cls == "moveHandLeft":
            action_success, feedback = moveHandLeft(self.controller)
        elif action_cls == "moveHandRight":
            action_success, feedback = moveHandRight(self.controller)
        elif action_cls == "rotateHand":
            action_success, feedback = rotateHand(self.controller, **action_args)

        elif action_cls == "turnTo_point":
            action_success, feedback = turnTo_point(self.controller, **action_args)
        elif action_cls == "turnTo_obj":
            action_success, feedback = turnTo_obj(self.controller, **action_args)
        elif action_cls == "goTo_obj":
            action_success, feedback = goTo_obj(
                self.controller,
                possible_locations=self.positions,
                room_polygons=self.room_id_to_floor_polygon,
                **action_args,
            )
        elif action_cls == "goTo_loc":
            action_success, feedback = goTo_loc(self.controller, **action_args)
        elif action_cls == "pick_obj":
            action_success, feedback = pick_obj(self.controller, **action_args)
        elif action_cls == "placeTo_recep":
            action_success, feedback = placeTo_recep(self.controller, **action_args)
        elif action_cls == "placeTo_point":
            action_success, feedback = placeTo_point(self.controller, **action_args)
        elif action_cls == "drop":
            action_success, feedback = drop(self.controller)
        elif action_cls == "throw_force":
            action_success, feedback = throw_force(self.controller, **action_args)
        elif action_cls == "pourTo_recep":
            action_success, feedback = pourTo_recep(self.controller, **action_args)
        elif action_cls == "close_recep":
            action_success, feedback = close_recep(self.controller, **action_args)
        elif action_cls == "open_recep":
            action_success, feedback = open_recep(self.controller, **action_args)
        elif action_cls == "slice_obj":
            action_success, feedback = slice_obj(self.controller, **action_args)
        elif action_cls == "turnOn_obj":
            action_success, feedback = turnOn_obj(self.controller, **action_args)
        elif action_cls == "turnOff_obj":
            action_success, feedback = turnOff_obj(self.controller, **action_args)
        elif action_cls == "useUp_obj":
            action_success, feedback = useUp_obj(self.controller, **action_args)

        elif action_cls == "find_objCls":
            action_success, feedback = find_objCls(self.controller, **action_args)

        elif action_cls == "heatWith_tool":
            action_success, feedback = heatWith_tool(self.controller, **action_args)
        elif action_cls == "coolWith_tool":
            action_success, feedback = coolWith_tool(self.controller, **action_args)
        elif action_cls == "washWith_tool":
            action_success, feedback = washWith_tool(self.controller, **action_args)
        elif action_cls == "cookWith_tool":
            action_success, feedback = cookWith_tool(self.controller, **action_args)
        elif action_cls == "fillWith_tool":
            action_success, feedback = fillWith_tool(self.controller, **action_args)

        elif action_cls == "done":
            done = True
            action_success, feedback = True, "done"
        else:
            action_success, feedback = False, f"Invalid action"
            self.num_steps -= 1
            self.failed_steps -= 1

        self.failed_steps += 0 if action_success else 1
        done = done or (self.failed_steps >= self.max_failed_steps)

        obs = self.controller.last_event.frame

        additional_info = self._build_additional_info()

        return obs, action_success, feedback, done, additional_info

    def close(
        self,
    ):
        self.controller.stop()
