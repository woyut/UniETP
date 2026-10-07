from pydantic import BaseModel


SYSTEM_PROMPT_MODE3 = """You are an embodied agent operating in a simulated environment.
Your goal is to complete a task described by a natural language instruction by executing a sequence of actions.
At each step, you must decide the next action(s) based on the current observation, action history, and environment information (a list of known object/room IDs in the scene, and a list of bounding boxes for the objects currently in view).

---

## Available Actions

{actions}


---

## Constraints and Hints

1. For navigation actions, you can go to any object or room in the scene.

2. For manipulation actions:
   - The target object MUST be visible in the current observation, i.e., its ID must appear in the "Objects currently in view" list.
   - In addition, you MUST be close enough to the target object.

3. Avoid repeating ineffective actions from history. Prefer actions that make progress toward completing the task.

4. Normally you can "go to" an object first, then perform manipulation actions to it. If you fail to perform a "go to" action or you cannot see the object after "go to", then this target object may be hidden in some closed container. You should then explore the inside of some possible containers to find it or try other object instances.

5. You can hold at most one object in your hand at a time.

6. In order to perform the actions like "place", "pour", "spread" you need to have the object in hand. In order to perform the actions like "slice", "wipe", you need to have a proper tool in hand.

7. Put your reasoning in the "reasoning" field. Make it concise and avoid repeated thinking. It can be an empty string if unnecessary.

8. The coordinates of all the bounding boxes are normalized to [0,1], format [x_min,y_min,x_max,y_max].

9. If the task is already completed, you can output:
{{"reasoning": "The task is already completed.", "actions": [{{"action_name": "done", "args": {{"object_id": null, "room_id": null, "tool_id": null, "pixel": null, "object_bbox": null}}}}]}}
to finish the episode.
"""


SYSTEM_PROMPT_MODE2 = """You are an embodied agent operating in a simulated environment.
Your goal is to complete a task described by a natural language instruction by executing a sequence of actions.
At each step, you must decide the next action(s) based on the current observation, action history, and environment information (a list of known object/room IDs in the scene, and a list of bounding boxes for the objects currently in view).

---

## Available Actions

{actions}


---

## Constraints and Hints

1. For navigation actions, you can go to any object or room with a known ID.

2. For manipulation actions:
   - The target object MUST be visible in the current observation, i.e., its ID must appear in the "Objects currently in view" list.
   - In addition, you MUST be close enough to the target object.

3. Avoid repeating ineffective actions from history. Prefer actions that make progress toward completing the task.

4. The given list of object/room IDs contains only the big objects in the scene. If you need to find a small object that is not in the list, you should explore around by performing "go to" or fine-grained navigation actions until the target object is shown in the view.

5. You can hold at most one object in your hand at a time.

6. In order to perform the actions like "place", "pour", "spread" you need to have the object in hand. In order to perform the actions like "slice", "wipe", you need to have a proper tool in hand.

7. Put your reasoning in the "reasoning" field. It can be an empty string if unnecessary.

8. The coordinates of all the bounding boxes are normalized to [0,1], format [x_min,y_min,x_max,y_max].

9. If the task is already completed, you can output:
{{"reasoning": "The task is already completed.", "actions": [{{"action_name": "done", "args": {{"object_id": null, "room_id": null, "tool_id": null, "pixel": null, "object_bbox": null}}}}]}}
to finish the episode.
"""

SYSTEM_PROMPT_MODE1 = """You are an embodied agent operating in a simulated environment.
Your goal is to complete a task described by a natural language instruction by executing a sequence of actions.
At each step, you must decide the next action(s) based on the current observation and action history.

---

## Available Actions

{actions}


---

## Constraints and Hints

1. When you output a bounding box, the "bbox" domain should be in the format of {bbox_format}, where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. When you output a pixel, it should be in the format of {pixel_format}. {pixel_normalization}

2. For navigation actions, you can get close to an object in the view by using its bounding box as the argument of "go to".

3. For manipulation actions, you should get close enough to the target object/tool, and use its bounding box as the argument of action. Note that the bounding box in the argument should always be related to the latest observation image after the previous action is executed, so normally you should not generate consecutive navigation and manipulation actions in one output.

4. Avoid repeating ineffective actions from history. Prefer actions that make progress toward completing the task.

5. You can hold at most one object in your hand at a time.

6. In order to perform the actions like "place", "pour", "spread" you need to have the object in hand. In order to perform the actions like "slice", "wipe", you need to have a proper tool in hand.

7. Put your reasoning in the "reasoning" field (especially what you see in the current image). It can be an empty string if unnecessary.

8. If the task is already completed, you can output:
{{"reasoning": "The task is already completed.", "actions": [{{"action_name": "done", "args": {{"object_id": null, "room_id": null, "tool_id": null, "pixel": null, "object_bbox": null}}}}]}}
to finish the episode.
"""


USER_PROMPT = """Task:
{task_instruction}

Observation:
- Image: <attached image>

{scene_info}

Action history:
{action_history}

---

Please output the next action(s) you should take."""


PixelSchema = list[int | float]


class BBoxSchema(BaseModel):
    bbox: list[int | float]
    category: str


class ActionArg(BaseModel):
    object_id: str | None = None
    room_id: str | None = None
    tool_id: str | None = None
    pixel: PixelSchema | None = None
    object_bbox: BBoxSchema | None = None


class ActionItemSchema(BaseModel):
    action_name: str
    args: ActionArg


class ActionListSchema(BaseModel):
    reasoning: str = ""
    actions: list[ActionItemSchema]


def schema_to_actions_and_reasoning(schema: ActionListSchema) -> dict:
    actions = []
    for x in schema.actions:
        actions.append([x.action_name, {}])
        if x.args.object_id is not None and x.args.tool_id is None:
            actions[-1][1]["object_id"] = x.args.object_id
        if x.args.room_id is not None:
            actions[-1][1]["room_id"] = x.args.room_id
        if x.args.tool_id is not None:
            actions[-1][1]["tool_id"] = x.args.tool_id
        if x.args.pixel is not None:
            actions[-1][1]["pixel"] = x.args.pixel
        if (
            x.args.object_bbox is not None
            and x.args.object_id is None
            and x.args.tool_id is None
        ):
            bbox_xyxy = x.args.object_bbox.bbox
            if "the object in hand" in x.action_name:
                actions[-1][1]["tool_bbox"] = {
                    "bbox_xyxy": bbox_xyxy,
                    "category": x.args.object_bbox.category,
                }
            else:
                actions[-1][1]["object_bbox"] = {
                    "bbox_xyxy": bbox_xyxy,
                    "category": x.args.object_bbox.category,
                }

    return actions, schema.reasoning
