# Portions of this file are adapted from partnr-planner
# (https://github.com/facebookresearch/partnr-planner), in particular from
# habitat_llm/world_model/entities/furniture.py and habitat_llm/world_model/entities/floor.py,
# and from habitat-lab (https://github.com/facebookresearch/habitat-lab),
# habitat-lab/habitat/tasks/rearrange/rearrange_sim.py (RearrangeSim.safe_snap_point).
# Copyright (c) Meta Platforms, Inc. and affiliates. Licensed under the MIT License.
# The adapted code has been modified for this project.

from typing import List, Tuple, Union, Optional
import random
import math

import numpy as np
import magnum as mn
import habitat_sim
from habitat_sim.simulator import Simulator
from habitat.datasets.rearrange.samplers.receptacle import (
    Receptacle,
)
from habitat.sims.habitat_simulator.sim_utilities import (
    get_obj_from_id,
    obj_next_to,
    snap_down,
)
from habitat_sim.physics import ManagedRigidObject, ManagedArticulatedObject


def calculate_distance_2D(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2)


def calculate_distance_3D(p1, p2):
    return (p1 - p2).length()


def sample_position_on_receptacle_links_with_reference(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    in_hand_object: Union[ManagedRigidObject, ManagedArticulatedObject],
    receptacle_object: Union[ManagedRigidObject, ManagedArticulatedObject],
    receptacles: List[Receptacle],
    reference_object: Optional[
        Union[ManagedRigidObject, ManagedArticulatedObject]
    ] = None,
    min_sample_distance: float = 0.10,
    sample_region_scale: float = 1,
    max_samples: int = 10,
    max_tries: int = 100,
    dist_thresh: float = 10.0,
    weights: Optional[List[float]] = None,
) -> Tuple[List[Tuple[mn.Vector3, mn.Quaternion]], List[Receptacle], str]:
    # Declare container to store sampled poses
    sampled_poses: List[Tuple[mn.Vector3, mn.Quaternion]] = []
    sampled_receptacles: List[Receptacle] = []
    num_tries = 0

    found_valid_pose = False

    # Rejection sampling
    while len(sampled_poses) < max_samples and num_tries < max_tries:
        # Select a random Receptacle from the valid spatial subset
        receptacle: Receptacle = random.choices(receptacles, weights=weights, k=1)[0]
        sampled_pos = receptacle.sample_uniform_global(sim, sample_region_scale)
        num_tries += 1

        # Cache the state of the grasped object
        cache_pos = in_hand_object.translation
        cache_rot = in_hand_object.rotation

        rec_link_id = receptacle.parent_link
        rec_sim_id = receptacle_object.object_id
        if rec_link_id is not None and rec_link_id >= 0:
            rec_sim_id = receptacle_object.link_ids_to_object_ids[rec_link_id]

        # Teleport the object to the sampled_pos
        in_hand_object.translation = sampled_pos + mn.Vector3(0, 0.08, 0)
        # randomize the yaw orientation (around Y axis)
        rot = random.uniform(0, math.pi * 2.0)
        in_hand_object.rotation = mn.Quaternion.rotation(
            mn.Rad(rot), mn.Vector3.y_axis()
        )
        snap_success = snap_down(
            sim,
            in_hand_object,
            support_obj_ids=[rec_sim_id],
        )

        if snap_success:
            sampled_pos = in_hand_object.translation
        else:
            in_hand_object.translation = cache_pos
            in_hand_object.rotation = cache_rot
            continue

        # However, this is configurable per-proposition and should be pulled from config

        # Call sim next to function to check
        can_add = True
        if reference_object is not None:
            can_add = obj_next_to(
                sim,
                in_hand_object.object_id,
                reference_object.object_id,
            )
        if (
            distance_to_other_samples(sampled_pos, sampled_poses) > min_sample_distance
            and can_add
        ):
            found_valid_pose = True
            dist_to_agent = calculate_distance_2D(robot.translation, sampled_pos)
            if dist_to_agent <= dist_thresh:
                sampled_poses.append((sampled_pos, in_hand_object.rotation))
                sampled_receptacles.append(receptacle)
        # Snap the object back to its original position
        in_hand_object.translation = cache_pos
        in_hand_object.rotation = cache_rot

        if len(sampled_poses) >= max_samples or num_tries >= max_tries:
            break

    if len(sampled_poses) == 0:
        if found_valid_pose:
            return [], [], "dist issue"
        return [], [], ""

    # Sort the samples based on the distance to robot
    return *sort_proposed_samples_based_on_distance_to_agent(
        sampled_poses, robot, sampled_receptacles
    ), ""


def distance_to_other_samples(
    new_sample: mn.Vector3,
    samples: List[Tuple[mn.Vector3, mn.Quaternion]],
) -> float:
    """Compute the distance to other samples in the list.
    :param new_sample: a new placement point
    :param samples: a list of placement tuples (point, orientation) to compare with

    :return: the min L2 distance to other samples
    """
    if len(samples) == 0:
        return float("inf")
    distance = [(new_sample - sample[0]).length() for sample in samples]
    return min(distance)


def sort_proposed_samples_based_on_distance_to_agent(
    sampled_poses: List[Tuple[mn.Vector3, mn.Quaternion]],
    robot: ManagedArticulatedObject,
    sampled_receptacles: Optional[List[Receptacle]] = None,
) -> List[np.ndarray]:
    """Sort the samples based on the distance to agent.
    :param sampled_poses: a list of placement tuples (point,orientation) to compare with
    :param agent: an ArticulatedAgent

    :return: the sorted list of placement points based on their L2 distance to the agent's base position
    """
    cur_base_pos = mn.Vector3(robot.translation)
    distance = [(cur_base_pos - sample[0]).length() for sample in sampled_poses]
    # Sort
    sort_i = sorted(range(len(distance)), key=lambda k: distance[k])
    return [sampled_poses[i] for i in sort_i], [sampled_receptacles[i] for i in sort_i]


def place_to_position_on_receptacle_link(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    in_hand_object: Union[ManagedRigidObject, ManagedArticulatedObject],
    support_object_ids: List[int],
    target_position: mn.Vector3,
) -> Tuple[bool, mn.Vector3, mn.Quaternion]:
    # Cache the state of the grasped object
    cache_pos = in_hand_object.translation
    cache_rot = in_hand_object.rotation

    # Teleport the object to the sampled_pos
    in_hand_object.translation = target_position + mn.Vector3(0, 0.08, 0)
    # randomize the yaw orientation (around Y axis)
    rot = random.uniform(0, math.pi * 2.0)
    in_hand_object.rotation = mn.Quaternion.rotation(mn.Rad(rot), mn.Vector3.y_axis())
    snap_success = snap_down(
        sim,
        in_hand_object,
        support_obj_ids=support_object_ids,
    )

    if snap_success:
        real_pos = in_hand_object.translation
        real_rot = in_hand_object.rotation
        in_hand_object.translation = cache_pos
        in_hand_object.rotation = cache_rot
        return True, real_pos, real_rot
    else:
        in_hand_object.translation = cache_pos
        in_hand_object.rotation = cache_rot
        return False, None, None


def sample_position_on_room_floor_with_reference(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    in_hand_object: Union[ManagedRigidObject, ManagedArticulatedObject],
    largest_indoor_island_idx: int,
    reference_object: Optional[
        Union[ManagedRigidObject, ManagedArticulatedObject]
    ] = None,
    min_sample_distance: float = 0.10,
    sample_region_scale: float = 1,
    max_samples: int = 10,
    max_tries: int = 100,
    dist_thresh: float = 10.0,
    require_region_id: Optional[int] = None,
) -> List[Tuple[mn.Vector3, mn.Quaternion]]:
    # Declare container to store sampled poses
    sampled_poses: List[Tuple[mn.Vector3, mn.Quaternion]] = []
    sampled_pos = None
    num_tries = 0

    found_valid_pose = False

    agent_object_ids = [robot.object_id] + [*robot.link_object_ids.keys()]

    target_object_ids = [in_hand_object.object_id]
    if in_hand_object.is_articulated:
        target_object_ids.extend([*in_hand_object.link_object_ids.keys()])

    relative_y_delta = in_hand_object.translation.y - in_hand_object.aabb.min.y

    while len(sampled_poses) < max_samples and num_tries < max_tries:
        sampled_pos = safe_snap_point(
            sim, in_hand_object.translation, largest_indoor_island_idx
        )
        sampled_pos = sim.pathfinder.get_random_navigable_point_near(
            sampled_pos, radius=2.0, island_index=largest_indoor_island_idx
        )

        num_tries += 1

        # Cache the state of the grasped object
        cache_pos = in_hand_object.translation
        cache_rot = in_hand_object.rotation

        # Teleport the object to the sampled_pos
        in_hand_object.translation = sampled_pos + mn.Vector3(0, 0.1, 0)
        in_hand_object.translation += mn.Vector3(0, relative_y_delta, 0)
        # randomize the yaw orientation (around Y axis)
        rot = random.uniform(0, math.pi * 2.0)
        in_hand_object.rotation = mn.Quaternion.rotation(
            mn.Rad(rot), mn.Vector3.y_axis()
        )
        snap_success = snap_down(
            sim, in_hand_object, max_collision_depth=0.2, support_obj_ids=[0]
        )

        if snap_success:
            sampled_pos = in_hand_object.translation
        else:
            in_hand_object.translation = cache_pos
            in_hand_object.rotation = cache_rot
            continue

        snap_region_id = None
        if require_region_id is not None:
            for region in sim.semantic_scene.regions:
                if region.contains(sampled_pos):
                    snap_region_id = region.id
                    break
            if snap_region_id != require_region_id:
                continue

        # However, this is configurable per-proposition and should be pulled from config

        # Call sim next to function to check
        can_add = True
        if reference_object is not None:
            can_add = obj_next_to(
                sim,
                in_hand_object.object_id,
                reference_object.object_id,
            )
        if (
            distance_to_other_samples(sampled_pos, sampled_poses) > min_sample_distance
            and can_add
        ):
            found_valid_pose = True
            dist_to_agent = calculate_distance_2D(robot.translation, sampled_pos)
            if dist_to_agent <= dist_thresh:
                sampled_poses.append((sampled_pos, in_hand_object.rotation))
        # Snap the object back to its original position
        in_hand_object.translation = cache_pos
        in_hand_object.rotation = cache_rot

        if len(sampled_poses) >= max_samples or num_tries >= max_tries:
            break

    if len(sampled_poses) == 0:
        if found_valid_pose:
            return [], [], "dist issue"
        return [], [], ""

    # Sort the samples based on the distance to robot
    return *sort_proposed_samples_based_on_distance_to_agent(
        sampled_poses, robot, ["" for _ in range(len(sampled_poses))]
    ), ""


def place_to_position_on_room_floor(
    sim: Simulator,
    robot: ManagedArticulatedObject,
    in_hand_object: Union[ManagedRigidObject, ManagedArticulatedObject],
    target_position: mn.Vector3,
) -> Tuple[bool, mn.Vector3, mn.Quaternion]:

    relative_y_delta = in_hand_object.translation.y - in_hand_object.aabb.min.y

    agent_object_ids = [robot.object_id] + [*robot.link_object_ids.keys()]

    target_object_ids = [in_hand_object.object_id]
    if in_hand_object.is_articulated:
        target_object_ids.extend([*in_hand_object.link_object_ids.keys()])

    # Cache the state of the grasped object
    cache_pos = in_hand_object.translation
    cache_rot = in_hand_object.rotation

    # Teleport the object to the sampled_pos
    in_hand_object.translation = target_position + mn.Vector3(0, 0.1, 0)
    in_hand_object.translation += mn.Vector3(0, relative_y_delta, 0)
    # randomize the yaw orientation (around Y axis)
    rot = random.uniform(0, math.pi * 2.0)
    in_hand_object.rotation = mn.Quaternion.rotation(mn.Rad(rot), mn.Vector3.y_axis())

    snap_success = snap_down(
        sim, in_hand_object, max_collision_depth=0.2, support_obj_ids=[0]
    )

    if snap_success:
        real_pos = in_hand_object.translation
        real_rot = in_hand_object.rotation
        in_hand_object.translation = cache_pos
        in_hand_object.rotation = cache_rot
        return True, real_pos, real_rot
    else:
        in_hand_object.translation = cache_pos
        in_hand_object.rotation = cache_rot
        return False, None, None


def get_floor_object_ids(
    sim: Simulator,
    navmesh_point: mn.Vector3,
    ignore_ids: Optional[List[int]] = None,
) -> List[int]:
    """
    Get all object between the provided navmesh point and the stage floor which are not in the ignore list.

    :param sim: The Simulator instance.
    :param navmesh_point: The navmesh point below which to search for objects.
    :param ignore_ids: All object ids which should be ignored in the search for a support surface. Typically those belonging to an object or agent which may be standing or sitting at the search location.
    :return: A list of object ids for the floor. Always includes the stage_id.

    Uses a raycast to detect objects hit before the stage and then culls out the ignored ids.
    Example: when a floor rug is navigable but causes snap_down to fail, we add the rug as a support object id.
    Raises a ValueError if there is nothing below the provided point.
    """
    ray = habitat_sim.geo.Ray(navmesh_point, mn.Vector3(0, -1, 0))
    raycast_results = sim.cast_ray(ray)
    floor_object_ids = [habitat_sim.stage_id]
    if raycast_results.has_hits:
        for hit in raycast_results.hits:
            if hit.object_id == habitat_sim.stage_id:
                # stop once we hit the stage
                break
            floor_object_ids.append(hit.object_id)
    else:
        raise ValueError(
            f"Provided navmesh_point {navmesh_point} does not have anything below it. Is it actually a valid navmesh point?"
        )

    if ignore_ids is not None:
        floor_object_ids = list(set(floor_object_ids) - set(ignore_ids))
    return floor_object_ids


def safe_snap_point(
    sim: Simulator, pos: np.ndarray, largest_indoor_island_idx: int
) -> np.ndarray:
    """
    Returns the 3D coordinates corresponding to a point belonging
    to the biggest navmesh island in the scene and closest to pos.
    When that point returns NaN, computes a navigable point at increasing
    distances to it.
    """
    new_pos = sim.pathfinder.snap_point(pos, largest_indoor_island_idx)

    max_iter = 10
    offset_distance = 1.5
    distance_per_iter = 0.5
    num_sample_points = 1000

    regen_i = 0
    while np.isnan(new_pos[0]) and regen_i < max_iter:
        # Increase the search radius
        new_pos = sim.pathfinder.get_random_navigable_point_near(
            pos,
            offset_distance + regen_i * distance_per_iter,
            num_sample_points,
            island_index=largest_indoor_island_idx,
        )
        regen_i += 1

    assert not np.isnan(new_pos[0]), (
        f"The snap position is NaN. new position: {new_pos}, original position: {pos}"
    )

    return np.array(new_pos)


def bind_object_to_parent_link(
    sync_obj_dict,
    obj: Union[ManagedRigidObject, ManagedArticulatedObject],
    recep: Union[ManagedRigidObject, ManagedArticulatedObject],
    link_id=-1,
):

    if link_id == -1:
        new_parent_node = recep.root_scene_node
    else:
        new_parent_node = recep.get_link_scene_node(link_id)
    obj_node = obj.root_scene_node

    relative_transform = (
        new_parent_node.absolute_transformation().inverted()
        @ obj_node.absolute_transformation()
    )
    sync_obj_dict[obj.object_id] = (new_parent_node, relative_transform)


def sync_object_bindings(sim: Simulator, sync_obj_dict) -> None:
    for object_id, binding in sync_obj_dict.items():
        if binding is None:
            continue
        anchor_node, relative_transform = binding
        obj = get_obj_from_id(sim, object_id)
        if obj is None:
            continue
        obj.root_scene_node.transformation = (
            anchor_node.absolute_transformation() @ relative_transform
        )

