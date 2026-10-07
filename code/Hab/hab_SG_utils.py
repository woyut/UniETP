# Portions of this file are adapted from partnr-planner
# (https://github.com/facebookresearch/partnr-planner), in particular from
# habitat_llm/sims/metadata_interface.py, habitat_llm/perception/perception_sim.py
# and habitat_llm/utils/sim.py.
# Copyright (c) Meta Platforms, Inc. and affiliates. Licensed under the MIT License.
# The adapted code has been modified for this project.

from typing import Dict, List, Optional, Union, Any
from collections import defaultdict
import os
import json
import csv
import re
from pathlib import Path

import magnum as mn
import pandas as pd
import habitat_sim
from habitat_sim.simulator import Simulator
from habitat_sim.physics import ManagedArticulatedObject, ManagedRigidObject
from habitat_sim.metadata import MetadataMediator
import habitat.sims.habitat_simulator.sim_utilities as sutils

from Hab.hab_env import Hab_Environment
from scene_graph.graph_schema import (
    Graph,
    Entity,
    EntityType,
    RelationType,
    ObjectAttributes,
    PropertyType,
    flip_edge,
    StateType,
)
from scene_graph.unified_scene_graph import UnifiedSceneGraph


from habitat.datasets.rearrange.samplers.receptacle import (
    Receptacle,
    find_receptacles,
    get_excluded_recs_from_filter_file,
    get_recs_from_filter_file,
)

from task.commonsense_knowledge.hab import GRASPABLE_OBJS


def _extract_asset_hash(handle: str) -> str:
    filename = os.path.basename(handle)
    return filename.split(".")[0].split("_:")[0]


def get_faucet_points(sim: Simulator) -> Dict[str, List[mn.Vector3]]:
    """
    Load all Faucet MarkerSets in global space.

    :param sim: The Simulator instance.

    :return: A dictionary with keys for each object handle containing a faucet, and values as a list of faucet points.
    """
    objs = sutils.get_all_objects(sim)
    obj_markersets: Dict[str, List[mn.Vector3]] = {}
    for obj in objs:
        all_obj_marker_sets = obj.marker_sets
        if all_obj_marker_sets.has_taskset("faucets"):
            # this object has faucet annotations
            obj_markersets[obj.handle] = []
            faucet_marker_sets = all_obj_marker_sets.get_taskset_points("faucets")
            for link_name, link_faucet_markers in faucet_marker_sets.items():
                link_id = -1
                if link_name != "root":
                    link_id = obj.get_link_id_from_name(link_name)
                for _marker_subset_name, points in link_faucet_markers.items():
                    global_points = obj.transform_local_pts_to_world(points, link_id)
                    obj_markersets[obj.handle].extend(global_points)
    return obj_markersets


class MetadataInterface:
    """
    MetadataInterface provides a lightweight interface for managing the semantic metadata from csv files for HSSD.

    MetadataInterface loads and processes metadata about objects and receptacles.
    This class also offers several methods to work with the loaded metadata, such as queries for: common-sense mapping from regions to objects, object/furniture semantic categories, and searching for objects or receptacles of a specific semantic class.
    """

    def __init__(self, hab_data_dir: str) -> None:
        """
        Initialize the MetadataInterface, loading the metadata from provided paths.
        Does not fill internal ManagedObject handle mapping caches until `refresh_scene_caches` is called with a Simulator instance provided.

        :param metadata_source_dict: A dict containing paths to metadata files. See default_metadata_dict.
        """

        # common sense mappings from region categories to object categories typically found in the room
        self.commonsense_room_objects: Dict[str, List[str]] = {}
        # cache the source of each hash
        self.hash_to_source: Dict[str, str] = {}
        # cache the lexicon of available non-static (ovmm) objects
        self.dynamic_lexicon: List[str] = []
        # maps object states to semantic classes which can have those states. See affordance_objects.csv
        self.affordance_info: Dict[str, List[str]] = {}
        self.metadata = self.load_metadata(hab_data_dir)

        self.receptacles: List[Receptacle] = None

        # generate a lexicon from the metadata
        self.hash_to_cat: Dict[str, str] = {}
        self.lexicon: List[str] = []  # all object classes annotated
        for index in range(self.metadata.shape[0]):
            cat = self.metadata.at[index, "type"]
            self.hash_to_cat[self.metadata.at[index, "handle"]] = cat
            self.lexicon.append(cat)
        # deduplicate lexicon
        self.lexicon = list(set(self.lexicon))
        # remove non strings (e.g. nan)
        self.lexicon = [entry for entry in self.lexicon if isinstance(entry, str)]

        # semantic naming for ReceptacleObjects (e.g. table_1)
        self.recobj_semname_to_handle: Dict[str, str] = {}
        self.recobj_handle_to_semname: Dict[str, str] = {}
        # consistent semantic naming for SemanticRegions (e.g. "living room" becomes "living_room_0")
        self.region_ix_to_semname: Dict[int, str] = {}
        self.region_semname_to_id: Dict[str, int] = {}
        # maps region index to key in commonsense_room_objects if a match was found
        self.region_ix_to_room_key: Dict[int, str] = {}

        asset_info_path = (
            Path(__file__).resolve().parents[1]
            / "task"
            / "commonsense_knowledge"
            / "hab_asset_info.json"
        )
        with asset_info_path.open(encoding="utf-8") as f:
            self.asset_info = json.load(f)
        self.new_obj_assets = {}
        with open(
            os.path.join(hab_data_dir, "metadata", "object_categories_filtered.csv"),
            newline="",
            encoding="utf-8",
        ) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.new_obj_assets[row[0]] = row[1].replace(" ", "_")

    def load_metadata(self, hab_data_dir: str) -> pd.DataFrame:
        """
        This method loads the metadata about objects and receptacles from csv and json files.
        This data will typically include tags associated with these objects such as semantic class, product descriptions, object state affordances, etc...

        Fills internal structures:
        - self.commonsense_room_objects
        - self.hash_to_source
        - self.dynamic_lexicon

        :param metadata_dict: A dict containing paths to metadata files. See default_metadata_dict.

        :return: The DataFrame object containing the object hash name to semantic classes map.
        """

        object_metadata_path = os.path.join(
            hab_data_dir, "metadata", "object_categories_filtered.csv"
        )
        static_object_metadata_path = os.path.join(
            hab_data_dir, "metadata", "fpmodels-with-decomposed.csv"
        )
        room_objects_json_path = os.path.join(
            hab_data_dir, "metadata", "room_objects.json"
        )
        object_affordances_csv_path = os.path.join(
            hab_data_dir, "metadata", "affordance_objects.csv"
        )

        # Make sure that the paths are valid
        if not os.path.exists(object_metadata_path):
            raise Exception(f"Object metadata file not found, {object_metadata_path}")
        if not os.path.exists(static_object_metadata_path):
            raise Exception(
                f"Receptacle metadata file not found, {static_object_metadata_path}"
            )
        if not os.path.exists(room_objects_json_path):
            raise Exception(
                f"Common sense region to object class json mapping file not found, {room_objects_json_path}"
            )

        # first load the json room->object categories map
        self.commonsense_room_objects = {}
        with open(room_objects_json_path, "r") as f:
            commonsense_room_objects = json.load(f)
            for room_name in commonsense_room_objects:
                # lower-case room names for lookup
                self.commonsense_room_objects[room_name.lower()] = (
                    commonsense_room_objects[room_name]
                )

        # load object affordances metadata
        self.affordance_info = {}
        with open(object_affordances_csv_path, "r", newline="") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",")
            for row in csvreader:
                state_type = row[0]
                allowed_classes = [
                    re.sub("[^a-zA-Z_]", "", r) for r in row[2:] if len(r) > 0
                ]
                self.affordance_info[state_type] = allowed_classes

        # Read the metadata files
        df_static_objects = pd.read_csv(static_object_metadata_path)
        df_objects = pd.read_csv(object_metadata_path)

        # Rename some columns
        df1 = df_static_objects.rename(
            columns={"id": "handle", "main_category": "type"}
        )
        df2 = df_objects.rename(columns={"id": "handle", "clean_category": "type"})

        # Drop the rest of the columns in both DataFrames
        df1 = df1[["handle", "type"]]
        df2 = df2[["handle", "type"]]

        # setup the hash to source mapping
        for index in range(df1.shape[0]):
            self.hash_to_source[df1.at[index, "handle"]] = "hssd"
        for index in range(df2.shape[0]):
            cat = df2.at[index, "type"]
            self.hash_to_source[df2.at[index, "handle"]] = "dynamic"
            self.dynamic_lexicon.append(cat)
        self.dynamic_lexicon = list(set(self.dynamic_lexicon))

        # Merge the two data frames
        union_df = pd.concat([df1, df2], ignore_index=True)

        return union_df

    def match_common_sense_region_objects_for_scene(self, sim: Simulator) -> None:
        """
        Attempts to match Simulator SemanticRegions to room name keys in self.commonsense_room_objects.
        Uses SemanticCategory.name() and attempts string regularization to match the keys in the map.

        :param sim: Simulator instance is necessary to extract active SemanticRegions.
        """

        self.region_ix_to_room_key = {}
        for rix, region in enumerate(sim.semantic_scene.regions):
            cat_names = [region.category.name()]
            cat_names.extend(
                cat_names[0].split("/")
            )  # sometimes annotations contain multiple synonyms split by "/"
            matching_room_name_keys = [
                cat_name
                for cat_name in cat_names
                if cat_name.lower() in self.commonsense_room_objects
            ]
            matching_room_name_keys = list(set(matching_room_name_keys))
            if matching_room_name_keys:
                self.region_ix_to_room_key[rix] = matching_room_name_keys[0]

    def refresh_scene_caches(
        self, sim: Simulator, filter_receptacles: bool = True
    ) -> None:
        """
        When a new Simulator instance is initialized we need to refresh internal instance caches.

        Also populates a mapping of Entity::Receptacle semantic names used to specify generation.

        Semantic names are generated by collecting all Receptacle parent objects and enumerating them.

        :param sim: Simulator instance. Necessary to extract active ManagedObjects, Receptacles, and SemanticRegions.
        :param filter_receptacles: If true, apply the rec_filter_file for the scene during Receptacle parsing. Only accessible and valid receptacles (as annotated in the filter file) will be available through this MetadataInterface if this option is used.
        """

        # first parse the scene's active receptacles
        self.receptacles = find_receptacles(sim, filter_receptacles)

        self.recobj_semname_to_handle = {}
        self.recobj_handle_to_semname = {}

        sem_class_counter: Dict[str, int] = defaultdict(lambda: 0)

        for receptacle in self.receptacles:
            receptacle_object = sutils.get_obj_from_handle(
                sim, receptacle.parent_object_handle
            )
            if receptacle_object.handle in self.recobj_handle_to_semname:
                # this parent object is already registered
                continue
            sem_class = self.get_object_instance_category(receptacle_object)
            if sem_class is None:
                continue
            # computes the semantic name of the receptacle
            instance_sem_name = f"{sem_class}_{sem_class_counter[sem_class]}"
            sem_class_counter[sem_class] += 1
            self.recobj_semname_to_handle[instance_sem_name] = receptacle_object.handle
            self.recobj_handle_to_semname[receptacle_object.handle] = instance_sem_name

        # construct the region semantic name maps
        self.region_ix_to_semname = {}
        self.region_semname_to_id = {}
        for rix, region in enumerate(sim.semantic_scene.regions):
            std_name = region.category.name().replace(" ", "_")
            indexed_std_name = std_name + "_0"
            count = 1
            while indexed_std_name in self.region_semname_to_id:
                indexed_std_name = f"{std_name}_{count}"
                count += 1
            self.region_ix_to_semname[rix] = indexed_std_name
            self.region_semname_to_id[indexed_std_name] = rix

        self.match_common_sense_region_objects_for_scene(sim)

    def get_object_category(self, obj_hash: str) -> Optional[str]:
        """
        Get the semantic class lexicon entry corresponding to the object hash.

        :param obj_hash: The shortened name of the object created from a ManagedObject handle by stripping filepath prefix, file ending postfix, and instance handle index strings postfix.

        :return: The semantic class string or None if either the object has no annotated class or the hash cannot be matched to an object.
        """

        if obj_hash in self.hash_to_cat:
            obj_class = self.hash_to_cat[obj_hash]
            if isinstance(obj_class, str) and len(obj_class) > 0:
                return obj_class
        return None

    def get_object_instance_category(
        self, obj: Union[ManagedRigidObject, ManagedArticulatedObject]
    ) -> Optional[str]:
        """
        Get the semantic class lexicon entry corresponding to the ManagedObject instance's template hash.

        :param obj: The ManagedObject for which to query the semantic class.

        :return: The semantic class string or None if either the object has no annotated class or the object cannot be found in internal caches.
        """

        obj_hash = sutils.object_shortname_from_handle(obj.handle)
        obj_cat = self.get_object_category(obj_hash)
        return obj_cat

    def get_scene_lexicon(self, sim: Simulator) -> List[str]:
        """
        Get the lexicon of the current scene contents by scraping the contents.

        :param sim: Get the lexicon of semantic classes for all active objects in the Simulator instance's active scene.

        :return: A list of all semantic classes in the currently active scene.
        """

        scene_lexicon = []
        rom = sim.get_rigid_object_manager()
        aom = sim.get_articulated_object_manager()
        # get all rigid and articulated object instances
        all_objs = list(rom.get_objects_by_handle_substring().values()) + list(
            aom.get_objects_by_handle_substring().values()
        )
        for obj in all_objs:
            obj_hash = sutils.object_shortname_from_handle(obj.handle)
            obj_cat = self.get_object_category(obj_hash)
            scene_lexicon.append(obj_cat)
        # de-dup
        scene_lexicon = list(set(scene_lexicon))
        # remove non-strings (e.g. None)
        scene_lexicon = [entry for entry in scene_lexicon if isinstance(entry, str)]
        return scene_lexicon

    def get_scene_objs_of_class(
        self, sim: Simulator, sem_class: str
    ) -> List[Union[ManagedRigidObject, ManagedArticulatedObject]]:
        """
        Get all object instances of a given class in the currently instanced scene.

        :param sim: The Simulator instance.
        :param sem_class: The semantic class name.

        :return: The list of ManagedObject instances belonging to the desired semantic class.
        """

        objs_of_class: List[Union[ManagedRigidObject, ManagedArticulatedObject]] = []
        if sem_class not in self.lexicon:
            return objs_of_class

        # get all rigid and articulated object instances
        all_objs = sutils.get_all_objects(sim)
        for obj in all_objs:
            obj_hash = sutils.object_shortname_from_handle(obj.handle)
            obj_cat = self.get_object_category(obj_hash)
            if obj_cat == sem_class:
                objs_of_class.append(obj)

        return objs_of_class

    def get_scene_recs_of_class(
        self,
        sem_class: str,
    ) -> List[Receptacle]:
        """
        Get all Receptacles of a given semantic class in the currently instanced scene.
        Concretely, searches for Receptacle's with parent objects belonging to the given class.
        NOTE: Must be called after 'self.refresh_scene_caches'.

        :param sem_class: The semantic class name.

        :return: The list of matching Receptacles.
        """

        if self.receptacles is None:
            raise ValueError(
                "self.receptacles is None. No receptacles have been scraped from the scene, call MetadataInterface.refresh_scene_caches() first."
            )

        class_recs: List[Receptacle] = []

        for rec in self.receptacles:
            parent_obj_hash = sutils.object_shortname_from_handle(
                rec.parent_object_handle
            )
            rec_cat = self.get_object_category(parent_obj_hash)
            if rec_cat == sem_class:
                class_recs.append(rec)

        return class_recs

    def get_template_handles_of_class(
        self,
        mm: MetadataMediator,
        sem_class: str,
        dynamic_source: bool = True,
    ) -> List[str]:
        """
        Search the MetadataMediator for all objects of a given class and return a list of template handles.
        This search will cover the entirety of a SceneDataset, returning all matches, not limited to the instanced scene.

        :param mm: The MetadataMediator from which to query the handles. Should already have loaded all asset config templates.
        :param sem_class: The semantic category of the objects which should be returned.
        :param dynamic_source: Whether or not to limit the resulting handles to those enumerated in the external dynamic objects csv file. I.e., exclude Furniture objects.

        :return: The list of relevant object template handles.
        """

        otm = mm.object_template_manager
        class_handles: List[str] = []
        if (
            dynamic_source and sem_class not in self.dynamic_lexicon
        ) or sem_class not in self.lexicon:
            return class_handles
        for handle in otm.get_file_template_handles():
            obj_short_name = sutils.object_shortname_from_handle(handle)
            if dynamic_source and (
                obj_short_name not in self.hash_to_source
                or self.hash_to_source[obj_short_name] != "dynamic"
            ):
                continue
            obj_cat = self.get_object_category(obj_short_name)
            if obj_cat == sem_class:
                class_handles.append(handle)
        return class_handles

    def get_region_rec_contents(self, sim: Simulator) -> Dict[str, List[str]]:
        """
        Get the current region set membership for the loaded scene.
        Returns a map of SemanticRegion semantic names (see self.region_ix_to_semname) to a list of Furniture object semantic names for all objects which have Receptacles annotated.

        :param sim: The Simulator instance.

        :return: The dict mapping region semantic names to lists of Furniture object handles.
        """

        if self.receptacles is None:
            raise ValueError(
                "No receptacles have been scraped from the scene, call 'refresh_scene_caches()' first."
            )

        region_recs: Dict[str, List[str]] = {}

        ao_link_map = sutils.get_ao_link_id_map(sim)

        # search directly in the semantic names map constructed during scene refresh
        for rec_obj_handle, obj_sem_name in self.recobj_handle_to_semname.items():
            parent_obj = sutils.get_obj_from_handle(sim, rec_obj_handle)
            obj_regions = sutils.get_object_regions(
                sim, parent_obj, ao_link_map=ao_link_map
            )
            for rix, _ratio in obj_regions:
                reg_name = self.region_ix_to_semname[rix]
                if reg_name not in region_recs:
                    region_recs[reg_name] = []
                region_recs[reg_name].append(obj_sem_name)

        return region_recs

    def get_object_property_from_metadata(
        self, handle: str, metadata_field: str
    ) -> Any:
        """
        This method returns the value of the requested property using the loaded metadata DataFrame.
        For example, this could be used to extract the semantic type of any object
        in HSSD. Not that the property should exist in the merged DataFrame object.
        See `load_metadata()`.

        NOTE: currently unused, but kept for potential usefulness.

        :param handle: The handle of the object for which the metadata entry should be queried.
        :param metadata_field: The metadata field to query for the object.

        :return: The value corresponding the requested 'metadata_field' for the passed object 'handle'.
        """

        # Declare default
        property_value = "unknown"

        # keep handle-hash parsing consistent across this module
        handle_hash = _extract_asset_hash(handle)

        # Use loc to locate the row with the specific key
        object_row = self.metadata.loc[self.metadata["handle"] == handle_hash]

        # Extract the value from the object_row
        if not object_row.empty:
            # Make sure the property value is not nan or empty
            if (
                object_row[metadata_field].notna().any()
                and (object_row[metadata_field] != "").any()
            ):
                property_value = object_row[metadata_field].values[0]
        else:
            raise NotImplementedError

        return property_value


def read_metadata(handle: str, metadata: MetadataInterface, prop: str):
    handle_hash = _extract_asset_hash(handle)
    assert prop == "type"

    if handle_hash in metadata.new_obj_assets:
        return metadata.new_obj_assets[handle_hash]
    elif handle_hash in metadata.asset_info:
        return metadata.asset_info[handle_hash]["category"]

    return "unknown"

    # Use loc to locate the row with the specific key
    object_row = metadata.loc[metadata["handle"] == handle_hash]

    # Extract the value from the object_row
    if not object_row.empty:
        # Make sure the property value is not nan or empty
        if object_row[prop].notna().any() and (object_row[prop] != "").any():
            property_value = object_row[prop].values[0]
        else:
            property_value = "unknown"
    else:
        raise ValueError(f"Handle {handle} not found in the metadata.")

    return property_value


def check_metadata_graspable(handle: str, metadata: MetadataInterface):
    handle_hash = _extract_asset_hash(handle)
    if handle_hash in metadata.new_obj_assets:
        return True
    if handle_hash in metadata.asset_info:
        return metadata.asset_info[handle_hash]["is_pickable"]

    return False


def get_receptacle_dict(
    sim: Simulator,
    cached_receptacles: List[Receptacle],
    scene_filter_filepath: str,
    metadata_interface: MetadataInterface,
    object_id_to_class: dict[int, str],
) -> Dict[str, Dict[str, List[Receptacle]]]:
    """
    Get a dictionary from parent ManagedObject handle to lists of child (Hab)Receptacles keyed by relationship name "on" or "within".
    "Within" objects are those which can only be accessed by opening the "default_link" of the parent ArticulatedObject.

    :param cached_receptacles: a list of (Hab)Receptacles.

    :return: A dict mapping parent ManagedObject instance handles to separate "on" and "within" subsets of (Hab)Receptacles.
    """

    rec_dict: Dict[str, Dict[str, List[Receptacle]]] = {}
    within_recs = get_recs_from_filter_file(
        scene_filter_filepath, filter_types=["within_set"] + ["access_filtered"]
    )
    for rec in cached_receptacles:
        parent_obj_handle = rec.parent_object_handle
        parent_obj = sutils.get_obj_from_handle(sim, parent_obj_handle)
        if parent_obj.is_articulated:
            rel_name = "within" if rec.unique_name in within_recs else "on"
        else:
            category = metadata_interface.get_object_instance_category(parent_obj)

            category = object_id_to_class[parent_obj.object_id]
            if category in ["bathtub", "sink"]:
                rel_name = "within"
            else:
                rel_name = "on"

        if parent_obj_handle not in rec_dict:
            rec_dict[parent_obj_handle] = {"on": [], "within": []}
        rec_dict[parent_obj_handle][rel_name].append(rec)

    return rec_dict


def get_room_ID(
    sim: Simulator, handle: str, sim_region_id_to_SG_node_ID: Dict[str, str]
) -> str:
    """
    Get the name of the room that contains a given object based off of the simulator object regions.

    :param handle: The handle of the object.

    :return: The name of the room that contains the object or 'unknown_room' if not found.

    :raises ValueError: If the object is not in any region.
    """
    ao_link_map = sutils.get_ao_link_id_map(sim)
    regions = sutils.get_object_regions(
        sim, sutils.get_obj_from_handle(sim, handle), ao_link_map=ao_link_map
    )
    if len(regions) == 0:
        room_ID = sim_region_id_to_SG_node_ID["unknown_room"]
    else:
        region_index, _ = regions[0]
        region_id = sim.semantic_scene.regions[region_index].id
        room_ID = sim_region_id_to_SG_node_ID[region_id]
    return room_ID


def add_house_to_graph(graph: Graph):
    """
    This method adds the root node house to the the gt_graph.
    """
    # Create root node
    house = Entity(
        ID="house",
        type=EntityType.HOUSE,
        category="house",
        attributes=ObjectAttributes(),
        additional_info={"asset_name": "house_0"},
    )
    graph.add_node(house)


def add_rooms_to_graph(graph: Graph, env: Hab_Environment, idx_start: int):
    """
    This method adds room nodes to the gt_graph.
    This is done by querying in which room does a given furniture lie.

    :param sim: Simulator instance

    """

    # Add room nodes to the graph
    region_names = {}
    if len(env.sim.semantic_scene.regions) == 0:
        raise ValueError(f"No regions found in the scene: {env.scene_id}")
    idx = idx_start
    sim_region_id_to_SG_node_ID = {}
    idx2region_idx = {}
    for region_idx, region in enumerate(env.sim.semantic_scene.regions):
        region_name = region.category.name().split("/")[0].replace(" ", "_")
        if region_name not in region_names:
            region_names[region_name] = 0
        region_names[region_name] = region_names[region_name] + 1
        room_name = f"{region_name}_{region_names[region_name]}"

        # Add a valid point on floor as room location
        point_on_floor = sutils.get_floor_point_in_region(env.sim, region_idx)

        # Create properties dict
        if point_on_floor is not None:
            point_on_floor = list(point_on_floor)
            additional_info = {
                "asset_name": room_name,
                "room_floor_positions": point_on_floor,
            }
        else:
            additional_info = {"asset_name": room_name}

        # Create room node
        room = Entity(
            ID=f"{region_name}_{idx}",
            type=EntityType.ROOM,
            category=region_name,
            attributes=ObjectAttributes(),
            additional_info=additional_info,
        )

        # Update mapping from region id to room name
        sim_region_id_to_SG_node_ID[region.id] = f"{region_name}_{idx}"
        idx2region_idx[idx] = region_idx
        idx += 1

        # Add room nodes to the ground truth graph
        # The edges to furniture will be added
        # in the add_furniture_and_receptacles_to_gt_graph method
        graph.add_node(room)

        # Connect room to the root node house
        graph.add_edge(
            room,
            "house",
            RelationType.INSIDE_HOUSE,
            flip_edge(RelationType.INSIDE_HOUSE),
        )

    # Add an unknown room for redundancy
    unknown_room = Entity(
        ID=f"unknown_room_{idx}",
        type=EntityType.ROOM,
        category="unknown_room",
        attributes=ObjectAttributes(),
        additional_info={"asset_name": "unknown_room"},
    )
    sim_region_id_to_SG_node_ID["unknown_room"] = f"unknown_room_{idx}"
    idx += 1
    # Add unknown room nodes to the ground truth graph
    graph.add_node(unknown_room)
    # Connect room to the root node house
    graph.add_edge(
        unknown_room,
        "house",
        RelationType.INSIDE_HOUSE,
        flip_edge(RelationType.INSIDE_HOUSE),
    )
    return idx, sim_region_id_to_SG_node_ID, idx2region_idx


def add_receptacles_to_gt_graph(
    graph: Graph,
    env: Hab_Environment,
    metadata_interface: MetadataInterface,
    sim_region_id_to_SG_node_ID: Dict[str, str],
):
    # Get faucet locations
    faucet_points = get_faucet_points(env.sim)

    scene_filter_filepath = os.path.join(
        env.hab_data_dir,
        "scene_filter_files/articulated_scene_filter_files",
        f"{env.scene_id}.rec_filter.json",
    )
    exclude_filter_strings = get_excluded_recs_from_filter_file(
        scene_filter_filepath, filter_types=["height_filtered", "stability_filtered"]
    )
    rom = env.sim.get_rigid_object_manager()
    rigid_objects_handles = rom.get_object_handles()
    all_receptacles = find_receptacles(
        env.sim,
        ignore_handles=None,
        exclude_filter_strings=exclude_filter_strings,
    )

    # Get dict mapping furniture sim handles to "on" and "within" sets containing lists of HabReceptacles
    receptacle_dict = get_receptacle_dict(
        env.sim,
        all_receptacles,
        scene_filter_filepath,
        metadata_interface,
        env.object_id_to_class,
    )

    sim_handle_to_SG_node_ID = {}
    rec_name_to_rec = {}
    # Iterate through furniture to rec dict and populate the graph
    for sim_handle in receptacle_dict:
        obj = sutils.get_obj_from_handle(env.sim, sim_handle)

        # Get furniture type using metadata
        category = read_metadata(sim_handle, metadata_interface, "type")

        # Generate name for furniture
        node_ID = f"{category}_{obj.object_id}"

        # An array to track non-receptacle sub-components of the furniture, i.e. faucet, power outlets,
        components = []
        if sim_handle in faucet_points:
            components.append("faucet")

        receptacle_links = defaultdict(list)
        rec_counter = 0
        for proposition in receptacle_dict[sim_handle]:
            assert proposition in ["on", "within"]
            rec_list = receptacle_dict[sim_handle][proposition]
            for hab_rec in rec_list:
                rec_link_idx = hab_rec.parent_link

                rec_name = hab_rec.unique_name
                if obj.handle in rigid_objects_handles:
                    assert rec_link_idx is None

                    receptacle_links[-1].append(
                        [rec_name, proposition, obj.object_id, hab_rec.unique_name]
                    )
                else:
                    assert rec_link_idx > -1, rec_name

                    receptacle_links[rec_link_idx].append(
                        [
                            rec_name,
                            proposition,
                            obj.link_ids_to_object_ids[rec_link_idx],
                            hab_rec.unique_name,
                        ]
                    )
                    if rec_link_idx == -1:
                        assert obj.link_ids_to_object_ids[rec_link_idx] == obj.object_id
                rec_name_to_rec[rec_name] = hab_rec
                rec_counter += 1
        # Create furniture instance and receptacle instance
        node = Entity(
            node_ID,
            EntityType.OBJECT,
            category,
            ObjectAttributes(),
            {
                "asset_name": sim_handle,
                "is_articulated": obj.is_articulated,
                "position": obj.translation,
                "components": components,
                "receptacle_links": receptacle_links,
            },
        )  # components: An array to track non-receptacle sub-components of the furniture, i.e. faucet, power outlets,

        if category in GRASPABLE_OBJS:
            node.set_property(PropertyType.GRASPABLE)

        node.set_property(PropertyType.IS_RECEPTACLE)

        # Add furniture to the graph
        graph.add_node(node)

        # Add name to handle mapping
        assert sim_handle not in sim_handle_to_SG_node_ID
        sim_handle_to_SG_node_ID[sim_handle] = node_ID

        # Fetch room for this furniture
        room_ID = get_room_ID(env.sim, sim_handle, sim_region_id_to_SG_node_ID)

        # Add edge between furniture and room
        graph.add_edge(
            node_ID,
            room_ID,
            RelationType.INSIDE_ROOM,
            flip_edge(RelationType.INSIDE_ROOM),
        )

    # Confirm that the gt graph is not empty
    if graph.is_empty():
        raise ValueError(
            "Attempted to load all furniture, but none were found in the scene"
        )
    return sim_handle_to_SG_node_ID, receptacle_dict, rec_name_to_rec


def add_objects_to_graph(
    graph: Graph,
    env: Hab_Environment,
    metadata_interface: MetadataInterface,
    sim_handle_to_SG_node_ID: dict[str, str],
):
    """
    This method adds objects to the gt_graph during the graph initialization
    """
    rom = env.sim.get_rigid_object_manager()
    aom = env.sim.get_articulated_object_manager()
    objects_handles = rom.get_object_handles() + aom.get_object_handles()
    for obj_handle in objects_handles:
        if obj_handle not in sim_handle_to_SG_node_ID:
            sim_obj = sutils.get_obj_from_handle(env.sim, obj_handle)
            category = read_metadata(obj_handle, metadata_interface, "type")

            position = list(sim_obj.translation)
            additional_info = {
                "asset_name": obj_handle,
                "position": position,
            }
            obj_ID = f"{category}_{sim_obj.object_id}"
            assert obj_handle not in sim_handle_to_SG_node_ID
            sim_handle_to_SG_node_ID[obj_handle] = obj_ID
            obj_node = Entity(
                obj_ID, EntityType.OBJECT, category, ObjectAttributes(), additional_info
            )

            if category in GRASPABLE_OBJS:
                obj_node.set_property(PropertyType.GRASPABLE)

            graph.add_node(obj_node)


def add_agents_to_graph(
    graph: Graph,
    env: Hab_Environment,
    sim_region_id_to_SG_node_ID: dict[str, str],
    sim_handle_to_SG_node_ID: dict[str, str],
    type: EntityType = EntityType.ROBOT,
    ID: str = "robot_agent",
):
    """
    Method to add agents to the ground truth graph during initialization.
    """
    # Get articulated agent
    articulated_agent = env.robot

    # Create properties dict
    additional_info = {
        "position": list(articulated_agent.translation),
        "is_articulated": True,
    }

    # Add Agent node to the world
    agent = Entity(ID, type, "agent", ObjectAttributes(), additional_info)
    graph.add_node(agent)

    # Add agent to the conversion dict
    assert env.robot.handle not in sim_handle_to_SG_node_ID
    sim_handle_to_SG_node_ID[env.robot.handle] = ID

    # Fetch room for this agent
    room_name = None
    for region in env.sim.semantic_scene.regions:
        if region.contains(agent.additional_info["position"]):
            room_name = sim_region_id_to_SG_node_ID[region.id]
            break

    # Add agent to unknown room if a valid room is not found
    if room_name == None:
        graph.add_edge(
            agent,
            sim_region_id_to_SG_node_ID["unknown_room"],
            RelationType.INSIDE_ROOM,
            flip_edge(RelationType.INSIDE_ROOM),
        )
    else:
        # Add edge between the agent and room
        graph.add_edge(
            agent,
            room_name,
            RelationType.INSIDE_ROOM,
            flip_edge(RelationType.INSIDE_ROOM),
        )


def set_object_properties_and_initial_states(
    graph: Graph,
    env: Hab_Environment,
    metadata_interface: MetadataInterface,
    sim_handle_to_SG_node_ID: dict[str, str],
):
    objs = sutils.get_all_objects(env.sim)

    need_faucet_to_clean = set()
    for obj in objs:
        obj_node_ID = sim_handle_to_SG_node_ID[obj.handle]
        node = graph.get_node_from_ID(obj_node_ID)
        metadata_category = metadata_interface.get_object_instance_category(obj)
        if metadata_category in metadata_interface.affordance_info["turned on or off"]:
            node.set_property(PropertyType.HAS_POWER)

        if (
            metadata_category
            in metadata_interface.affordance_info["cleaned with a brush if dirty"]
        ):
            node.set_property(PropertyType.CAN_BE_CLEANED)
        if (
            metadata_category
            in metadata_interface.affordance_info["cleaned under a faucet if dirty"]
        ):
            node.set_property(PropertyType.CAN_BE_CLEANED)
            need_faucet_to_clean.add(node.category)

        if metadata_category in metadata_interface.affordance_info["filled with water"]:
            node.set_property(PropertyType.CAN_BE_FILLED_WITH_LIQUID)

        if obj.is_articulated:
            if any(
                obj.get_link_joint_type(i)
                in [
                    habitat_sim.physics.JointType.Revolute,
                    habitat_sim.physics.JointType.Prismatic,
                ]
                for i in range(obj.num_links)
            ):
                node.set_property(PropertyType.OPENABLE)

                if node.ID == "robot_agent":
                    continue

                opened_links = []
                for link_idx in range(obj.num_links):
                    if obj.get_link_joint_type(link_idx) in [
                        habitat_sim.physics.JointType.Revolute,
                        habitat_sim.physics.JointType.Prismatic,
                    ]:
                        joint_pos_ix = obj.get_link_joint_pos_offset(link_idx)
                        limits = obj.joint_position_limits
                        if (
                            abs(
                                obj.joint_positions[joint_pos_ix]
                                - limits[1][joint_pos_ix]
                            )
                            < 0.01
                        ):
                            opened_links.append(link_idx)
                if len(opened_links):
                    node.add_state(
                        StateType.OPEN, {"opened_link_idxs": set(opened_links)}
                    )

    return need_faucet_to_clean


def get_receptacle_name(
    graph: Graph,
    env: Hab_Environment,
    object_handle: str,
    sim_handle_to_SG_node_ID: dict[str, str],
    sim_region_id_to_SG_node_ID: dict[str, str],
    rec_name_to_rec: Dict[str, Receptacle],
) -> Optional[tuple[str, str]]:

    # get the ManagedObject
    obj = sutils.get_obj_from_handle(env.sim, object_handle)
    # match the object to Receptacles
    rec_names, _confidence, info_string = sutils.get_obj_receptacle_and_confidence(
        env.sim, obj, rec_name_to_rec, island_index=env.largest_indoor_island_idx
    )
    if len(rec_names) == 0:
        return None, ""

    rec_name = rec_names[0]
    return rec_name, info_string


def update_object_relations(
    graph: Graph,
    env: Hab_Environment,
    sim_handle_to_SG_node_ID: dict[str, str],
    sim_region_id_to_SG_node_ID: dict[str, str],
    rev_receptacle_dict: dict[str, List],
    rec_name_to_rec: Dict[str, Receptacle],
    cache: Optional[dict],
    init: bool = False,
):
    objs = sutils.get_all_objects(env.sim)
    for obj in objs:
        obj_ID = sim_handle_to_SG_node_ID[obj.handle]
        if "agent" in obj_ID:
            continue
        obj_pos = list(obj.translation)
        cached_pos = cache["object_position"].get(obj_ID, None)
        if cached_pos is None or cached_pos != obj_pos:
            if graph.has_edge("robot_agent", obj_ID, RelationType.GRASPING):
                continue

            graph.get_node_from_ID(obj_ID).update_info({"position": obj_pos})

            obj_handle = obj.handle
            rec_name, info_string = get_receptacle_name(
                graph,
                env,
                obj_handle,
                sim_handle_to_SG_node_ID,
                sim_region_id_to_SG_node_ID,
                rec_name_to_rec,
            )

            if rec_name is None:
                room_ID = sim_region_id_to_SG_node_ID["unknown_room"]
                graph.remove_all_edges(obj_ID)
                graph.add_edge(
                    obj_ID,
                    room_ID,
                    RelationType.INSIDE_ROOM,
                    flip_edge(RelationType.INSIDE_ROOM),
                )
            elif "floor" in rec_name:
                room_ID = get_room_ID(env.sim, obj_handle, sim_region_id_to_SG_node_ID)
                graph.remove_all_edges(obj_ID)
                graph.add_edge(
                    obj_ID,
                    room_ID,
                    RelationType.INSIDE_ROOM,
                    flip_edge(RelationType.INSIDE_ROOM),
                )
            elif info_string == "region match":
                room_ID = sim_region_id_to_SG_node_ID[rec_name]
                graph.remove_all_edges(obj_ID)
                graph.add_edge(
                    obj_ID,
                    room_ID,
                    RelationType.INSIDE_ROOM,
                    flip_edge(RelationType.INSIDE_ROOM),
                )
            else:
                assert rec_name in rec_name_to_rec, [
                    rec_name,
                    rec_name_to_rec.keys(),
                    obj_handle,
                ]
                parent_obj_handle, prop = rev_receptacle_dict[rec_name_to_rec[rec_name]]
                parent_obj_ID = sim_handle_to_SG_node_ID[parent_obj_handle]

                if prop == "on":
                    if obj_ID == parent_obj_ID:
                        pass
                    else:
                        assert (
                            graph.get_node_from_ID(parent_obj_ID).type
                            == EntityType.OBJECT
                        ), [parent_obj_ID, graph.get_node_from_ID(parent_obj_ID)]
                        if init:
                            graph.remove_all_edges(obj_ID)
                            graph.add_edge(
                                obj_ID,
                                parent_obj_ID,
                                RelationType.ON_RECEPTACLE,
                                flip_edge(RelationType.ON_RECEPTACLE),
                                {"rec_name": rec_name},
                            )

                        else:
                            pass
                elif prop == "within":
                    if obj_ID == parent_obj_ID:
                        pass
                    else:
                        assert (
                            graph.get_node_from_ID(parent_obj_ID).type
                            == EntityType.OBJECT
                        ), [parent_obj_ID, graph.get_node_from_ID(parent_obj_ID)]
                        if init:
                            graph.remove_all_edges(obj_ID)
                            graph.add_edge(
                                obj_ID,
                                parent_obj_ID,
                                RelationType.INSIDE_RECEPTACLE,
                                flip_edge(RelationType.INSIDE_RECEPTACLE),
                                {"rec_name": rec_name},
                            )

                        else:
                            assert graph.has_edge(
                                obj_ID, parent_obj_ID, RelationType.INSIDE_RECEPTACLE
                            ), [obj_ID, parent_obj_ID]
                else:
                    raise NotImplementedError

            cache["object_position"][obj_ID] = obj_pos
            cache["object_to_receptacle"][obj_ID] = rec_name


def update_agent_room_associations(
    graph: Graph, env: Hab_Environment, sim_region_id_to_SG_node_ID: dict[str, str]
):
    """
    This method will update the associations between agents and rooms.
    This is required because we need to update the graph every time
    the agents move in environment
    """
    agents: List[Entity] = graph.get_all_nodes_of_type(
        EntityType.HUMAN
    ) + graph.get_all_nodes_of_type(EntityType.ROBOT)
    for agent_node in agents:
        articulated_agent = env.robot
        current_pos = list(articulated_agent.translation)
        agent_node.update_info({"position": current_pos})

        # Get old room of the agent
        old_rooms = graph.get_neighbors_of_type(agent_node, EntityType.ROOM)

        # Make sure that its only one neighbor
        if len(old_rooms) != 1:
            raise ValueError(
                f"agent with name {agent_node.ID} was found to have more or less than one Rooms connected."
            )

        # Fetch new room for this agent
        new_room_ID = None
        for region in env.sim.semantic_scene.regions:
            if region.contains(agent_node.get_info("position")):
                new_room_ID = sim_region_id_to_SG_node_ID[region.id]
                break

        # It was found that sometimes, agent is not found to be in any room
        # In that case we skip changing its room
        if new_room_ID != None:
            # Delete edge between old room and agent
            graph.remove_edge(
                agent_node,
                old_rooms[0],
                RelationType.INSIDE_ROOM,
                flip_edge(RelationType.INSIDE_ROOM),
            )

            # Add edge between the agent and room
            graph.add_edge(
                agent_node,
                new_room_ID,
                RelationType.INSIDE_ROOM,
                flip_edge(RelationType.INSIDE_ROOM),
            )


def hab_sim_to_init_SG(env: Hab_Environment) -> Graph:

    graph = Graph()
    metadata_interface = MetadataInterface(env.hab_data_dir)
    add_house_to_graph(graph)
    node_idx = 20000
    node_idx, sim_region_id_to_SG_node_ID, idx2region_idx = add_rooms_to_graph(
        graph, env, node_idx
    )
    sim_handle_to_SG_node_ID, receptacle_dict, rec_name_to_rec = (
        add_receptacles_to_gt_graph(
            graph, env, metadata_interface, sim_region_id_to_SG_node_ID
        )
    )
    rev_receptacle_dict = {}
    for obj in receptacle_dict:
        for prop in receptacle_dict[obj]:
            for rec in receptacle_dict[obj][prop]:
                rev_receptacle_dict[rec] = [obj, prop]
    add_agents_to_graph(
        graph, env, sim_region_id_to_SG_node_ID, sim_handle_to_SG_node_ID
    )
    add_objects_to_graph(graph, env, metadata_interface, sim_handle_to_SG_node_ID)

    need_faucet_to_clean = set_object_properties_and_initial_states(
        graph, env, metadata_interface, sim_handle_to_SG_node_ID
    )

    cache = {"object_position": {}, "object_to_receptacle": {}}
    update_object_relations(
        graph,
        env,
        sim_handle_to_SG_node_ID,
        sim_region_id_to_SG_node_ID,
        rev_receptacle_dict,
        rec_name_to_rec,
        cache=cache,
        init=True,
    )
    update_agent_room_associations(graph, env, sim_region_id_to_SG_node_ID)

    cache["sim_handle_to_SG_node_ID"] = sim_handle_to_SG_node_ID
    cache["sim_region_id_to_SG_node_ID"] = sim_region_id_to_SG_node_ID
    cache["rev_receptacle_dict"] = rev_receptacle_dict
    cache["receptacle_dict"] = receptacle_dict
    cache["rec_name_to_rec"] = rec_name_to_rec
    cache["need_faucet_to_clean"] = need_faucet_to_clean
    cache["idx2region_idx"] = idx2region_idx

    return graph, cache


def hab_update_USG(env: Hab_Environment, usg: UnifiedSceneGraph) -> UnifiedSceneGraph:
    update_object_relations(
        usg.graph,
        env,
        usg.cache["sim_handle_to_SG_node_ID"],
        usg.cache["sim_region_id_to_SG_node_ID"],
        usg.cache["rev_receptacle_dict"],
        usg.cache["rec_name_to_rec"],
        cache=usg.cache,
    )
    update_agent_room_associations(
        usg.graph, env, usg.cache["sim_region_id_to_SG_node_ID"]
    )
    return usg
