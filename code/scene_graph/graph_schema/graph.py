#!/usr/bin/env python3
# Portions of this file are adapted from partnr-planner
# (https://github.com/facebookresearch/partnr-planner), in particular from
# habitat_llm/world_model/graph.py and habitat_llm/world_model/world_graph.py.
# Copyright (c) Meta Platforms, Inc. and affiliates. Licensed under the MIT License.
# The adapted code has been modified for this project.

import copy
import random
from typing import Dict, List, Union, Optional

import numpy as np

from scene_graph.graph_schema.entity import Entity, EntityType
from scene_graph.graph_schema.attribute import ObjectAttributes
from scene_graph.graph_schema.relation import Relations, RelationType


class Graph:
    """
    This class represents a Directed Acyclic Graph.
    """

    # Parameterized Constructor
    def __init__(self, graph: Optional[Dict[Entity, Dict[Entity, Relations]]] = None):
        # Create a graph to store different entities in the world
        # and their relations to one another
        if graph is None:
            graph: Optional[Dict[Entity, Dict[Entity, Relations]]] = {}
        self.graph = graph

    def __deepcopy__(self, memo):
        new_graph = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_graph

        new_graph.graph = {}

        for src_entity, nbrs in self.graph.items():
            new_src = copy.deepcopy(src_entity, memo)
            new_graph.graph[new_src] = {}
            for dst_entity, rel_set in nbrs.items():
                new_dst = copy.deepcopy(dst_entity, memo)
                new_graph.graph[new_src][new_dst] = copy.deepcopy(rel_set, memo)

        return new_graph

    def __copy__(self):
        cls = self.__class__.__new__(self.__class__)
        new_graph = {}
        for src, nbrs in self.graph.items():
            new_src = copy.copy(src)
            new_graph[new_src] = {}
            for dst, rel_set in nbrs.items():
                new_dst = copy.copy(dst)
                new_graph[new_src][new_dst] = copy.copy(rel_set)
        new_graph_obj = cls
        new_graph_obj.graph = new_graph
        return new_graph_obj

    def size(self):
        """
        This method returns the number of nodes in the graph
        """
        return len(self.graph)

    def is_empty(self):
        """
        This method tells if the graph is empty or not
        """
        return self.size() == 0

    def get_node_from_ID(self, node_ID: str) -> Entity:
        """
        This method returns the node with matching name
        """
        for node in self.graph:
            if node.ID == node_ID:
                return node
        return None

    def has_node(self, input_node):
        """
        This method checks if the graph contains given node
        """

        # Reason if the input is of type string
        if isinstance(input_node, str):
            return any(node.ID == input_node for node in self.graph)

        # Reason if the input is not string
        return input_node in self.graph

    def has_edge(self, node1, node2, edge_label: Optional[RelationType] = None):
        """
        This method checks if the graph contains edge between two nodes
        """

        if isinstance(node1, str):
            node1 = self.get_node_from_ID(node1)
        if isinstance(node2, str):
            node2 = self.get_node_from_ID(node2)
        return any(
            neighbor == node2 and (edge_label is None or edge.has(edge_label))
            for neighbor, edge in self.graph[node1].items()
        )

    def add_node(self, node):
        """
        This method adds a node to the world graph
        """
        if node not in self.graph:
            self.graph[node] = {}

    def add_edge(self, node1, node2, label, opposite_label, details=None, verbose=False):
        """
        This method adds edge between two nodes.
        opposite label represents semantically opposite relation.
        E.g. if label is "inside" its opposite lable should be "outside"
        """
        if details is None:
            details = {}
        # Fetch node1 if input type is string
        if isinstance(node1, str):
            node1 = self.get_node_from_ID(node1)

        # Fetch node2 if input type is string
        if isinstance(node2, str):
            node2 = self.get_node_from_ID(node2)

        # or only opposite_label
        if node1 in self.graph and node2 in self.graph:
            # Add directional edge from node1 to node2
            if node2 not in self.graph[node1]:
                self.graph[node1][node2] = Relations({label}, details)
            else:
                self.graph[node1][node2].add(label)
                for k, v in details.items():
                    self.graph[node1][node2].update_details(k, v)

            if node1 not in self.graph[node2]:
                self.graph[node2][node1] = Relations({opposite_label})
            else:
                self.graph[node2][node1].add(opposite_label)

    def remove_node(self, node):
        """
        This method removes node and corresponding edges from the graph.
        """

        # Fetch node if input type is string
        if isinstance(node, str):
            node = self.get_node_from_ID(node)

        # Delete the node and edges to it
        del self.graph[node]
        for edges in self.graph.values():
            if node in edges:
                del edges[node]

    def remove_edge(self, node1, node2, label, opposite_label=None):
        """
        This method removes edge between two nodes
        """

        # Fetch node1 if input type is string
        if isinstance(node1, str):
            node1 = self.get_node_from_ID(node1)

        # Fetch node2 if input type is string
        if isinstance(node2, str):
            node2 = self.get_node_from_ID(node2)

        if node1 in self.graph and node2 in self.graph:
            if node2 in self.graph[node1] and self.graph[node1][node2].has(label):
                self.graph[node1][node2].remove(label)
                if self.graph[node1][node2].empty():
                    del self.graph[node1][node2]
                if opposite_label:
                    self.graph[node2][node1].remove(opposite_label)
                    if self.graph[node2][node1].empty():
                        del self.graph[node2][node1]

    def remove_all_edges(self, node):
        """
        Remove all edges associated with a particular node.
        """

        # Fetch node if input type is string
        if isinstance(node, str):
            node = self.get_node_from_ID(node)

        # Throw if node is invalid
        if node not in self.graph:
            raise ValueError(f"{node} not present in the graph")

        # Clear the outgoing edges
        self.graph[node] = {}

        # Clear incoming edges
        for edges in self.graph.values():
            if node in edges:
                del edges[node]

    def pop_node(self, node):
        """
        This method pops node and corresponding edges from the graph.
        """
        # Fetch node if input type is string
        if isinstance(node, str):
            node = self.get_node_from_ID(node)

        # Pop the node
        popped_node = self.graph.pop(node)

        # Clean up the connections
        for edges in self.graph.values():
            if node in edges:
                del edges[node]

        return popped_node

    def get_all_node_IDs(self):
        """
        Method to retrieve list of all node IDs from the graph
        """
        # Find all nodes with matching type
        node_IDs = [node.ID for node in self.graph]

        if len(node_IDs) > 0:
            return node_IDs
        else:
            return None

    def get_all_nodes_of_type(self, class_type):
        """
        Method to retrieve all nodes of a specific class
        """
        # Find all nodes with matching type
        matching_nodes = [node for node in self.graph if node.type == class_type]

        return matching_nodes

    def get_all_nodes_of_categories(self, categories: List[str]):
        matching_nodes = [node for node in self.graph if node.category in categories]

        if not matching_nodes and categories:
            all_cats = set(node.category for node in self.graph)
            for cat in categories:
                similar = [
                    c
                    for c in all_cats
                    if cat.lower() == c.lower()
                    or cat.lower() in c.lower()
                    or c.lower() in cat.lower()
                ]
                if similar:
                    for node in self.graph:
                        if node.category in similar:
                            pass
        return matching_nodes

    def get_random_node_of_type(self, class_type):
        """
        Method to get a random node of a given type
        """
        # Find all nodes with matching type
        matching_nodes = self.get_all_nodes_of_type(class_type)

        if len(matching_nodes) > 0:
            return random.choice(matching_nodes)
        else:
            return None

    def remove_all_nodes_of_type(self, class_type):
        """
        Method to remove all nodes of a given type
        """
        # Find all nodes with matching type
        matching_nodes = self.get_all_nodes_of_type(class_type)

        # Remove them from the dictionary
        if matching_nodes is None:
            return
        for node in matching_nodes:
            self.remove_node(node)

        return

    def get_neighbors(self, node):
        """
        This method returns all neighbors of the current node
        """

        # Fetch node if input type is string
        if isinstance(node, str):
            node = self.get_node_from_ID(node)

        # Throw if node is invalid
        if node not in self.graph:
            raise ValueError(f"{node} not present in the graph")

        return [x for x in self.graph[node]]

    def get_neighbors_of_type(self, node, class_type: EntityType):
        """
        This method returns list of all neighbors
        of a node that have given class_type.
        """

        # Fetch node if input type is string
        if isinstance(node, str):
            node = self.get_node_from_ID(node)

        # Throw if node is invalid

        if node not in self.graph:
            raise ValueError(f"{node} not present in the graph")

        return [
            neighbor for neighbor in self.graph[node] if neighbor.type == class_type
        ]

    def get_neighbors_of_relation(
        self, node: Union[str, Entity], relation_type: RelationType
    ):
        """
        This method returns list of all neighbors
        of a node that have given class_type.
        """

        # Fetch node if input type is string
        if isinstance(node, str):
            node = self.get_node_from_ID(node)

        # Throw if node is invalid

        if node not in self.graph:
            raise ValueError(f"{node} not present in the graph")

        return [
            neighbor
            for neighbor in self.graph[node]
            if self.graph[node][neighbor].has(relation_type)
        ]

    def count_nodes_of_type(self, class_type):
        """
        This method returns count of all nodes of given type
        """

        count = 0
        for node in self.graph:
            if node.type == class_type:
                count += 1

        return count

    def display_flattened(self):
        """
        Method to print the flattened world graph
        """
        pass

    def display_hierarchy(self, file_handle=None):
        """
        Method to print the graph with hierarchical
        """

        # Call the recursive printing method
        self.dfs_traverse(
            self.get_node_from_ID("house"), set(), file_handle=file_handle
        )

        return

    def to_string(self, compact=False):
        """
        Method to convert graph into a string
        """

        # Call the recursive printing method
        out = self.dfs_traverse(self.get_node_from_ID("house"), set(), "", compact)

        return out

    def dfs_traverse(
        self, node, visited_nodes_set, out=None, compact=False, file_handle=None
    ):
        """
        Recursive method to print the graph with DFS.
        """
        # Early return if the node has already been printed
        if node in visited_nodes_set:
            return None

        # Add node to the visited list
        visited_nodes_set.add(node)

        # Iterate through all neighbors of the current node.
        for neighbor in sorted(self.graph[node]):
            # Skip if the neighbor has already been visited
            if neighbor in visited_nodes_set:
                continue

            # Print the node based on the class type
            if neighbor.type == EntityType.ROBOT:
                text = f"\tRobot: {neighbor.ID}"
                out = (
                    out + text + "\n" if out != None else print(text, file=file_handle)
                )

            elif neighbor.type == EntityType.HUMAN:
                text = f"\tHuman: {neighbor.ID}"
                out = (
                    out + text + "\n" if out != None else print(text, file=file_handle)
                )

            elif neighbor.type == EntityType.ROOM:
                text = f"Room: {neighbor.ID}"
                out = (
                    out + text + "\n" if out != None else print(text, file=file_handle)
                )

            elif neighbor.type == EntityType.OBJECT:
                text = f"\t\t\tObject: {neighbor.ID}"
                out = (
                    out + text + "\n" if out != None else print(text, file=file_handle)
                )

            else:
                raise ValueError("Unsupported node type")

            # Call this method recursively on the neighbor
            out = self.dfs_traverse(
                neighbor, visited_nodes_set, out, compact, file_handle=file_handle
            )

        return out

    def get_robot(self):
        """
        This method returns spot robot node
        """
        for node in self.graph:
            if node.type == EntityType.ROBOT:
                return node

        raise ValueError("World graph does not contain a node of type Robot")

    def get_human(self):
        """
        This method returns human node
        """
        for node in self.graph:
            if node.type == EntityType.HUMAN:
                return node

        raise ValueError("World graph does not contain a node of type Human")

    def get_agents(self):
        """
        This method returns all agent nodes
        """
        out = []
        for node in self.graph:
            if node.type in [EntityType.HUMAN, EntityType.ROBOT]:
                out.append(node)

        if len(out) == 0:
            raise ValueError(
                "World graph does not contain a node of type Human or Robot"
            )

        return out

    def get_room_for_entity(self, entity):
        """
        This method returns the room in which the given entity is
        """

        # Get nodes of type room
        room = self.get_neighbors_of_type(entity, EntityType.ROOM)

        if room is None or len(room) == 0:
            raise ValueError(f"No room found for entity {entity}")

        if len(room) > 1:
            raise ValueError(f"Multiple rooms found for entity {entity}")

        return room[0]

    def get_closest_object(
        self, obj_node: Entity, n: int, dist_threshold: float = 1.5
    ) -> List[Entity]:
        """
        This method returns n closest objects to the given object node
        """
        closest = sorted(
            [node for node in self.graph if node.type == EntityType.OBJECT],
            key=lambda x: np.linalg.norm(
                np.array(obj_node.additional_info["position"])
                - np.array(x.additional_info["position"])
            ),
        )[:n]
        within_threshold = [
            obj
            for obj in closest
            if np.linalg.norm(
                np.array(obj_node.additional_info["position"])
                - np.array(obj.additional_info["position"])
            )
            < dist_threshold
        ]
        return within_threshold

    def get_distance_to_agent(
        self, obj_nodes: List[Entity], agent_ID: str = "robot_agent"
    ) -> List[float]:
        agent_node = self.get_node_from_ID("robot_agent")
        return [
            np.linalg.norm(
                np.array(obj_node.additional_info["position"])
                - np.array(agent_node.additional_info["position"])
            )
            for obj_node in obj_nodes
        ]

    def is_object_with_human(self, obj):
        """
        This method checks if the object is connected to any agent
        """
        # Fetch node if input type is string
        if isinstance(obj, str):
            obj = self.get_node_from_ID(obj)

        return any(neighbor.type == EntityType.HUMAN for neighbor in self.graph[obj])

    def is_object_with_robot(self, obj):
        """
        This method checks if the object is connected to any agent
        """
        # Fetch node if input type is string
        if isinstance(obj, str):
            obj = self.get_node_from_ID(obj)

        return any(neighbor.type == EntityType.ROBOT for neighbor in self.graph[obj])

    def is_object_with_agent(self, obj, agent_type="any"):
        """
        This method checks if the object is connected to any agent
        """
        # Fetch node if input type is string
        if isinstance(obj, str):
            obj = self.get_node_from_ID(obj)
        return_dict = {
            "any": any(
                neighbor.type in [EntityType.ROBOT, EntityType.HUMAN]
                for neighbor in self.graph[obj]
            ),
            "human": any(
                neighbor.type == EntityType.HUMAN for neighbor in self.graph[obj]
            ),
            "robot": any(
                neighbor.type == EntityType.ROBOT for neighbor in self.graph[obj]
            ),
        }
        if agent_type in return_dict:
            return return_dict[agent_type]
        else:
            raise ValueError(f"Agent type {agent_type} not recognized.")

    def find_path(
        self,
        root_node: Union[str, Entity] = "house",
        end_node_types: list = None,
        visited: set = None,
        verbose: bool = False,
    ) -> Optional[Dict[Entity, Dict[Entity, Relations]]]:
        """
        This method returns the path from the given node to the first node of type
        in end_node_types. It uses DFS to find the path.
        """
        if end_node_types is None:
            end_node_types = [EntityType.ROOM]
        if isinstance(root_node, str):
            root_node = self.get_node_from_ID(root_node)

        if root_node.type in end_node_types:
            return {}  # Return empty path if we are already at the end node

        if visited is None:
            visited = set()

        for neighbor, edges in self.graph[root_node].items():
            if neighbor not in visited:
                visited.add(neighbor)
                path = self.find_path(neighbor, end_node_types, visited)
                if path is not None:
                    if root_node in path:
                        path[root_node][neighbor] = edges
                    else:
                        path[root_node] = {neighbor: edges}
                    if neighbor in path:
                        path[neighbor][root_node] = self.graph[neighbor][root_node]
                    else:
                        path[neighbor] = {root_node: self.graph[neighbor][root_node]}
                    return path
        return None

    def get_subgraph(self, nodes_in, verbose: bool = False):
        """
        Method to get subgraph over objects in the view and agents.
        The relevant subgraph is considered the path from object to closest furniture,
        from agent to object-in-hand and from agent to the room they are in.

        Input is a list of name of entities in the agent's view. We sort through them and
        only keep objects. We then find a path from each object to the first Furniture node,
        which is called that object's relevant-subgraph. This relevant subgraph is then
        used to add/update objects in the world graph.
        """

        # Initialize empty subgraph
        subgraph = Graph()

        # Create root node
        house = Entity(
            ID="house",
            type="House",
            category="house",
            attributes=ObjectAttributes(),
            additional_info={"asset_name": "house_0"},
        )
        subgraph.add_node(house)

        # Create list of nodes if input is list of strings
        nodes = []
        for node in nodes_in:
            curr_node = self.get_node_from_ID(node) if isinstance(node, str) else node
            if curr_node.type in [
                EntityType.OBJECT,
                EntityType.ROBOT,
                EntityType.HUMAN,
            ]:
                nodes.append(curr_node)

        # add all required nodes in the subgraph
        for curr_node in nodes:
            subgraph.add_node(curr_node)

        # Loop through all object+agent nodes
        # and populate edges in the subgraph up to House
        for curr_node in nodes:
            path_graph = self.find_path(
                root_node=curr_node,
                end_node_types=EntityType.HOUSE,
                verbose=False,
            )

            if path_graph is not None:
                for curr_node in path_graph:
                    subgraph.add_node(curr_node)
                    for neighbor, edges in path_graph[curr_node].items():
                        if neighbor not in nodes:
                            subgraph.add_node(neighbor)
                        for rel in edges.relations:
                            subgraph.add_edge(
                                curr_node,
                                neighbor,
                                rel,
                                path_graph[neighbor][curr_node],
                            )

        return subgraph

    def to_dot(self):
        """
        Convert the graph to DOT format for visualization.
        """
        dot = "digraph {\n"
        for node in self.graph:
            for neighbor, edges in self.graph[node].items():
                dot += f'    "{node}" -> "{neighbor}" [label="{edges}"];\n'
        dot += "}"
        return dot
