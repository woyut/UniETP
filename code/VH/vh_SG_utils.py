from VH.vh_env import VH_Environment
from scene_graph.graph_schema import (
    Graph,
    Entity,
    EntityType,
    RelationType,
    ObjectAttributes,
    PropertyType,
    StateType,
    flip_edge,
)
from scene_graph.unified_scene_graph import UnifiedSceneGraph


def vh_event_to_SG(env: VH_Environment) -> tuple[UnifiedSceneGraph, dict]:
    graph = Graph()
    ret, vheg = env.comm.environment_graph()
    assert ret

    id2vheg_node = {}

    for vheg_node in vheg["nodes"]:
        id2vheg_node[vheg_node["id"]] = vheg_node
        if vheg_node["category"] == "Characters":
            assert len(graph.get_all_nodes_of_type(EntityType.ROBOT)) == 0
            node = Entity(
                ID="robot_agent",
                type=EntityType.ROBOT,
                category="agent",
                attributes=ObjectAttributes(),
                additional_info={
                    "position": vheg_node["obj_transform"]["position"],
                    "asset_name": vheg_node["prefab_name"],
                    "vh_category": vheg_node["category"],
                },
            )
        elif vheg_node["category"] == "Rooms":
            node = Entity(
                ID=f"{vheg_node['class_name']}_{vheg_node['id']}",
                type=EntityType.ROOM,
                category=vheg_node["class_name"],
                attributes=ObjectAttributes(),
                additional_info={
                    "position": vheg_node["obj_transform"]["position"],
                    "asset_name": vheg_node["prefab_name"],
                    "vh_category": vheg_node["category"],
                },
            )

        else:
            node = Entity(
                ID=f"{vheg_node['class_name']}_{vheg_node['id']}",
                type=EntityType.OBJECT,
                category=vheg_node["class_name"],
                attributes=ObjectAttributes(),
                additional_info={
                    "position": vheg_node["obj_transform"]["position"],
                    "asset_name": vheg_node["prefab_name"],
                    "vh_category": vheg_node["category"],
                },
            )

            for p in vheg_node["properties"]:
                if p == "CAN_OPEN":
                    node.set_property(PropertyType.OPENABLE)
                elif p == "GRABABLE":
                    node.set_property(PropertyType.GRASPABLE)
                elif p == "HAS_SWITCH":
                    node.set_property(PropertyType.HAS_POWER)
            for s in vheg_node["states"]:
                if s == "OPEN":
                    node.add_state(StateType.OPEN)
                elif s == "CLOSED":
                    if "CAN_OPEN" in vheg_node["properties"]:
                        node.add_state(StateType.CLOSED)
                elif s == "ON":
                    node.add_state(StateType.POWER_ON)
                elif s == "OFF":
                    node.add_state(StateType.POWER_OFF)
                else:
                    assert False, vheg_node

        graph.add_node(node)

    for vheg_edge in vheg["edges"]:
        from_vheg_node = id2vheg_node[vheg_edge["from_id"]]
        from_node_ID = (
            f"{from_vheg_node['class_name']}_{from_vheg_node['id']}"
            if from_vheg_node["class_name"] != "character"
            else "robot_agent"
        )
        to_vheg_node = id2vheg_node[vheg_edge["to_id"]]
        to_node_ID = f"{to_vheg_node['class_name']}_{to_vheg_node['id']}"
        if vheg_edge["relation_type"] == "ON":
            graph.add_edge(
                node1=from_node_ID,
                node2=to_node_ID,
                label=RelationType.ON_RECEPTACLE,
                opposite_label=flip_edge(RelationType.ON_RECEPTACLE),
            )
        elif vheg_edge["relation_type"] == "INSIDE":
            if to_vheg_node["category"] == "Rooms":
                graph.add_edge(
                    node1=from_node_ID,
                    node2=to_node_ID,
                    label=RelationType.INSIDE_ROOM,
                    opposite_label=flip_edge(RelationType.INSIDE_ROOM),
                )
            else:
                graph.add_edge(
                    node1=from_node_ID,
                    node2=to_node_ID,
                    label=RelationType.INSIDE_RECEPTACLE,
                    opposite_label=flip_edge(RelationType.INSIDE_RECEPTACLE),
                )
        elif vheg_edge["relation_type"] == "HOLDS_RH":
            assert (
                from_vheg_node["category"] == "Characters"
            )  # and to_vheg_node["category"] in ["Appliances", "Lamps", "Props", "Decor", "Electronics", "Foods", ], [vheg_edge, from_vheg_node, to_vheg_node]
            graph.add_edge(
                node1=from_node_ID,
                node2=to_node_ID,
                label=RelationType.GRASPING,
                opposite_label=flip_edge(RelationType.GRASPING),
            )
            graph.graph[graph.get_node_from_ID(from_node_ID)][
                graph.get_node_from_ID(to_node_ID)
            ].update_details("grasping_hand", "right")
        elif vheg_edge["relation_type"] == "HOLDS_LH":
            assert (
                from_vheg_node["category"] == "Characters"
            )  # and to_vheg_node["category"] in ["Appliances", "Lamps", "Props", "Decor", "Electronics", "Foods", ], [vheg_edge, from_vheg_node, to_vheg_node]
            graph.add_edge(
                node1=from_node_ID,
                node2=to_node_ID,
                label=RelationType.GRASPING,
                opposite_label=flip_edge(RelationType.GRASPING),
            )
            graph.graph[graph.get_node_from_ID(from_node_ID)][
                graph.get_node_from_ID(to_node_ID)
            ].update_details("grasping_hand", "left")

    return graph, {}
