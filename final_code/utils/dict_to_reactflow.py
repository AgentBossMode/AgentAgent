import json
from final_code.states.NodesAndEdgesSchemas import NodeSchema, EdgeSchema
from typing import List
def dict_to_tree_positions(nodes: List[NodeSchema], edges: List[EdgeSchema]) -> str:
    reactflow = {"nodes": [], "edges": []}

    # Create a mapping of edges to track parent-child relationships
    children_map = {}
    for edge in edges:
        parent = edge.source
        child = edge.target
        if parent not in children_map:
            children_map[parent] = []
        children_map[parent].append(child)

    # Helper function to assign positions recursively
    def assign_positions(node_id, depth, x_offset, x_step, visited):
        # Prevent infinite recursion by checking if the node is already visited
        if node_id in visited:
            return
        visited.add(node_id)

        if node_id not in node_positions:
            node_positions[node_id] = {"x": x_offset, "y": depth * y_spacing}
            # Add node to ReactFlow nodes
            node = next((n for n in nodes if n.id == node_id), None)
            if node:
                reactflow["nodes"].append({
                    "id": node_id,
                    "data": {
                        "description": node.description,
                        "function_name": node.function_name,
                        "label": node_id.replace("_", " ").title()
                    },
                    "position": {"x": x_offset, "y": depth * y_spacing}
                })

        # Recurse to children
        if node_id in children_map:
            num_children = len(children_map[node_id])
            child_x_offset = x_offset - (x_step * (num_children - 1)) / 2
            for index, child_id in enumerate(children_map[node_id]):
                assign_positions(
                    child_id, depth + 1, child_x_offset + (index * x_step), x_step // 2, visited
                )

    # Root node position configuration
    y_spacing = 150  # Vertical distance between levels
    x_step = 400  # Horizontal distance between child nodes
    node_positions = {}

    # Assume __START__ is the root node
    visited_nodes = set()  # Track visited nodes to avoid infinite loops
    assign_positions("__START__", 0, 0, x_step, visited_nodes)

    # Add regular edges to ReactFlow
    for edge in edges:
        reactflow["edges"].append({
            "id": f'{edge.source}_to_{edge.target}',
            "source": edge.source,
            "target": edge.target,
            "animated": edge.conditional  # Set animated if conditional is True
        })

    # Convert to JSON with double quotes
    return json.dumps(reactflow, indent=2)
