from pydantic import BaseModel, Field
from typing import List, Optional

class Tool(BaseModel):
    name: str = Field(description="Name of the tool")
    description: str = Field(description="Description of the tool's functionality, the input to the node, and the output of the node")
    is_composio_tool: bool = Field(default=False, description="Indicates if the tool is present in the composio toolset, default is False")
    composio_toolkit_slug: Optional[str] = Field(default=None, description="The specific Composio toolkit SLUG, ex: Apollo, Hubspot and so on, if applicable, default is None")
    composio_tool_slug: Optional[str] = Field(default=None, description="The specific Composio tool SLUG, if applicable, default is None")
    py_code: Optional[str] = Field(default=None, description="Python code for the tool, if applicable")
    node_ids: List[str] = Field(description="The node ids of the nodes for which the tool is relevant. This is the id variable in NodeSchema")

class NodeSchema(BaseModel):
    id: str = Field(description="The node's identifier")
    schema_info: str = Field(description="A string describing the structure of the `GraphState` (e.g., \"GraphState:\\n type: TypedDict\\n fields:\\n - name: input\\n type: str...\"). You will need to parse this to define the `GraphState` TypedDict.")
    input_schema: str = Field(description="The expected input schema for the node (typically \"GraphState\").")
    output_schema: str = Field(description="The schema of the output produced by the node (typically \"GraphState\", indicating a partial update).")
    description: str = Field(description="Natural language description of what the node does")
    function_name: str = Field(description="The suggested Python function name for this node.")

class EdgeSchema(BaseModel):
    source: str = Field(description="ID of the source node (or '__START__' for the graph's entry point)")
    target: str = Field(description="ID of the target node (or '__END__' for a graph termination point)")
    routing_conditions: str = Field(description="A natural language description of the condition under which this edge is taken, especially for conditional edges.")
    conditional: bool = Field(description="A boolean flag, `true` if the edge is part of a conditional branch, `false` otherwise")

class JSONSchema(BaseModel):
    justification: str = Field(description="Identified architecture and justification of deciding the architecture")
    nodes: List[NodeSchema] = Field(description="List of nodes in the graph, each with its unique identifier and schema information")
    edges: List[EdgeSchema] = Field(description="List of edges in the graph, each describing a directed connection between nodes")
    tools: Optional[List[Tool]] = Field(description="The list of tools needed by different nodes")


def get_tools_info(tools: Optional[List[Tool]]) -> str:
    """
    Generate a string representation of the tools information.
    
    Args:
        tools (List[Tool]): List of Tool objects.
    
    Returns:
        str: A formatted string containing the tool information.
    """
    tools_info = ""
    for tool in tools:
        tools_info += f"tool_name: {tool.name}, tool_description: {tool.description}, is_composio_tool: {tool.is_composio_tool}, node_ids:[{", ".join(tool.node_ids)}]\n"
    return tools_info

def get_nodes_and_edges_info(json_schema: JSONSchema) -> str:
    json_schema_without_tools: JSONSchema = json_schema.model_copy(deep=True) # Create a deep copy to avoid modifying original state
    json_schema_without_tools.tools = [] # Clear the tools field in the copy
    return json_schema_without_tools.model_dump_json(indent=2)