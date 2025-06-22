from pydantic import BaseModel, Field
from typing import List, Optional

class Tool(BaseModel):
    name: str = Field(description="Name of the tool")
    description: str = Field(description="Description of the tool's functionality")
    is_composio_tool: bool = Field(default=False, description="Indicates if the tool is present in the composio toolset, default is False")
    composio_action_name: Optional[str] = Field(default=None, description="Name of the Composio action, if applicable, default is None")
    py_code: Optional[str] = Field(default=None, description="Python code for the tool, if applicable")

class ToolList(BaseModel):
    tool_required: bool = Field(default=False, description="Indicates if tools are required for the task")
    tools: Optional[List[Tool]] = Field(description="List of tools, each with a name and description, will be empty if tool_required is false")

class NodeSchema(BaseModel):
    id: str = Field(description="The node's identifier")
    schema_info: str = Field(description="A string describing the structure of the `GraphState` (e.g., \"GraphState:\\n type: TypedDict\\n fields:\\n - name: input\\n type: str...\"). You will need to parse this to define the `GraphState` TypedDict.")
    input_schema: str = Field(description="The expected input schema for the node (typically \"GraphState\").")
    output_schema: str = Field(description="The schema of the output produced by the node (typically \"GraphState\", indicating a partial update).")
    description: str = Field(description="Natural language description of what the node does")
    function_name: str = Field(description="The suggested Python function name for this node.")
    toolset: ToolList = Field(description="List of tools which should be binded to the langchain model which will be implemented in this node")

class EdgeSchema(BaseModel):
    source: str = Field(description="ID of the source node (or '__START__' for the graph's entry point)")
    target: str = Field(description="ID of the target node (or '__END__' for a graph termination point)")
    routing_conditions: str = Field(description="A natural language description of the condition under which this edge is taken, especially for conditional edges.")
    conditional: bool = Field(description="TA boolean flag, `true` if the edge is part of a conditional branch, `false` otherwise")

class JSONSchema(BaseModel):
    justification: str = Field(description="Identified architecture and justification of deciding the architecture")
    nodes: List[NodeSchema] = Field(description="List of nodes in the graph, each with its unique identifier and schema information")
    edges: List[EdgeSchema] = Field(description="List of edges in the graph, each describing a directed connection between nodes")
