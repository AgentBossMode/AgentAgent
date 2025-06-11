from pydantic import BaseModel, Field

class NodeSchema(BaseModel):
    id: str = Field(description="Unique identifier for the node")
    schema_info: str = Field(description="Description of the structure of the GraphState")
    input_schema: str = Field(description="Expected input schema for the node")
    output_schema: str = Field(description="Schema of the output produced by the node")
    description: str = Field(description="Natural language description of what the node does")
    function_name: str = Field(description="Suggested Python function name for this node")

class EdgeSchema(BaseModel):
    source: str = Field(description="ID of the source node (or '__START__' for the graph's entry point)")
    target: str = Field(description="ID of the target node (or '__END__' for a graph termination point)")
    routing_conditions: str = Field(description="Condition under which this edge is taken, especially for conditional edges")
    conditional: bool = Field(description="True if the edge is part of a conditional branch, false otherwise")
