from langgraph.graph import  MessagesState
import operator
from pydantic import Field
from typing import List, Annotated, Tuple


class NodeBuilderState(MessagesState):
    """State for the node builder."""
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
    next: str = Field(description="Next node to go to")
    schema_info: str = Field(description="Schema information about the node")
    input_schema: str = Field(description="Input schema of the node")
    output_schema: str = Field(description="Output schema of the node")
    description: str = Field(description="Description of the node")
    function_name: str = Field(description="Function name of the node")
    code: str = Field(description="Code for the node")
    toolset: list[str] = Field(description="List of tools to be used in the node by the llm")
    node_type : str = Field(description="Type of node, deterministic if the function is deterministic and requires simple python code generation, ai if the function is not deterministic and requires a llm usage for meeting the requirements")
    node_info: str = Field(description="node information")
    task: str = Field(description="current task")
    final_code: str = Field(description="the final code to output")
