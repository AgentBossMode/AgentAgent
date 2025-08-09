from final_code.states.NodesAndEdgesSchemas import JSONSchema
from langgraph.graph import MessagesState
from pydantic import Field
from copilotkit import CopilotKitState
class ToolBuilderState(CopilotKitState):
    """
    State for the graph that builds the agent.
    """
    json_schema: JSONSchema = Field(description="The JSON schema that will be used to build the agent. This is the schema that will be used to build the agent.")
