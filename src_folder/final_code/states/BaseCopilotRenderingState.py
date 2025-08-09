from copilotkit.langgraph import CopilotKitState
from pydantic import Field
from typing import List
from final_code.states.NodesAndEdgesSchemas import JSONSchema

class BaseCopilotRenderingState(CopilotKitState):
    # code files
    python_code: str = Field(description="python code")
    tools_code: str = Field(description="tools code")
    pytest_code: str = Field(description="pytest code")
    mock_tools_code: str = Field(description="The mock tools code generated for the agent")
    # console logs
    console_logs: List[str] = Field(description="console logs")
    console_logs_incoming: bool = Field(description="console logs incoming")
    # tab controls
    current_tab: str = Field(description="current tab")
    # json/graph
    reactflow_json: str = Field(description="The JSON representation of the ReactFlow graph for the agent")
    json_schema: JSONSchema = Field(description="The JSON schema of the agent's architecture, including nodes and edges")
    json_dict: str = Field(description="The JSON representation of the agent's architecture and nodes")
    # in chat status
    current_status: dict = Field(description="current status")
    