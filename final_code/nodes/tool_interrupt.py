from final_code.states.AgentBuilderState import AgentBuilderState, JSONSchema
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt


def tool_interrupt(state: AgentBuilderState, config: RunnableConfig):
    json_schema: JSONSchema = state['json_schema']
    # define a set 
    tool_set = set()
    for tool in json_schema.tools:
        if tool.is_composio_tool:
            tool_set.add(tool.composio_toolkit_slug)
    confirmation = interrupt({"type": "toolset_integration", "tool_list" : tool_set})
    return
        