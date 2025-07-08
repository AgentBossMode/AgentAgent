from typing import List
from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.states.NodesAndEdgesSchemas import JSONSchema, Tool
from final_code.nodes.composio_toolset_builder import composio_tool_builder
from final_code.nodes.native_tool_builder import native_tool_builder
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

workflow = StateGraph(AgentBuilderState)
def get_composio_tools_node(state: AgentBuilderState):
    json_schema: JSONSchema = state["json_schema"]
    tool_list = ""

    for tool in json_schema.tools:
        tool_list+= f"tool_name: {tool.name}, tool_description: {tool.description}\n"
    if tool_list == "":
        return 
    updated_json_schema = composio_tool_builder.invoke({"messages": [HumanMessage(content=tool_list)], "json_schema": json_schema})

    #json_schema: JSONSchema = state['json_schema']
    final_tools: List[Tool] = updated_json_schema["json_schema"].tools
    # define a set 
    
    tool_set = set()
    for tool in final_tools:
        if tool.is_composio_tool:
            tool_set.add(tool.composio_toolkit_slug)
    return {"tool_set": tool_set, "json_schema": updated_json_schema["json_schema"], "messages": updated_json_schema["messages"]}


def process_non_composio_tools(state: AgentBuilderState):
    json_schema: JSONSchema = state["json_schema"]
    tool_list = ""

    for tool in json_schema.tools:
        if not tool.is_composio_tool:
            tool_list += f"tool_name: {tool.name}, tool_description: {tool.description}\n" 
    if tool_list == "":
        return 
    updated_json_schema = native_tool_builder.invoke({"messages": [HumanMessage(content=tool_list)], "json_schema": json_schema})
    return {"json_schema": updated_json_schema["json_schema"], "messages": updated_json_schema["messages"]}


workflow.add_node("get_composio_tools", get_composio_tools_node)
workflow.add_node("process_non_composio_tools", process_non_composio_tools)

workflow.add_edge(START, "get_composio_tools")
workflow.add_edge("get_composio_tools", "process_non_composio_tools")
workflow.add_edge("process_non_composio_tools", END)

tool_graph = workflow.compile()