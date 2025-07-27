from typing import List
from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.states.NodesAndEdgesSchemas import JSONSchema, Tool
from final_code.nodes.native_tool_builder import native_tool_builder
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

workflow = StateGraph(AgentBuilderState)
def get_composio_tools_node(state: AgentBuilderState):
    json_schema: JSONSchema = state["json_schema"]
    tool_list = []

    for tool in json_schema.tools:
        tool_list.append({
            "tool_name": tool.name,
            "tool_description": tool.description,
        }) 
    tools_selection_data = interrupt({
        "type": "select_composio_tools",
        "payload": tool_list
    }) 
    tool_set = set()
    for tool_selected in tools_selection_data["completed"]:
        tool_set.add(tool_selected["toolkit_slug"])

    tools_to_update: List[Tool] = json_schema.tools 

    for tool_selected in tools_selection_data["completed"]:
        for tool_to_update in tools_to_update:
            if tool_selected["tool_name"] == tool_to_update.name:
                tool_to_update.is_composio_tool = True
                tool_to_update.composio_toolkit_slug = tool_selected["toolkit_slug"]
                tool_to_update.composio_tool_slug = tool_selected["tool_slug"]
                tool_to_update.py_code = None # Composio tools don't have py_code directly
                break


    json_schema.tools = tools_to_update
    print(json_schema.tools)
    return {"tool_set": tool_set, "json_schema": json_schema, "messages": "Composio tools selected successfully"}


async def process_non_composio_tools(state: AgentBuilderState):
    json_schema: JSONSchema = state["json_schema"]
    tool_list = ""

    for tool in json_schema.tools:
        if not tool.is_composio_tool:
            tool_list += f"tool_name: {tool.name}, tool_description: {tool.description}\n" 
    if tool_list == "":
        return 
    updated_json_schema = await native_tool_builder.ainvoke({"messages": [HumanMessage(content=tool_list)], "json_schema": json_schema})
    return {"json_schema": updated_json_schema["json_schema"], "messages": updated_json_schema["messages"]}


workflow.add_node("get_composio_tools", get_composio_tools_node)
workflow.add_node("process_non_composio_tools", process_non_composio_tools)

workflow.add_edge(START, "get_composio_tools")
workflow.add_edge("get_composio_tools", "process_non_composio_tools")
workflow.add_edge("process_non_composio_tools", END)

tool_graph = workflow.compile()