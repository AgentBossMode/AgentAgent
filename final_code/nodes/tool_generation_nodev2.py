from final_code.nodes.tool_generation_node import ToolCollectorState
from final_code.states.NodesAndEdgesSchemas import JSONSchema
from final_code.nodes.composio_toolset_builder import composio_tool_builder
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

workflow = StateGraph(ToolCollectorState)
def get_composio_tools_node(state: ToolCollectorState):
    json_schema: JSONSchema = state["json_schema"]
    tool_list = ""

    for node in json_schema.nodes:
        if node.toolset.tool_required and node.toolset.tools:
            for tool in node.toolset.tools:
                    tool_list+= f"tool_name: {tool.name}, tool_description: {tool.description}\n"
    composio_tool_builder.invoke({"messages": [HumanMessage(content=tool_list)], "json_schema": json_schema})

workflow.add_node("get_composio_tools", get_composio_tools_node)

workflow.add_edge(START, "get_composio_tools")
workflow.add_edge("get_composio_tools", END)

app2 = workflow.compile()