from final_code.nodes.tool_generation_node import get_tool_description_node, ToolCollectorState, ToolDescriptionList
from final_code.llms.model_factory import get_model
from experiments.composio_testing.testing_composiotoolset import app
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage


llm = get_model()


workflow = StateGraph(ToolCollectorState)

def add_to_messages(state: ToolCollectorState):
    tool_descriptions_list: ToolDescriptionList = state["tool_descriptions_list"]
    tool_names_and_descriptions = "\n".join([f"tool_name: {tool.tool_name}, tool_description: {tool.description}" for tool in tool_descriptions_list.tools])
    return {
        "messages": [HumanMessage(content=tool_names_and_descriptions)]
    }

workflow.add_node("get_tool_descriptions", get_tool_description_node)
workflow.add_node("add_to_message", add_to_messages)
workflow.add_node("get_composio_tools", app)

workflow.add_edge(START, "get_tool_descriptions")
workflow.add_edge("get_tool_descriptions", "add_to_message")
workflow.add_edge("add_to_message", "get_composio_tools")
workflow.add_edge("get_composio_tools", END)

app2 = workflow.compile()