from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from composio_langgraph import Action, ComposioToolSet, App

composio_toolset = ComposioToolSet()

request = composio_toolset.initiate_connection(app=App.GITHUB)
print(f"Open this URL to authenticate: {request.redirectUrl}")


tools = composio_toolset.get_tools(
    apps=[App.GITHUB])
tool_node = ToolNode(tools)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0, streaming=True)
model_with_tools = model.bind_tools(tools)

def call_model(state: MessagesState):
    """
    Process messages through the LLM and return the response
    """
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge("__start__", "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
)
workflow.add_edge("tools", "agent")

app = workflow.compile()

for chunk in app.stream(
    {
        "messages": [
            (
                "human",
                "Star the GitHub Repository composiohq/composio",
            )
        ]
    },
    stream_mode="values",
):
    chunk["messages"][-1].pretty_print()
