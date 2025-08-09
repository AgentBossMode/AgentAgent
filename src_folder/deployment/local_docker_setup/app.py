from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage

def basic_node(state: MessagesState):
    return {"messages": [HumanMessage(content="Hello, how are you?")]}

def basic_node1(state: MessagesState):
    return {"messages": [HumanMessage(content="I am great, thank you!")]}

workflow = StateGraph(MessagesState)
workflow.add_node("basic_node", basic_node)
workflow.add_node("basic_node1", basic_node1)
workflow.add_edge(START, "basic_node")
workflow.add_edge("basic_node", "basic_node1")
workflow.add_edge("basic_node1", END)

app = workflow.compile()