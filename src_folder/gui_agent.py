"""
This is the main entry point for the agent.
It defines the workflow graph, state, tools, nodes and edges.
"""

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START
from copilotkit.langgraph import AIMessage, CopilotKitState, HumanMessage, SystemMessage, copilotkit_emit_state, copilotkit_customize_config
from promptius_gui_schema import PromptiusGuiSchema

class AgentState(CopilotKitState):
    """
    State for the GUI agent.
    """
    promptius_gui_schema: dict
    requires_gui: bool = False

model = ChatOpenAI(model="gpt-4o-mini")

async def generate_gui(state: AgentState, config: RunnableConfig) -> AgentState:
    modifiedConfig = copilotkit_customize_config(config, emit_messages=False)
    llm = model.with_structured_output(PromptiusGuiSchema)
    response: PromptiusGuiSchema = await llm.ainvoke([SystemMessage(content="You are a UI generator, you are required to generate UI, to represent the human input. Keep the styling compact, use grid when required. You need to ensure that the UI looks good, think like a graphic designer, The edges define the relationshop between the nodes which are the ui components, refer to the PromptiusGuiSchema"), HumanMessage(content=state["messages"][-1].content)], config=modifiedConfig)
    state["promptius_gui_schema"] = response.model_dump()
    await copilotkit_emit_state(config=modifiedConfig, state=state)
    return {
        "messages": [AIMessage(content="UI generated successfully")],
        "promptius_gui_schema": response.model_dump()
    }

workflow = StateGraph(AgentState)
workflow.add_node("generate_gui", generate_gui)

workflow.add_edge(START, "generate_gui")
workflow.add_edge("generate_gui", "__end__")

app = workflow.compile()