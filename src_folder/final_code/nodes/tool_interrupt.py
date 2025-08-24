from final_code.states.AgentBuilderState import AgentBuilderState
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from langchain_core.messages import AIMessage

def tool_interrupt(state: AgentBuilderState, config: RunnableConfig):
    confirmation = interrupt({
        "type": "toolset_integration",
        "tool_list" : state["tool_set"]})
    return {"messages": [AIMessage(content="Tools have been successfully authenticated.")]}
    
        