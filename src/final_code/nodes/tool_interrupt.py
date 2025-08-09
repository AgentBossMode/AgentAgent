from final_code.states.AgentBuilderState import AgentBuilderState, JSONSchema
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

def add_toolset(state: AgentBuilderState):
        
    tool_set = set()
    tool_set.add("Apollo")
    tool_set.add("Hubspot")
    tool_set.add("NOTION")
    return {"tool_set": tool_set}

def tool_interrupt(state: AgentBuilderState, config: RunnableConfig):
    confirmation = interrupt({
        "type": "toolset_integration",
        "tool_list" : state["tool_set"]})
        