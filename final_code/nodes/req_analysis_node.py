from final_code.states.AgentBuilderState import AgentBuilderState, AgentInstructions
from langchain_core.messages import SystemMessage
from final_code.utils.copilotkit_interrupt_temp import copilotkit_interrupt
from typing import Literal
from final_code.llms.model_factory import get_model
from langgraph.types import Command
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_emit_state, copilotkit_customize_config

llm = get_model()

REQ_ANALYSIS_PROMPT = """Your job is to get information from a user about what kind of agent they wish to build.

You should get the following information from them:

- What the objective of the agent is
- Various usecases of the agent
- Some examples of what the agent will be doing (Input and expected output pairs)

If you are not able to discern this info, ask them to clarify, you can suggest and see if user confirms the suggestions.

After you are able to discern all the information, call the tool AgentInstruction"""

async def requirement_analysis_node(state: AgentBuilderState, config: RunnableConfig) -> Command[Literal["requirement_analysis_node", "json_node"]]:
    """
    LangGraph node for performing requirement analysis.
    It interacts with the LLM to gather agent specifications from the user.
    If information is insufficient, it interrupts the graph for user input.
    Otherwise, it proceeds to the code generation node.
    """
    modifiedConfig = copilotkit_customize_config(
        config,
    )
    state["current_status"] = {"inProcess":True ,"status": "Analyzing user requirements for agent building.."} 
    await copilotkit_emit_state(config=modifiedConfig, state=state)

    llm_with_tool = llm.bind_tools([AgentInstructions]) # Bind the AgentInstructions Pydantic model as a tool
    
    # Invoke the LLM with the system prompt and current message history
    response = llm_with_tool.invoke([SystemMessage(content=REQ_ANALYSIS_PROMPT)] + state["messages"])
    
    if not response.tool_calls:
        answer, new_messages = copilotkit_interrupt(message=response.content)
        return Command(goto="requirement_analysis_node", update={"messages": new_messages})
        
    agent_instructions_args = response.tool_calls[0]["args"]
    agent_instructions = AgentInstructions(**agent_instructions_args)
    
    return Command(
        goto="json_node",
        update={"messages": [AIMessage(content="Requirements have been identified")], "agent_instructions": agent_instructions}
    )
