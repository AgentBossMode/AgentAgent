from final_code.states.AgentBuilderState import AgentBuilderState, AgentInstructions
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Literal
from final_code.llms.model_factory import get_model
from langgraph.types import Command, interrupt

llm = get_model()

REQ_ANALYSIS_PROMPT = """Your job is to get information from a user about what kind of agent they wish to build.

You should get the following information from them:

- What the objective of the agent is
- Various usecases of the agent
- Some examples of what the agent will be doing (Input and expected output pairs)

If you are not able to discern this info, ask them to clarify, you can suggest and see if user confirms the suggestions.

After you are able to discern all the information, call the tool AgentInstruction"""

def requirement_analysis_node(state: AgentBuilderState) -> Command[Literal["requirement_analysis_node", "json_node"]]:
    """
    LangGraph node for performing requirement analysis.
    It interacts with the LLM to gather agent specifications from the user.
    If information is insufficient, it interrupts the graph for user input.
    Otherwise, it proceeds to the code generation node.
    """
    llm_with_tool = llm.bind_tools([AgentInstructions]) # Bind the AgentInstructions Pydantic model as a tool
    
    # Invoke the LLM with the system prompt and current message history
    response = llm_with_tool.invoke([SystemMessage(content=REQ_ANALYSIS_PROMPT)] + state["messages"])
    
    if not response.tool_calls:
        value = interrupt(response.content)
        return Command(goto="requirement_analysis_node", update={"messages": [response, HumanMessage(content=value)]})
        
    agent_instructions_args = response.tool_calls[0]["args"]
    agent_instructions = AgentInstructions(**agent_instructions_args)
    
    return Command(
        goto="json_node",
        update={"messages": [response], "agent_instructions": agent_instructions}
    )
