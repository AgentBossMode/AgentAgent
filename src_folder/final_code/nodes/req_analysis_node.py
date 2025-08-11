from final_code.states.AgentBuilderState import AgentBuilderState, AgentInstructions
from final_code.llms.model_factory import get_model

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Command, interrupt
from typing import Literal
from langgraph.types import Command
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_emit_state, copilotkit_customize_config
from final_code.states.ReqAnalysis import ReqAnalysis
from final_code.prompt_lib.high_level_info.tooling import tooling_instructions
from final_code.prompt_lib.high_level_info.knowledge import knowledge_instructiona
llm = get_model()




REQ_ANALYSIS_PROMPT = """
Follow 'INSTRUCTIONS' section to analyze the user input and generate a ReqAnalysis object.

<CONTEXT>
You are a sales executive, who is approached by a client. 
Your product is and AI agents and agentic workflows builder.

Your builder can:
1. Create agents or workflows that can help with automating tasks via natural language or via deterministic logic.
3. The agent built by your builder has access the most famous toolkits out there, like gmail, notion, salesforce, github etc. (the list is long)
4. The builder also can integrate your existing knowledge bases into the functioning of the agent, so it could be used when necessary.
You should get the following information from them:
</CONTEXT>

<INSTRUCTIONS>
1. Analyze the user input, understand what are they trying to do, now based on CONTEXT, make 5 suggestions for each of the following:
    purpose
    capabilities
    knowledge_sources : Follow 'knowledge_instructiona' section for this.
    targetted_users
    toolings: Follow 'tooling_instructions' section for this.
    dry_run
3. Based on user input, first see if any of the information provided falls into the above categories:
   EX 1: user's input indicates 2 knowledge sources, in the 5 knowledge_sources suggestions two of them should be what user provided and marked as confident, then you can make upto 3 suggestions which should be extrapolated based on the input and follows 'knowledge_instructions', but donot make repetitive suggestions.
   EX2 : If nothing from input indicates a knowledge source, then you can make upto 5 suggestions based on the CONTEXT and user message, should not be repetitive and follows 'knowledge_instructions'.
4. If any information provided by the user does not fall into categories defined in 1, add the information cleanly in the additional_information column.
5. There should not be overlapping suggestions in toolings and knowledge_sources, categorize them properly.
</INSTRUCTIONS>
<tooling_instructions>
    {tooling_instructions}
</tooling_instructions>
<knowledge_instructiona> 
    {knowledge_instructiona}
</knowledge_instructiona>

"""

async def analyze_reqs(state: AgentBuilderState, config: RunnableConfig) -> Command[Literal["requirement_analysis_node"]]:
    modifiedConfig = copilotkit_customize_config(
        config,
        emit_all=False
    )
    state["current_status"] = {"inProcess":True ,"status": "Analyzing user requirements for agent building.."} 
    await copilotkit_emit_state(config=modifiedConfig, state=state)
    llm_req = llm.with_structured_output(ReqAnalysis)
    reqs_analysis: ReqAnalysis = await llm_req.ainvoke([SystemMessage(content=REQ_ANALYSIS_PROMPT.format(tooling_instructions=tooling_instructions, knowledge_instructiona=knowledge_instructiona))] +  state["messages"], config=modifiedConfig)
    return Command(goto="requirement_analysis_node", update={"req_analysis": reqs_analysis, "current_status": {"inProcess":False ,"status": "Requirements analysis completed"}})



async def requirement_analysis_node(state: AgentBuilderState, config: RunnableConfig) -> Command[Literal["json_node"]]:
    """
    LangGraph node for performing requirement analysis.
    It interacts with the LLM to gather agent specifications from the user.
    If information is insufficient, it interrupts the graph for user input.
    Otherwise, it proceeds to the code generation node.
    """
    
    #llm_with_tool = llm.bind_tools([AgentInstructions]) # Bind the AgentInstructions Pydantic model as a tool
    value_1: dict = interrupt({"type":"req_analysis", "payload": state["req_analysis"] })
    value: ReqAnalysis = ReqAnalysis.model_validate(value_1)
    req_analysis: ReqAnalysis = state["req_analysis"]
    # Filter purposes
    if value.purposes:
        req_analysis.purposes = [p for p in value.purposes if p.selected]
    
    # Filter capabilities
    if value.capabilities:
        req_analysis.capabilities = [c for c in value.capabilities if c.selected]

    # Filter knowledge_sources
    if value.knowledge_sources:
        req_analysis.knowledge_sources = [k for k in value.knowledge_sources if k.selected]

    # Filter targetted_users
    if value.targetted_users:
        req_analysis.targetted_users = [t for t in value.targetted_users if t.selected]

    # Filter toolings
    if value.toolings:
        req_analysis.toolings = [t for t in value.toolings if t.selected]

    # Filter dry_runs
    if value.dry_runs:
        req_analysis.dry_runs = [d for d in value.dry_runs if d.selected]

    # Update additional_information if provided
    if value.additional_information:
        req_analysis.additional_information = value.additional_information

    print(req_analysis)
    return Command(
        goto="json_node",
        update={"messages": [AIMessage(content="Requirements have been identified")], "req_analysis": req_analysis}
    )
