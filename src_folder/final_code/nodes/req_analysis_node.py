from final_code.pydantic_models.AgentStatusList import AgentStatusList
from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.llms.model_factory import get_model
from langchain_core.messages import SystemMessage,AIMessage
from langgraph.types import Command, interrupt
from typing import Literal
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_customize_config
from final_code.prompt_lib.high_level_info.tooling import tooling_instructions
from final_code.prompt_lib.high_level_info.knowledge import knowledge_instructiona
from final_code.states.ReqAnalysis import ReqAnalysis, Purpose, Capabiity, Tool, DryRun, DryRuns
from final_code.utils.copilotkit_emit_status import append_in_progress_to_list, update_last_status, append_success_to_list_without_emit
from final_code.prompt_lib.high_level_info.get_json_info import get_json_info
import traceback


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

async def analyze_reqs(state: AgentBuilderState, config: RunnableConfig) -> Command[Literal["requirement_analysis_node", "__end__"]]:
    try:
        if "agent_status_list" in state:
            return Command(goto="__end__", update={"messages": [AIMessage(content="Promptius currently only supports building one agent at a time. Please start a new session to build another agent by clicking on the New Chat button")]}) 
        modifiedConfig = copilotkit_customize_config(
            config,
            emit_messages=False,
            emit_tool_calls=False
        )

        await append_in_progress_to_list(modifiedConfig, state, "Analyzing user requirements for agent building...")

        llm_req = llm.with_structured_output(ReqAnalysis)
        reqs_analysis: ReqAnalysis = await llm_req.ainvoke([SystemMessage(content=REQ_ANALYSIS_PROMPT.format(tooling_instructions=tooling_instructions, knowledge_instructiona=knowledge_instructiona))] +  state["messages"], config=modifiedConfig)
        
        await update_last_status(modifiedConfig, state, "Requirements analysis completed", True)
        
        return Command(goto="requirement_analysis_node", update={
            "req_analysis": reqs_analysis,
            "agent_status_list": state["agent_status_list"]
            })
    except Exception as e:
        return Command(
            goto="__end__",
            update={
                "exception_caught": f"{e}\n{traceback.format_exc()}",
                "messages": [AIMessage(content="An error occurred during analyzing requirements. Please try again.")]
            }
        )



async def requirement_analysis_node(state: AgentBuilderState, config: RunnableConfig) -> Command[Literal["generate_dry_run", "__end__"]]:
    """
    LangGraph node for performing requirement analysis.
    It interacts with the LLM to gather agent specifications from the user.
    If information is insufficient, it interrupts the graph for user input.
    Otherwise, it proceeds to the code generation node.
    """
    import json
    value_1: dict = json.loads(interrupt({"type":"req_analysis", "payload": state["req_analysis"] }))  
    try:
        #llm_with_tool = llm.bind_tools([AgentInstructions]) # Bind the AgentInstructions Pydantic model as a tool
        parsing_error = False
        msg = "Unknown error occurred. Please restart the process by providing a detailed input"
        try:
            value: ReqAnalysis = ReqAnalysis.model_validate(value_1)
            if value_1["approved"] == False:
                parsing_error = True
                msg = "requirements analysis suggestions rejected. Please restart the process by providing a detailed input"
        except Exception as e:
            parsing_error = True
            msg = "Unknown error occurred. Please restart the process by providing a detailed input"
        if parsing_error:   
            return Command(goto="__end__", update={"messages":[AIMessage(content=msg)]})

        req_analysis: ReqAnalysis = state["req_analysis"]
        # Filter purposes
        if value.purposes:
            req_analysis.purposes = [Purpose.model_validate(p) for p in value.purposes if p.selected]
        
        # Filter capabilities
        if value.capabilities:
            req_analysis.capabilities = [Capabiity.model_validate(c) for c in value.capabilities if c.selected]

        # Filter knowledge_sources
        # if value.knowledge_sources:
        #     req_analysis.knowledge_sources = [KnowledgeAndDataRequirements.model_validate(k) for k in value.knowledge_sources if k.selected]

        # Filter targetted_users
        # if value.targetted_users:
        #     req_analysis.targetted_users = [TargettedUser.model_validate(t) for t in value.targetted_users if t.selected]

        # Filter toolings
        if value.toolings:
            req_analysis.toolings = [Tool.model_validate(t) for t in value.toolings if t.selected]

        # Update additional_information if provided
        if value.additional_information:
            req_analysis.additional_information = value.additional_information

        req_analysis.user_selections = value.user_selections if value.user_selections else {}
        return Command(
            goto="generate_dry_run",
            update={ "req_analysis": req_analysis }
        )
    except Exception as e:
        return Command(
            goto="__end__",
            update={
                "exception_caught": f"{e}\n{traceback.format_exc()}",
                "messages": [AIMessage(content="An error occurred during processing requirement inputs. Please try again.")]
            }
        )

async def generate_dry_run(state: AgentBuilderState, config: RunnableConfig) -> Command[Literal["dry_run_interrupt", "__end__"]]:
    try:
        GENERATE_DRY_RUN_PROMPT = """
Generate dry runs for the agent based on the requirements analysis provided.
    The dry run should include:
    - Input information
    - A list of actions performed by the agent in the dry run
    - Output information

    Use the following requirements analysis as context:
    {req_analysis}

    This is the information regarding JSON schema which would be formed, you need to generate dry runs based on this:
    {json_info}

    Also check the user provided messages for any additional context.
    """
        modifiedConfig = copilotkit_customize_config(
            config,
            emit_messages=False,
            emit_tool_calls=False
        )
        req_analysis: ReqAnalysis = state["req_analysis"]
        messages = state["messages"]
        llm_dry_run = llm.with_structured_output(DryRuns)
        await append_in_progress_to_list(modifiedConfig, state, "Generating dry runs...")

        dry_runs: DryRuns = await llm_dry_run.ainvoke([
            SystemMessage(content=GENERATE_DRY_RUN_PROMPT
                          .format(
                              json_info = get_json_info(),
                              req_analysis=req_analysis.model_dump_json(indent=2)))]+ messages)

        await update_last_status(modifiedConfig, state, "Dry runs generated.", True)

        return Command(
            goto="dry_run_interrupt",
            update={
                "agent_status_list": state["agent_status_list"],
                  "dry_runs": dry_runs}
        )
    except Exception as e:
        return Command(
            goto="__end__",
            update={
                "exception_caught": f"{e}\n{traceback.format_exc()}",
                "messages": [AIMessage(content="An error occurred during generating dry runs. Please try again.")]
            }
        )

def dry_run_interrupt(state: AgentBuilderState, config: RunnableConfig) -> Command[Literal["json_node", "__end__"]]:
    """
    Interrupt handler for dry run generation.
    It collects the dry run information from the user and updates the state.
    """
    dry_runs: DryRuns = state["dry_runs"]
    import json
    value_1: dict = json.loads(interrupt({"type":"dry_runs", "payload": state["dry_runs"] }))
    try:
        try:
            dry_runs_1: DryRuns = DryRuns.model_validate(value_1)    
        except Exception:
            return Command(goto="__end__", update={"messages":[AIMessage(content="Dry_runs rejected. Please restart by providing detailed inputs")]})

        dry_runs.dry_runs = [DryRun.model_validate(d) for d in dry_runs_1.dry_runs if d.selected]
        dry_runs.user_selections = dry_runs_1.user_selections if dry_runs_1.user_selections else {}
        append_success_to_list_without_emit(state, "Dry runs collected successfully")
        return Command(goto="json_node", update={"dry_runs": dry_runs, "agent_status_list": state["agent_status_list"]})
    except Exception as e:
        return Command(
            goto="__end__",
            update={
                "exception_caught": f"{e}\n{traceback.format_exc()}",
                "messages": [AIMessage(content="An error occurred during collecting dry run inputs. Please try again.")]
            }
        )
