from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.states.ReqAnalysis import ReqAnalysis
from final_code.utils.dict_to_reactflow import dict_to_tree_positions
from final_code.llms.model_factory import get_model
from final_code.states.NodesAndEdgesSchemas import JSONSchema
from final_code.states.DryRunState import DryRunResults
from langchain_core.prompts import ChatPromptTemplate
from copilotkit.langgraph import copilotkit_emit_state 
from copilotkit.langgraph import copilotkit_customize_config
from langchain_core.runnables import RunnableConfig
from final_code.prompt_lib.high_level_info.tooling import tooling_instructions
from final_code.prompt_lib.high_level_info.json_notes import json_notes

llm = get_model()

JSON_GEN_PROMPT = PromptTemplate.from_template("""
You are tasked with generating a JSONSchema object which represents a langgraph workflow, Follow the 'INSTRUCTIONS' section to generate the JSONSchema object.:

<INPUT>
{req_analysis}
</INPUT>

<INSTRUCTIONS>
1.  **Identify Potential Architectures:** Consider if the described INPUT aligns with or would benefit from known advanced LangGraph architectures such as:
    * **Plan and Execute**: Does the INPUT imply an agent which might need a planning step (e.g., breaking down a complex task) followed by the execution of those plans by one or more action nodes?
    * **Agent Supervisor / Hierarchical Agent Teams**: Is the INPUT best served by a supervisor agent dispatching tasks to specialized worker agents, or a hierarchy of agents making decisions and delegating?
    * **Multi-Agent Collaboration (e.g., Swarm Architecture)**: Does the problem benefit from multiple agents working in parallel or collaboratively, perhaps sharing insights or contributing to a common goal?
    * **Reflection / Self-Correction (e.g., Self-Discover frameworks)**: Are there indications of iterative refinement, where results are evaluated and the process is adjusted?
    * **Human in the Loop (HITL)**: Does the `description` of any node, or the overall process, imply a need for human review, approval, correction, or explicit input at specific stages (e.g., before executing a critical action, when confidence is low, or for subjective assessments)?

2.  **Architectural Decision:**
    * If you determine that one or more of these architectures are strongly applicable to the INPUT, choose to implement it.
    * If no specific advanced architecture seems directly applicable for the given INPUT, proceed with a standard stateful graph construction based on the explicit langgraph nodes and edges.
3.  **Populating the tools field":
<tooling_instructions>
    {tooling_instructions}
</tooling_instructions>
4. Go through the dry runs, each dry run should be satisfied by the langgraph workflow, think critically.
5. {json_notes}
6. DONOT manipulate the py_code or composio related fields, they will be generated later.
</INSTRUCTIONS>
""")


async def  json_node(state: AgentBuilderState, config: RunnableConfig):
    req_analysis: ReqAnalysis = state["req_analysis"]
    
    # Invoke LLM to generate code based on the detailed prompt and instructions
    modifiedConfig = copilotkit_customize_config(
        config,
        emit_messages=False
    )
    state["current_status"] = {"inProcess":True ,"status": "Generating JSON schema.."} 
    state["current_tab"] = "graph"
    await copilotkit_emit_state(config=modifiedConfig, state=state)
    json_extraction_llm = llm.with_structured_output(JSONSchema)
    json_extracted_output: JSONSchema = json_extraction_llm.invoke([HumanMessage(content=JSON_GEN_PROMPT.format(
        req_analysis=req_analysis,
        tooling_instructions=tooling_instructions,
        json_notes=json_notes
    ))], config=modifiedConfig)
    state["current_status"] = {"inProcess":False ,"status": "JSON schema generated successfully"} 
    await copilotkit_emit_state(config=modifiedConfig, state=state)
    reactflow_json = dict_to_tree_positions(json_extracted_output.nodes, json_extracted_output.edges)
    # Return the generated Python code and an AI message
    return {
        "json_schema": json_extracted_output,
        "json_dict": reactflow_json,
        "justification": json_extracted_output.justification,
        "reactflow_json": reactflow_json,
        "current_tab": "graph",
        "messages": [AIMessage(content="Workflow schema has been successfully generated!")]
    }


async def dry_run_node(state: AgentBuilderState, config: RunnableConfig):
    modifiedConfig = copilotkit_customize_config(
        config,
        emit_messages=False
    )
    json_schema: JSONSchema = state["json_schema"]
    SYS_PROMPT= ChatPromptTemplate.from_template("""
Your job is to verify if the given langgraph workflow meets the specificied agent requirements.
You are given the json of a workflow graph below.
{json_schema}

You are also provided with the the functional requirements of the agent, which includes the objectives, usecases and examples.
<FUNCTIONAL_REQUIREMENTS>
{req_analysis}
</FUNCTIONAL_REQUIREMENTS>
                                                 
1. Analyze the FUNCTIONAL_REQUIREMENTS and the JSON schema of the langgraph workflow.
2. Go through the purpose, capabilities, knowledge_sources, targetted_users, toolings, dry_runs, and critically analyze the langgraph workflow.
4. If you find that the langgraph workflow does not satisfy the requirements, update the json schema to reflect the changes needed to satisfy the requirements.
6. Go through the dry runs, figure out if each dry_run is satisfied via the given langgraph workflow, think critically.
keys:
- name: The name of the use case
- description: The description of the use case
- dry_run: The dry run of the use case, explaining your critical reasoning that why this langgraph workflow does/does not satisfy the run. 
                                                 
If you find that the dry run fails in any of the cases, you should return the updated json_schema.
                                                 
A dry run should be a step by step flow of the langgraph workflow, explaining how the nodes and edges interact to achieve the objectives, usecases and examples.

{json_notes}
""")
    state["current_status"] = {"inProcess":True ,"status": "Analyzing the generated JSON schema against the functional requirements.."} 
    await copilotkit_emit_state(config=modifiedConfig, state=state)

    llm_with_struct = get_model().with_structured_output(DryRunResults)

    dry_run_analysis:DryRunResults = llm_with_struct.invoke([HumanMessage(content=SYS_PROMPT.format(json_schema=json_schema.model_dump_json(indent=2), req_analysis= state["req_analysis"], json_notes=json_notes))], config=modifiedConfig)
    state["current_status"] = {"inProcess":False ,"status": "Analysis completed, updating the JSON schema"} 
    await copilotkit_emit_state(config=modifiedConfig, state=state)
    
    if dry_run_analysis.updated_json_schema is not None:
        updated_json_schema: JSONSchema = dry_run_analysis.updated_json_schema
        reactflow_json = dict_to_tree_positions(updated_json_schema.nodes, updated_json_schema.edges)
        return {
            "json_schema": updated_json_schema,
            "use_cases": dry_run_analysis.use_cases,
            "reactflow_json": reactflow_json,
            "messages": [AIMessage(content="Re-evaluated the workflow against the user requirements and modified the workflow accordingly.")]
            }
    else:
        return {"use_cases": dry_run_analysis.use_cases}
