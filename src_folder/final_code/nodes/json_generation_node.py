from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.states.ReqAnalysis import ReqAnalysis
from final_code.utils.dict_to_reactflow import dict_to_tree_positions
from final_code.llms.model_factory import get_model
from final_code.states.NodesAndEdgesSchemas import JSONSchema
from copilotkit.langgraph import copilotkit_customize_config
from langchain_core.runnables import RunnableConfig
from final_code.prompt_lib.high_level_info.get_json_info import get_json_info
from final_code.prompt_lib.examples.json_examples import json_example_ecommerce, json_example_marketing, json_example_report_finance
from final_code.utils.copilotkit_emit_status import append_in_progress_to_list, update_last_status

llm = get_model()


JSON_GEN_PROMPT = PromptTemplate.from_template("""
You are tasked with generating a JSONSchema object which represents a langgraph workflow. Follow the 'INSTRUCTIONS' section to generate the JSONSchema object.

<INPUT>
<REQUIREMENTS>
{req_analysis}
</REQUIREMENTS>

<DRY_RUNS>
{dry_runs}
</DRY_RUNS>
</INPUT>

{json_info}
                                               
<EXAMPLES>
Few-shot examples that can be helpful as reference. THESE EXAMPLES ARE TO ONLY BE USED AS REFERENCE
    1. {json_example_ecommerce}
    2. {json_example_marketing}
    3. {json_example_report_finance}
</EXAMPLES>
""")


async def json_node(state: AgentBuilderState, config: RunnableConfig):
    req_analysis: ReqAnalysis = state["req_analysis"]
    
    # Invoke LLM to generate code based on the detailed prompt and instructions
    modifiedConfig = copilotkit_customize_config(
        config,
        emit_messages=False)
    
    state["current_tab"] = "graph"
    await append_in_progress_to_list(modifiedConfig, state, "Generating JSON schema...")
    
    json_extraction_llm = llm.with_structured_output(JSONSchema)
    json_extracted_output: JSONSchema = json_extraction_llm.invoke([HumanMessage(content=JSON_GEN_PROMPT.format(
        req_analysis=req_analysis.model_dump_json(indent=2),
        dry_runs=state["dry_runs"].model_dump_json(indent=2),
        json_info = get_json_info(),
        json_example_ecommerce=json_example_ecommerce,
        json_example_marketing=json_example_marketing,
        json_example_report_finance=json_example_report_finance
    ))], config=modifiedConfig)

    await update_last_status(modifiedConfig, state, "JSON schema generated successfully", True)
    
    reactflow_json = dict_to_tree_positions(json_extracted_output.nodes, json_extracted_output.edges)
    # Return the generated Python code and an AI message
    return {
        "json_schema": json_extracted_output,
        "json_dict": reactflow_json,
        "justification": json_extracted_output.justification,
        "reactflow_json": reactflow_json,
        "current_tab": "graph",
        "agent_status_list": state["agent_status_list"]
    }