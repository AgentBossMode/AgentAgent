from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.states.ReqAnalysis import ReqAnalysis
from final_code.utils.dict_to_reactflow import dict_to_tree_positions
from final_code.llms.model_factory import get_model
from final_code.states.NodesAndEdgesSchemas import JSONSchema
from copilotkit.langgraph import copilotkit_emit_state 
from copilotkit.langgraph import copilotkit_customize_config
from langchain_core.runnables import RunnableConfig
from final_code.prompt_lib.high_level_info.tooling import tooling_instructions
from final_code.prompt_lib.high_level_info.json_notes import json_notes
from final_code.prompt_lib.node_info.node_actions import node_action_types
from final_code.prompt_lib.examples.json_examples import json_example_ecommerce, json_example_marketing, json_example_report_finance

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

4. **Node LLM Action Algorithm**: For each node, provide detailed descriptions of what the LLM should do. Each action should include natural language instructions that a code-writing agent can interpret to implement the node's functionality:
    {node_action_types}
    Note: A single node can perform multiple actions (e.g., a node might `tool_call` to fetch data, then `analyze` it, then `generate` a response)

5. **Schema Construction Guidelines**:
   - Each node should specify exactly which GraphState fields it reads as input
   - Each node should specify exactly which GraphState fields it modifies/adds as output
   - Provide clear descriptions for input and output behaviors
   - Include comprehensive GraphState definition that covers all fields used across the workflow

6. Go through the DRY_RUNS, each dry_run should be satisfied by the langgraph workflow, think critically.

7. {json_notes}

8. DO NOT manipulate the py_code or composio related fields, they will be generated later

## JSON Schema Structure

The output should follow this exact structure:

```json
{{
  "graphstate": {{
    "type": "TypedDict",
    "fields": [
      {{
        "name": "[field_name]",
        "type": "[field_type]",
        "description": "[field_description]"
      }}
    ]
  }},
  "tools": [
    {{
      "name": "[tool_name]",
      "description": "[what_this_tool_does]",
      "node_ids": [
        "[node_id_that_uses_this_tool]"
      ]
    }}
  ],
  "nodes": [
    {{
      "id": "[node_id]",
      "function_name": "[function_name]",
      "description": "[node_description]",
      "llm_actions": "[list_of_actions]",
      "input_schema": [
        {{
          "name": "[field_name]",
          "type": "[field_type]",
          "description": "[how_this_field_is_used_as_input]"
        }}
      ],
      "output_schema": [
        {{
          "name": "[field_name]",
          "type": "[field_type]",
          "description": "[how_this_field_is_modified_or_created]"
        }}
      ]
    }}
  ],
  "edges": [
    {{
      "source": "[source_node_id]",
      "target": "[target_node_id]",
      "conditional": "[true|false]",
      "routing_conditions": "[conditions_if_conditional]"
    }}
  ],
  "justification": "[explanation_of_architectural_choice_and_workflow_design]"
}}

</INSTRUCTIONS>
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
    state["current_status"] = {"inProcess":True ,"status": "Generating JSON schema.."} 
    state["current_tab"] = "graph"
    await copilotkit_emit_state(config=modifiedConfig, state=state)
    json_extraction_llm = llm.with_structured_output(JSONSchema)
    json_extracted_output: JSONSchema = json_extraction_llm.invoke([HumanMessage(content=JSON_GEN_PROMPT.format(
        req_analysis=req_analysis.model_dump_json(indent=2),
        dry_runs=state["dry_runs"].model_dump_json(indent=2),
        tooling_instructions=tooling_instructions,
        json_notes=json_notes,
        node_action_types=node_action_types,
        json_example_ecommerce=json_example_ecommerce,
        json_example_marketing=json_example_marketing,
        json_example_report_finance=json_example_report_finance
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