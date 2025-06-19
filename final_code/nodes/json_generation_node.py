from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from final_code.states.AgentBuilderState import AgentBuilderState, AgentInstructions
from final_code.utils.dict_to_reactflow import dict_to_tree_positions
from final_code.llms.model_factory import get_model
from final_code.states.NodesAndEdgesSchemas import JSONSchema
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List


llm = get_model()

JSON_GEN_PROMPT = PromptTemplate.from_template("""
You are tasked with generating a JSONSchema object, you've been given the below input:

<INPUT>
<OBJECTIVE>
{objective}
</OBJECTIVE>
<USECASES>
{usecases}
</USECASES>
<EXAMPLES>
{examples}
</EXAMPLES>
</INPUT>

---
1.  **Identify Potential Architectures:** Consider if the described INPUT aligns with or would benefit from known advanced LangGraph architectures such as:
    * **Plan and Execute**: Does the INPUT imply an agent which might need a planning step (e.g., breaking down a complex task) followed by the execution of those plans by one or more action nodes?
    * **Agent Supervisor / Hierarchical Agent Teams**: Is the INPUT best served by a supervisor agent dispatching tasks to specialized worker agents, or a hierarchy of agents making decisions and delegating?
    * **Multi-Agent Collaboration (e.g., Swarm Architecture)**: Does the problem benefit from multiple agents working in parallel or collaboratively, perhaps sharing insights or contributing to a common goal?
    * **Reflection / Self-Correction (e.g., Self-Discover frameworks)**: Are there indications of iterative refinement, where results are evaluated and the process is adjusted?
    * **Human in the Loop (HITL)**: Does the `description` of any node, or the overall process, imply a need for human review, approval, correction, or explicit input at specific stages (e.g., before executing a critical action, when confidence is low, or for subjective assessments)?

2.  **Architectural Decision:**
    * If you determine that one or more of these architectures are strongly applicable to the INPUT, choose to implement it.
    * If no specific advanced architecture seems directly applicable for the given INPUT, proceed with a standard stateful graph construction based on the explicit langgraph nodes and edges.
    * Does any node need real-time/external data or if it requires data from web or has something in it's functionality that can be made deterministic through an API call? → In that case the toolset_required should be set to true and the tools list should be populated, the is_composio_tool and py_code should be set to default values.
    """)


def json_node(state: AgentBuilderState):
    instructions: AgentInstructions = state["agent_instructions"]
    
    # Invoke LLM to generate code based on the detailed prompt and instructions

    json_extraction_llm = llm.with_structured_output(JSONSchema)
    json_extracted_output: JSONSchema = json_extraction_llm.invoke([HumanMessage(content=JSON_GEN_PROMPT.format(
        objective=instructions.objective,
        usecases=instructions.usecases,
        examples=instructions.examples
    ))])
    reactflow_json = dict_to_tree_positions(json_extracted_output.nodes, json_extracted_output.edges)
    # Return the generated Python code and an AI message
    return {
        "messages": [AIMessage(content="Generated json schema!")],
        "json_schema": json_extracted_output,
        "json_dict": reactflow_json,
        "justification": json_extracted_output.justification,
        "reactflow_json": reactflow_json
    }

class UseCaseAnalysis(BaseModel):
    name: str = Field(description="Name of the use case")
    description: str = Field(description="Description of the use case")
    dry_run: str = Field(description="Dry run of the use case, which is a string representation of the dry run results")

class DryRunResults(BaseModel):
    use_cases: List[UseCaseAnalysis] = Field(default_factory=list,description="List of use cases with their names, descriptions, and dry runs.")
    updated_json_schema: JSONSchema | None = Field(default=None, description="Updated JSON schema if dry run fails")
    justification: str | None = Field(default=None, description="Justification for updated JSON schema if applicable")

def dry_run_node(state: AgentBuilderState):
    json_schema: JSONSchema = state["json_schema"]
    SYS_PROMPT= ChatPromptTemplate.from_template("""
You are given the json of a workflow graph below.
{json_schema}

You are also provided with the following user information:
{agent_instructions}
You are supposed to write use cases for the graph.
You will also do dry run of the graph with the use cases.
The use cases should be in the format of a list of dictionaries.
Each dictionary should have the following
keys:
- name: The name of the use case
- description: The description of the use case
- dry_run: The dry run of the use case
If you find that the dry run fails in any of the cases, you should return the updated json_schema.
""")
    llm_with_struct = get_model().with_structured_output(DryRunResults)

    dry_run_analysis:DryRunResults = llm_with_struct.invoke([HumanMessage(content=SYS_PROMPT.format(json_schema=json_schema.model_dump_json(indent=2), agent_instructions=state["agent_instructions"].model_dump_json(indent=2)))])
    if dry_run_analysis.updated_json_schema is not None:
        updated_json_schema: JSONSchema = dry_run_analysis.updated_json_schema
        return {"json_schema": updated_json_schema}
    else:
        return        
