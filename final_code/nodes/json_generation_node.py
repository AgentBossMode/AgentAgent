from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from final_code.states.AgentBuilderState import AgentBuilderState, AgentInstructions
from final_code.utils.dict_to_reactflow import dict_to_tree_positions
from final_code.llms.model_factory import get_model
import json

llm = get_model()

JSON_GEN_PROMPT = PromptTemplate.from_template("""
You are tasked with generating a json, you've been given the below input:

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

3.  **Initial Comment:** At the very beginning of your generated Python script, include a comment block stating:
    * Which LangGraph architecture(s) (if any) you've identified and chosen to implement, with a brief justification based on your interpretation of the INPUT, provide dry runs of the usecases/examples.
    * If you are proceeding with a standard graph, mention that.

4. Generate a JSON representation of the architecture you have chosen, including:
a.  `nodes`: A dictionary where each key is a unique node ID. The value for each node ID is an object containing:
    * `id`: The node's identifier.
    * `schema_info`: A string describing the structure of the `GraphState` (e.g., "GraphState:\\n type: TypedDict\\n fields:\\n - name: input\\n type: str..."). You will need to parse this to define the `GraphState` TypedDict.
    * `input_schema`: The expected input schema for the node (typically "GraphState").
    * `output_schema`: The schema of the output produced by the node (typically "GraphState", indicating a partial update).
    * `description`: A natural language description of what the node does.
    * `function_name`: The suggested Python function name for this node.

b.  `edges`: A list of objects, each describing a directed edge in the graph. Each edge object contains:
    * `source`: The ID of the source node (or "__START__" for the graph's entry point).
    * `target`: The ID of the target node (or "__END__" for a graph termination point).
    * `routing_conditions`: A natural language description of the condition under which this edge is taken, especially for conditional edges.
    * `conditional`: A boolean flag, `true` if the edge is part of a conditional branch, `false` otherwise.
c. Final schema would look like:
{{
"nodes": [
  {{
    "id" : "<id of the node, similar to name>"
    "schema_info": "<define the class structure of the state of a node>",
    "input_schema": "<input state object>",
    "output_schema": "<output state object>",
    "description": "<description>",
    "function_name": "<function_name>",
  }},
  ....
],
"edges": [
    {{
      "source": "<id of the source node>",
      "target": "<id of the target node>",
      "routing_conditions": "If conditional is true, then explain what is the condition for routing",
      "conditional": <True or False>
    }},
    ....
]
}}
                                               
JSON should conform to section 4c.
Do not add escape characters unless absolutely necessary, if they are not for new line or tab, always add to backslash
Do not add words like input's say input\\'s
    """)

class JSONExtraction(BaseModel):
    justification: str = Field(description="Identified architecture and justification of deciding the architecture")
    json_output: str = Field(description="JSON representation of the architecture, this should be well formatted.")

def json_node(state: AgentBuilderState):
    instructions: AgentInstructions = state["agent_instructions"]
    
    # Invoke LLM to generate code based on the detailed prompt and instructions

    json_extraction_llm = llm.with_structured_output(JSONExtraction)
    json_extracted_output: JSONExtraction = json_extraction_llm.invoke([HumanMessage(content=JSON_GEN_PROMPT.format(
        objective=instructions.objective,
        usecases=instructions.usecases,
        examples=instructions.examples
    ))])
    reactflow_json = dict_to_tree_positions(json.loads(json_extracted_output.json_output))
    # Return the generated Python code and an AI message
    return {
        "messages": [AIMessage(content="Generated json schema!")],
        "json_dict": json_extracted_output.json_output,
        "justification": json_extracted_output.justification,
        "reactflow_json": reactflow_json
    }
