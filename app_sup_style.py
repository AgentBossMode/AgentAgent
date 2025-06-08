from dotenv import load_dotenv
load_dotenv()

import json
from langgraph.graph import StateGraph, START, END
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.types import Send
from final_code.utils.dict_to_reactflow import dict_to_tree_positions
from final_code.utils.fetch_docs import fetch_documents
from final_code.nodes_sup_style.phase1_json_dfs import compiler_graph
from final_code.llms.model_factory import get_model, ModelName
from final_code.nodes.req_analysis_node import requirement_analysis_node
from final_code.states.AgentBuilderState import AgentBuilderState, AgentInstructions
from final_code.states.ArchEvaluations import ArchEvaluationReport, ArchitectureEvaluationState, ArchEvaluationWithUrl


agent_architecture_urls = ["https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration",
 "https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor",
 "https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams",
 "https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute",
 "https://langchain-ai.github.io/langgraph/tutorials/self-discover/self-discover",
]

llm = get_model()

ARCH_EVALUATION_PROMPT = PromptTemplate.from_template(
    """
You are tasked with assessing the provided agentic architecture documentation to determine its relevance and applicability to the user requirements outlined below.

Inputs:

- Agentic Architecture Documentation:
{agent_architecture}

- User Requirements:- Objectives: {objective}
- Use Cases: {responsibilities}
- Examples: {examples}


Deliverables:
Your evaluation should be presented in the following structured format:

- Architecture Name:
Provide the name of the agent architecture being assessed.
- Concise Identifier:
Suggest a brief, descriptive name for the architecture that encapsulates its purpose.
- Key Highlights (2-3 Lines):
Identify and summarize the most significant aspects or features of the architecture.
- Feature Summary:
Offer a detailed overview of the architecture's unique elements, functionalities, and capabilities.
- Relevance Score (1-10):
Assign a score based on the alignment between the architecture's features and the user's requirements (1 = minimally relevant, 10 = highly relevant).
- Score Justification:
Provide a clear and concise rationale (5-10 sentences) for the relevance score, highlighting how specific features match—or fail to match—the user's objectives, use cases, and examples.
- Implementation Proposal:
Outline a tailored approach for leveraging the architecture to meet the user’s requirements. Be specific and actionable, addressing how it can fulfill the stated objectives and responsibilities.
"""
)

def architecture_evaluation_map_node(state: AgentBuilderState):
    return [Send("evaluate_against_architecture", {"agent_instructions": state["agent_instructions"], "url": url}) for url in agent_architecture_urls]

def route_state(state: AgentBuilderState):
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "add_tool_message"
    elif not isinstance(messages[-1], HumanMessage):
        return END
    return "requirement_analysis"

def evaluate_against_architecture(state: ArchitectureEvaluationState):
    agent_instructions: AgentInstructions = state["agent_instructions"]
    url: str = state["url"]
    llm = get_model(model_name=ModelName.GPT41MINI)
    llm_with_structured_output = llm.with_structured_output(ArchEvaluationReport)
    archEvaluationReport: ArchEvaluationReport = llm_with_structured_output.invoke(
        [HumanMessage(content=ARCH_EVALUATION_PROMPT.format(agent_architecture=fetch_documents(url),
                                                             objective=agent_instructions.objective,
                                                             responsibilities=agent_instructions.usecases,
                                                             examples = agent_instructions.examples))])
    
    return {
        "messages": [AIMessage(content=f"Evaluated architecture {url}, arch_name: {archEvaluationReport.name}")],
        "arch_evaluation_reports": [ArchEvaluationWithUrl(url,archEvaluationReport)],
    }

def best_architecture(state: AgentBuilderState):
    """Select the best architecture based on the evaluation reports."""
    # Sort the architectures based on their evaluation scores
    arch_reports : List[ArchEvaluationWithUrl] = state["arch_evaluation_reports"]
    sorted_architectures = sorted(arch_reports, key=lambda x: x.report.evaluation_score , reverse=True)
    
    # Select the best architecture (the first one in the sorted list)
    best_architecture = sorted_architectures[0]
    
    print("found the best architecture")
    
    # Return the best architecture as the output
    return {
        "messages": [AIMessage(content="Best architecture selected!")],
        "best_agent_architecture": best_architecture,
    }


AGENT_KERNEL_PROMPT = PromptTemplate.from_template(
    """
        Task Overview:
        Design a langgraph StateGraph object implementing the {agent_architecture_name} architecture, tailored to fulfill the user requirements defined below

        <Requirements>
        Objectives: {objective}
        usecases: {responsibilities}
        examples: {examples}
        </Requirements>

        Provided Documentation:
        Refer to the documentation for {agent_architecture_name} below to guide your implementation:
        <Documentation for {agent_architecture_name}>
        {agent_architecture}
        </Documentation for {agent_architecture_name}>
        
        Implementation Suggestions:
        Apply the {agent_architecture_name} architecture concepts and align them with the user requirements provided. Suggestions on tailoring the implementation:
        {agent_tailored}
        
        Expected Output:
        Your task is to produce a compiled StateGraph object.

        Guidelines for Code Generation:
        - Accuracy: Avoid hallucinations or speculative assumptions when writing code. Refer exclusively to the provided documentation.
        - Understanding: Thoroughly comprehend the architecture and examples of code.
        - Customization: Generate code tailored specifically to meet the user requirements.

    """)

def agent_kernel_builder(state: AgentBuilderState):
    """Build the agent kernel using the best architecture."""
    best_architecture: ArchEvaluationWithUrl = state["best_agent_architecture"]
    agent_instructions : AgentInstructions = state["agent_instructions"]
    agent_architecture_url : str = best_architecture.url
    agent_architecture_report : ArchEvaluationReport = best_architecture.report

    llm = get_model(ModelName.GPT41MINI)
    response =  llm.invoke([HumanMessage(content=AGENT_KERNEL_PROMPT.format(
        objective=agent_instructions.objective,
        responsibilities=agent_instructions.usecases,
        examples = agent_instructions.examples,
        agent_tailored=agent_architecture_report.tailored_design,
        agent_architecture_name = agent_architecture_report.name,
        agent_architecture=fetch_documents(agent_architecture_url)))])
    
    # Return the generated agent kernel as the output
    return {
        "messages": [AIMessage(content="Generated agent kernel code!")],
        "python_code": response.content,
    }

CODE_TO_JSON_PROMPT = PromptTemplate.from_template("""
You are tasked with converting the following stategraph comPilation code into a JSON. 

The Input code is as follows:
{code_snippet}

OUTPUT: Explaination and JSON. Do not include any code blocks. Seperate the JSON and explaination blocks and ensure that there is an explaination for each line of JSON produced but keep the blocks seperated.
Each Output JSON will have a nodes sections containing all the nodes and an edges section

Please follow:
1. Produce the explaination first and then the JSON after it. DO not produce the JSON first. 
2. For any conditional edges, please include all the nodes that the source of a conditional edge can reach as part of the explaination.
3. Any Edge entry in the JSON can only be conditional(mention conditional: true) if the source for that edge acts as a source for multiple edges. If you cannot point to atleast 2 targets for 1 source, then that source will not have any conditional edges
4. A source can have any number of targets. Please write the explaination for each source node to target node edge
5. Please ensure that the JSON starts with __START__ node and __END__ node with the correct edges from and to them
6. Ensure all elements in the nodes sections of the output json contain the following fields: Schema_info, input_schema, output_schema, description, function_name. Please do not return any entries in the nodes without these fields and these fields can't be empty
7. Ensure all elements in the edges sections of the output json contain the following fields: source, target, routing_conditions, conditional. Please do not return any entries in the edges without these fields and they can't be empty
8. Every node should be a part of atleast one edge, Please ensure this is followed
9. Attach the code snippet for each node aswell that. Please extract it from the Input code


Example output JSON for a node:
                                             
   {{
        "id": "code_node"
        "schema_info": /"/"/"CodeWriterState:
        type: TypedDict
        fields:
        - name: user_query
          type: str
        - name: execution_result
            type: str/"/"/",
        "input_schema": "CodeWriterState",
        "output_schema":"RequiremenCodeWriterStatetAnalysisState",
        "description":"This node analyzes the user_query, if the query is to write a code, it will make a tool call to run the proposed code. This node returns command object",
        "function_name": "code_step"
    }}

Example output JSON for an edge:
edge:{{ source: "abc", target: "cde", routing_condition: "if abc made a tool call then go to cde", "conditional": true}}
edge:{{ source: "abc", target: "xyz", routing_condition: "if abc made an interupt to a human then go to xyz", "conditional": true}}
edge:{{ source: "xyz", target: "_END_", routing_condition: "no nodes to go after xyz, we have our final output for this path", "conditional": false}}


High level JSON format of the graph

{{
    "nodes": [
        {{ ... }},
        {{ ... }}
    ],
    "edges": [
        {{ ... }},
        {{ ... }}
    ]
}}
""")

def code_to_json_node(state: AgentBuilderState):
    """Convert the generated code to JSON."""
    json_code_ouptut = llm.invoke([HumanMessage(content=CODE_TO_JSON_PROMPT.format(
        code_snippet=state["python_code"]
        ))])
    
    # Return the JSON code as the output
    return {
        "messages": [AIMessage(content="Generated JSON code!")],
        "json_code": json_code_ouptut.content,
    }


JSON_CODE_COMBINE_PROMPT = PromptTemplate.from_template("""
You are tasked with verifying and updating the provided JSON that represents the nodes and edges of the langgraph to ensure it is correct with respect to the input code.

Contextual Documents for Understanding Code:
{documents}

Input Code:
<Input code>{code_snippet}</Input code>
JSON:
<JSON>{node_json}</JSON>

OUTPUT: JSON
Your task is divided into two parts:
- Validation: Verify that the JSON adheres to the rules outlined below.
- Correction and Augmentation: If the JSON is incorrect or incomplete, update it with a clear justification for every change, and include the code snippet for each node extracted from the input code.

Rules for Validation and Update:
- Conditional Edges:- An edge can be marked as conditional: true only if its source acts as a source for multiple edges (i.e., at least two targets).
- If the source does not meet this condition, then it cannot have conditional edges.

- Edge Targets:- Each source can have any number of targets. This flexibility must be maintained.

- Start and End Nodes:- The JSON must begin with the __START__ node and conclude with the __END__ node, with correct edges to and from them. Edges into the END node can also be conditional if they meet the above mentioned conditions      

- Node Structure:- Each node entry in the nodes section of the JSON must include the following non-empty fields:- schema_info
- id
- schema_info
- input_schema
- output_schema
- description
- function_name


- Edge Structure:- Each edge entry in the edges section of the JSON must include the following non-empty fields:- source
- target
- routing_conditions
- conditional


- Node-Edge Relationship:- Every node must be part of at least one edge. Ensure this relationship is consistently followed.

- Node Code Snippets:- Attach a code field to every node in the JSON, extracted directly from the input code.

- Schema Requirements:- The final JSON must conform to the schema provided below:


Schema for JSON. Ensure the following schema is followed. No field should be missing for any node or edge:
- Node Example:

{{
  "code_node": {{
    "id" : "<id of the node, similar to name>"
    "schema_info": "<define the class structure of the state of a node>",
    "input_schema": "<input state object>",
    "output_schema": "<output state object>",
    "description": "<description>",
    "function_name": "<function_name>",
    "code": "<python_code>"
  }}
}}


- Edge Example:

{{
  "edge": {{
    "source": "<source_node>",
    "target": "<target_node>",
    "routing_condition": "<routing_condition>",
    "conditional": true/false
  }}
}}


Key Instructions:
- Do not update any pre-existing field of the JSON unless you have an extremely strong justification for doing so.
- Clearly document the reasoning behind any additions, updates, or modifications to the JSON. Justifications should draw inspiration from the contextual documents mentioned earlier.
- Ensure conditional edges strictly adhere to the rules outlined above.
- Input_Schema and output_schemas can only have value None in JSON for START and END nodes. Please follow this without fail
- Include a code field for each code_node entry, with the exact code that corresponds to the node in the input code.
""")

json_formatter_doc = """
You will be given a json by the user, along with explanations and markdown, your target is to filter and only output the json schema.



Generate a JSON representation of the architecture you have chosen, including:
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
{
"nodes": [
  {
    "id" : "<id of the node, similar to name>"
    "schema_info": "<define the class structure of the state of a node>",
    "input_schema": "<input state object>",
    "output_schema": "<output state object>",
    "description": "<description>",
    "function_name": "<function_name>",
  },
  ....
],
"edges": [
    {
      "source": "<id of the source node>",
      "target": "<id of the target node>",
      "routing_conditions": "If conditional is true, then explain what is the condition for routing",
      "conditional": <True or False>
    },
    ....
]
}
                                               
JSON should conform to section c.

OUTPUT SHOULD ONLY BE THE DESIRED JSON, IT SHOULD NOT START WITH ```json nor end with ```, It SHOULD START WITH { and END WITH }
'"""


def json_better_node(state: AgentBuilderState):
    """Add code to the json flow"""
    langgraph_glossary_url = "https://langchain-ai.github.io/langgraph/concepts/low_level/"
    json_code_ouptut = llm.invoke([HumanMessage(content=JSON_CODE_COMBINE_PROMPT.format(
        code_snippet=state["python_code"],
        documents = fetch_documents(langgraph_glossary_url),
        node_json = state["json_code"]
        ))])

    formatted_json = llm.invoke([SystemMessage(content=json_formatter_doc)]+ [HumanMessage(content=json_code_ouptut.content)])
    reactflowjson = dict_to_tree_positions(json.loads(formatted_json.content))
    # Return the JSON code as the output
    return {
        "messages": [AIMessage(content="Generated updated JSON code!")],
        "json_code": formatted_json.content,
        "reactflow_json" : reactflowjson

    }

def json_node(state: AgentBuilderState):
    pass

workflow = StateGraph(AgentBuilderState)
workflow.add_node("requirement_analysis_node", requirement_analysis_node)
workflow.add_node("evaluate_against_architecture", evaluate_against_architecture)
workflow.add_node("best_architecture", best_architecture)
workflow.add_node("agent_kernel_builder", agent_kernel_builder)
workflow.add_node("code_to_json", code_to_json_node)
workflow.add_node("json_update", json_better_node)
workflow.add_node("json_to_code", compiler_graph)
workflow.add_node("json_node", json_node)

workflow.add_edge("json_to_code", END)
workflow.add_edge("json_update", "json_to_code")
workflow.add_edge("code_to_json", "json_update")
workflow.add_edge("agent_kernel_builder", "code_to_json")
workflow.add_edge("best_architecture", "agent_kernel_builder")
workflow.add_edge("evaluate_against_architecture", "best_architecture")
workflow.add_conditional_edges("json_node", architecture_evaluation_map_node,["evaluate_against_architecture"])
workflow.add_edge(START, "requirement_analysis_node")
app = workflow.compile()
        