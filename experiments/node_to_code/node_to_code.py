from pydantic import BaseModel, Field
from typing import List, Annotated, Tuple
from langgraph.graph import  MessagesState, START, END, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from experiments.model_factory import get_model
from .node_to_code_interrupt import interrupt_generation
from .node_to_code_planner import plan_step
from .node_to_code_prompt import prompt_generation
from .node_to_code_replan import replan_step
from .node_to_code_struct import structured_output_generation
from .node_to_code_sup import make_supervisor_node
from .node_to_code_toolset import toolset_generation
from .node_to_code_base import NodeBuilderState

# Choose the LLM that will drive the agent
llm = get_model()

class NodeType(BaseModel):
    node_type: Literal["deterministic", "planner"] = Field(description=
                                                      """Type of node, 
                                                      : deterministic if the function is deterministic and requires simple python code generation,
                                                      : planner if the function is not deterministic, like analysis of input, plan generation or any other thing which is fuzzy logic requires Artificial intelligence for meeting the requirements""")


def determine_node_type(state: NodeBuilderState):
    """Determine the type of node."""
    node_type: str = state["node_type"]
    if node_type == "deterministic":
        return "deterministic_code"
    elif node_type == "planner":
        return "planner"

node_info_prompt= """
You are provided with the following information about the node:
<SchemaInfo>
{schema_info}
</SchemaInfo>
<InputSchema>
{input_schema}
</InputSchema>
<OutputSchema>
{output_schema}
</OutputSchema>
<Description>
{description}
</Description>
<FunctionName>
{function_name}
</FunctionName>

Below is the skeleton of the function that you need to implement:
def {function_name}(state:{input_schema}) -> {output_schema}:
    \"\"\"{description}\"\"\"
    # Implement the function to meet the description.
    
the state is of type {input_schema} and the function is of type {output_schema}
The general idea is that the implementation would involve extracting the input from the state, and updating the state with the output. Description contains the logic for this blackbox
"""
    
def identify_node(state: NodeBuilderState):
    """Identify the node and return the information."""
    # Extract the information from the state
    llm_with_structured_output = llm.with_structured_output(NodeType)
    node_type_identifier_prompt = """
        You are an expert in identifying whether a function definition could be handled via simple algorithmic logic, or requires the use of large language model aka LLMs 
        
        Examples:
        Q: The node's job is to generate a plan given user query.
        A: planner
        
        Q: Node adds two numbers
        A: determinisitc
        
        Q: Node needs to analyze user query and classify sentiment
        A: planner
        
        Q: Node performs function that needs critical thinking or fuzzy logic
        A: planner
        """
    type_of_node = llm_with_structured_output.invoke([SystemMessage(content=node_type_identifier_prompt)]+ [HumanMessage(content = node_info_prompt.format(
        schema_info=state["schema_info"],
        input_schema=state["input_schema"],
        output_schema=state["output_schema"],
        description=state["description"],
        function_name=state["function_name"]))])
    
    return {"node_type": type_of_node.node_type, "messages": [HumanMessage(content=node_info_prompt.format(
        schema_info=state["schema_info"],
        input_schema=state["input_schema"],
        output_schema=state["output_schema"],
        description=state["description"],
        function_name=state["function_name"])) ], 
            "node_info": node_info_prompt.format(
        schema_info=state["schema_info"],
        input_schema=state["input_schema"],
        output_schema=state["output_schema"],
        description=state["description"],
        function_name=state["function_name"])}

def deterministic_code_gen(state: NodeBuilderState):
    """Generate the code for the node."""
    prompt = ChatPromptTemplate.from_template(
        """Generate the python code for the function {function_name}.
You are provided with the following information about the node:
The schema information is as follows: {schema_info}
The input schema is: {input_schema}
The output schema is: {output_schema}
The description of the function is: {description}

Implement the function to meet the requirements.
""")
    code = llm.invoke(prompt.format(
        schema_info=state["schema_info"],
        input_schema=state["input_schema"],
        output_schema=state["output_schema"],
        description=state["description"],
        function_name=state["function_name"]))
    return {"final_code": code.content}


code_compiler_prompt = """
    You are tasked with furnishing a final output code for the user provided node information. 
    
    Analyze the message history, you will find multiple code pieces
    You need to carefully merge all the generate code together to furnish a final output.
    
    Only write python code with comments as output, no other explanations
    """


def code_compiler(state: NodeBuilderState):
    """Generate the code for the node."""
    response = llm.invoke([SystemMessage(content=code_compiler_prompt)]  + state["messages"])
    return {"final_code": response.content}

tool_indetification_prompt = """
Analyze the given Python script and extract insights for all functions decorated with @tool. For each function, determine the following:
- Detailed Objective: Explain in 1-5 lines what the function is likely designed to do based on the function name and docstring. If the implementation is missing, infer its intended behavior.
- Function Name: Extract the name of the function.
- Inputs & Argument Types: Identify any required input arguments along with their expected data types. If the function does not have explicit inputs, infer possible arguments that might be required for full functionality.
- Output & Type: Describe the expected output of the function and specify its data type. If the return statement is missing, infer the possible output type.
Example Input:
@tool
def get_coolest_cities():
# Get a list of coolest cities
    return "nyc, sf"

@tool
def find_best_food_spots(location):
# Find top-rated food spots in a given location
    pass  # Implementation missing


Example Output:
- Function Name: get_coolest_cities
- Detailed Objective: Returns a predefined list of cities that are considered "cool," such as New York City (NYC) and San Francisco (SF).
- Inputs & Argument Types: None explicitly defined.
- Output & Type: Returns a str: "nyc, sf".
- Function Name: find_best_food_spots
- Detailed Objective: Likely intended to find top-rated food spots based on a given location. Without an implementation, the exact logic is unclear, but it may use APIs or predefined datasets to fetch recommendations.
- Inputs & Argument Types: location (str) 
– likely required to determine where to search for food spots.
- Output & Type: Possibly a list of recommended places (list[str] or dict with additional details).

"""

def identify_tools(state: NodeBuilderState):
    """identify the tools to be created"""
    response = llm.invoke([SystemMessage(content=code_compiler_prompt)]  + state["messages"])
    return {"final_code": response.content}

ai_node_gen_supervisor = make_supervisor_node(llm, ["prompt_generation", "toolset_generation", "structured_output_generation", "interrupt_generation"])

from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver


workflow = StateGraph(NodeBuilderState)
workflow.add_node("identify_node", identify_node)
workflow.add_node("deterministic_code", deterministic_code_gen)
workflow.add_node("replan", replan_step)
workflow.add_node("planner", plan_step)
workflow.add_node("ai_node_gen_supervisor", ai_node_gen_supervisor)
workflow.add_node("prompt_generation", prompt_generation)
workflow.add_node("toolset_generation", toolset_generation)
workflow.add_node("structured_output_generation", structured_output_generation)
workflow.add_node("interrupt_generation", interrupt_generation)
workflow.add_node("code_compiler", code_compiler)


def should_end(state: NodeBuilderState
):
    if "response" in state and state["response"]:
        return "code_compiler"
    else:
        return "ai_node_gen_supervisor"
    

workflow.add_edge(START, "identify_node")
workflow.add_conditional_edges("identify_node", determine_node_type, ["deterministic_code", "planner"])
workflow.add_edge("deterministic_code", END)
workflow.add_edge("planner", "ai_node_gen_supervisor")
workflow.add_conditional_edges(
    "replan",
    # Next, we pass in the function that will determine which node is called next.
    should_end,
    ["ai_node_gen_supervisor", "code_compiler"],
)
workflow.add_edge("code_compiler", END)

checkpointer = InMemorySaver()
node_to_code_app : CompiledStateGraph = workflow.compile(checkpointer=checkpointer)