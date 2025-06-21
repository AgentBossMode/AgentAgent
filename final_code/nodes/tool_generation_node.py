import json 
from typing import List 
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field 
from langgraph.graph import MessagesState,StateGraph, START, END
from final_code.utils.fetch_docs import fetch_documents
from final_code.llms.model_factory import get_model
from langgraph.types import interrupt
from final_code.states.NodesAndEdgesSchemas import JSONSchema
from copilotkit import CopilotKitState

# --- LLM Initialization ---
# Initialize the Language Model (LLM) to be used throughout the application
llm = get_model()


# --- Tool/Function Definition Sub-Graph Components ---
# This section defines components for a sub-graph that creates individual Python functions (tools).

# Load tool/SDK information from JSON files.
# Note: Hardcoded paths might need adjustment depending on the deployment environment.
#try:
with open("./files/tools_link_json.json") as f:
    dict_tool_link = json.load(f)
with open("./files/tools_doc_json.json") as f:
    dict_tool_doc = json.load(f)
# except FileNotFoundError:
#     dict_tool_link = {} # Default to empty dict if file not found
#     dict_tool_doc = {}  # Default to empty dict if file not found


def lowercase_keys(input_dict: dict) -> dict:
    """
    Utility function to convert all keys in a dictionary to lowercase.
    """
    return {k.lower(): v for k, v in input_dict.items()}

# Normalize keys in the loaded tool dictionaries
dict_tool_link = lowercase_keys(dict_tool_link)
dict_tool_doc = lowercase_keys(dict_tool_doc)

# Prompt for initial analysis of a function's description
GET_FUNCTION_INFO_PROMPT = """You are an expert python developer. You will be given a description of a python function. 

Your job is to estimate and extract the following information:

- What exactly does this python do. What is the detailed objective of the function. Please write 1-5 lines
- Suggest or extract the name of the the function
- What would be the inputs/arguements required into this function to make it work. Please all mentioned the type of each input
- What would be output produced by this input. Please mention the output type 

Here is the description of the function you need to create:
<description>
{desc}
</description>
"""

class FunctionInstructions(BaseModel):
    """
    Pydantic model to structure the definition of a Python function (tool).
    Used for structured output from the LLM during function analysis and code generation.
    """
    objective: str = Field(description="what does this python function do")
    name: str = Field(description="name of the python function")
    input: List[str] = Field(description="what would be the input arguements to this function along with the types")
    output: List[str] = Field(description="what would be the output/return attributes for the function along with the types")
    name_toolkit: str = Field(description="what would be the toolkit/ code SDK that will be used") # Name of the SDK
    code: str = Field(description="the final python code for the function")
    user_comment: str = Field(description="User tells more about the tool/sdk/service they want to use to implement this tool.")

def functional_analysis_node(state: FunctionInstructions):
    """
    LangGraph node for analyzing a function description.
    It uses the LLM with structured output (FunctionInstructions) to parse the description.
    """
    llm_with_structured_output = llm.with_structured_output(FunctionInstructions)
    
    # Invoke LLM to get structured information about the function
    functionalReport: FunctionInstructions = llm_with_structured_output.invoke(
        [HumanMessage(content=GET_FUNCTION_INFO_PROMPT.format(desc=state.objective))]
    )
    return {
        "messages": [AIMessage(content="Generated JSON code for function analysis!")], # Consider a more descriptive message
        "objective": functionalReport.objective,
        "name": functionalReport.name,
        "input": functionalReport.input,
        "output": functionalReport.output,
        # name_toolkit and code are not set here, they are determined in subsequent nodes
    }


# Prompt for selecting the best SDK for a given function
IDENTIFY_BEST_SDK_PROMPT = """
You are a highly specialized language model designed to assist in selecting the most suitable SDK for a given use case. You are provided with the following:
- A dictionary containing pairs of SDK names and their respective descriptions.
- Requirements for a piece of code, including the objective, input, and output.
- You may also be provided with a user comment, directing what tool to use.

Your task is to:
- First check if the <USERCOMMENT> section is not empty, in that case prioritize whatever tool they mention, search in the <dictionary>, if not found then provide the next best sdk.
- Identify the SDK from the provided dictionary whose description best matches the given use case described in the code requirements.
- Also give preferences to SDKs that are generally more well known or are used more frequently in the industry (Use google tools for anything search related)
- Return only the name of the matching SDK without any additional text or formatting.
- Please ensure that the string you return is a valid key of the dictionary you get as input. PLEASE VERIFY THAT THE STRING YOU RETURN EXISTS AS A KEY IN THE INPUT DICTIONARY

Input Example:
Dictionary:
{{
"SDK_A": "[SDK_CC_ABC]Provides tools for web scraping and data extraction.",
"SDK_B": "Enables natural language processing for unstructured text.",
"SDK_C": "Facilitates the integration of payment gateways in applications."
}}
Code Requirements:
Objective: Extract data from multiple web pages.
Input: URLs of the web pages.
Output: Structured data in JSON format.

Expected Output:
SDK_A


Input :

<USERCOMMENT>
{user_comment}
</USERCOMMENT>

<dictionary>
{dictionary}
</dictionary>

<objective>
{objective}
</objective>

<input schema>
{inputs}
</input schema>

<output schema>
{output}
</output schema>

<name of function>
{name}
</name of function>
"""

def sdk_production_node(state: FunctionInstructions):
    """
    LangGraph node to identify the most suitable SDK for the function.
    It uses the LLM with the Best_sdk_prompt and the loaded SDK documentation.
    """
    objective_agent: str = state.objective
    name: str = state.name
    input_args: List[str] = state.input
    output_args: List[str] = state.output
    
    response = llm.invoke([HumanMessage(content=IDENTIFY_BEST_SDK_PROMPT.format(
        user_comment=state.user_comment,
        objective=objective_agent,
        inputs=input_args,
        output=output_args,
        name=name,
        dictionary=dict_tool_doc # Using the dictionary of SDK descriptions
    ))])
    
    sdk_name = response.content.lower().strip()
 
    return {
        "messages": [AIMessage(content=f"SDK '{sdk_name}' identified for function {name}.")],
        "name_toolkit": sdk_name
    }

# Prompt for writing the code of a function using a specific SDK
TOOL_CODE_WRITER_PROMPT = """
You are a skilled code generation assistant. Your task is to create executable code using the following information:
- SDK Documentation: The provided documentation outlines the functionalities and usage details of the SDK. Use this as the reference for constructing your code.
- Objective: A clear description of what the code is intended to achieve.
- Input: The expected input for the code (e.g., variables, parameters, data types).
- Output: The desired result or outcome of the code (e.g., format, type, or structure).
- SDK Name: The name of the SDK that must be used in the code.

Your goal is to generate executable code that:
- Adheres to the requirements outlined above.
- Follows standard coding practices and is optimized for readability and efficiency.
- Utilizes the specified SDK appropriately based on the documentation provided.
- Only return a self contained function
- Your output should only contain a code block containing the required function and nothing else. Please do no include any explainantions
- Write your code in python
- Please also provide which API keys will be required and define the API keys as part of the function
- Please also write the doc string for the python function
- Ensure that the function you produce is decorated with @tool. That means its defination should be preceeded by '@tool' in the line above

Here are some details about the python function you will be creating:
<objective>
{objective}
</objective>

<input schema>
{inputs}
</input schema>

<output schema>
{output}
</output schema>

<name of function>
{name}
</name of function>

Documentation for SDK that might be helpful:
<documentation>
{docs}
</documentation>
"""

def code_production_node(state: FunctionInstructions):
    """
    LangGraph node to generate the Python code for the function using the identified SDK.
    It fetches SDK documentation and uses the write_code_prompt.
    """
    objective_agent: str = state.objective
    name: str = state.name
    input_args: List[str] = state.input
    output_args: List[str] = state.output
    toolkit: str = state.name_toolkit
    
    docs = ""
    if toolkit in dict_tool_link:
        docs = fetch_documents(dict_tool_link[toolkit]) # Fetch documentation for the chosen SDK

    response = llm.invoke([HumanMessage(content=TOOL_CODE_WRITER_PROMPT.format(
        objective=objective_agent,
        inputs=input_args,
        output=output_args,
        name=name,
        docs=docs,
    ))])
    
    return {
        "messages": [AIMessage(content=f"Generated code for tool/function: {name}")],
        "code": response.content
    }

# --- Tool Information Graph Definition (`tool_infograph`) ---
# This graph orchestrates the creation of a single tool/function.
# It takes a function description and produces its Python code.

# Using InMemorySaver for this sub-graph as its state might not need to persist long-term
tool_info_workflow = StateGraph(FunctionInstructions) # Define state type for this graph

# Add nodes to the tool information graph
tool_info_workflow.add_node("func_analysis", functional_analysis_node)
tool_info_workflow.add_node("sdk_write", sdk_production_node)
tool_info_workflow.add_node("code_write", code_production_node)

# Define edges for the tool information graph
tool_info_workflow.add_edge(START, "func_analysis") # Start with functional analysis
tool_info_workflow.add_edge("func_analysis", "sdk_write") # Then identify SDK
tool_info_workflow.add_edge("sdk_write", "code_write")   # Then write code
tool_info_workflow.add_edge("code_write", END)          # End after code writing

# Compile the tool information graph
tool_infograph = tool_info_workflow.compile()

class ToolDescription(BaseModel):
    tool_name: str = Field(description="The name of the tool.")
    description: str = Field(description="A description of what the tool does.")

class ToolDescriptionList(BaseModel):
    """
    Pydantic model to structure a list of tool descriptions.
    Used for structured output from the LLM during tool identification.
    """
    tools: List[ToolDescription] = Field(description="List of tool descriptions identified in the code.")

class ToolCollectorState(CopilotKitState): # Renamed from 'toolcollector' for convention
    """
    State for the graph that collects and compiles multiple tool codes.
    """
    json_schema: JSONSchema = Field(description="The JSON schema of the graph, including nodes and edges.")
    python_code: str = Field(description="The main agent Python code, potentially with placeholders for tools.")
    total_code: List[str] = Field(default_factory=list, description="List of generated Python code snippets for tools.")
    compiled_code: str = Field(description="The main agent Python code, potentially with placeholders for tools.")
    json_dict: str = Field(description="The JSON representation of the graph")
    tool_descriptions_list: ToolDescriptionList = Field(description="List of tool descriptions identified in the code.")

# Prompt to compile main code with generated tool function definitions
tool_compile_prompt = """
You are python code writing expert. You are given 2 snippets of code, your job is to combine them. 
The first snippet of code contains a compilable code with some functions compilable but empty. 
The second snippet of code contains the defination of those functions. 
Please go through the second snippet of code, match the function in the first snippet and replace the functional definition written in the first snippet with one found in second snippet

Please only return compilable python code
Here are the code snippets:
<code_snippet1>
{complete_code}
</code_snippet1>
<code_snippet2>
{functions}
</code_snippet2>
"""


# Prompt to extract descriptions of functions decorated with @tool
GET_TOOL_DESCRIPTION_PROMPT = """
You are an AI assistant designed to analyze Python code. Your task is to identify all function definitions in the provided Python snippet that are decorated with @tool. You must return a dictionary where:
- The keys are the names of the identified functions.
- You only need to pick up a function if it is decorated with '@tool' or '@tool' just preceeds the function. Otherwise leave the function alone
- The values are descriptions of what each function is supposed to do. If a function contains a docstring, extract it as the description. If a docstring is missing, infer the function's purpose from its structure and comments.
- The output should just be a json. it should not include "```json" and "```" at the start or end of it. it should start with a "{{" and end with a "}}"
Example Input:
@tool
def calculate_area(length, width):
    "Calculates the area of a rectangle."
    return length * width

@tool
def greet(name):
    return f"Hello, {{name}}!"


Expected Output:
{{
    "calculate_area": "Calculates the area of a rectangle.",
    "greet": "Greets a user by name."
}}


Instructions:
- Identify functions that have the @tool decorator.
- Extract function names and descriptions (either from docstrings or inferred).
- Return the output as a structured JSON.
- Please only return a json object that can be converted into a json directly. DO NOT RETURN ANYTHING OTHER THAN A JSON. it should start with a "{{" and end with a "}}" and there should not be any markers in the response to show that it is a json

Python code:
<code>
{code}
</code>
"""


def get_tool_description_node(state: ToolCollectorState):
    current_code = state['python_code']
    llm_with_struct_output  = llm.with_structured_output(ToolDescriptionList)
    tool_descriptions_list: ToolDescriptionList = llm_with_struct_output.invoke([HumanMessage(content=GET_TOOL_DESCRIPTION_PROMPT.format(code=current_code))])
    return {
        "tool_descriptions_list": tool_descriptions_list
        }

def graph_map_step(state: ToolCollectorState):
    """
    LangGraph node to identify tools in the compiled agent code and generate their implementations.
    It iterates over functions marked with @tool, invokes `tool_infograph` for each,
    and collects the generated code.
    """
    current_code = state['python_code']
    tool_descriptions_list: ToolDescriptionList = state['tool_descriptions_list']    
    tool_names_and_descriptions = "\n".join([f"tool_name: {tool.tool_name}, tool_description: {tool.description}" for tool in tool_descriptions_list.tools])
    message_to_human = tool_names_and_descriptions + "\n\nWould you like to provide us more information if you have a prefered services/sdk to implement the above tools?"

    human_input = interrupt({
        "type": "get_tool_suggestion",
        "question": message_to_human})
    generated_tool_codes = []
    initial_tool_states = []
    for tool in tool_descriptions_list.tools:        
        # Initial state for the tool_infograph
        initial_tool_state = {
            "objective": tool.description, # The description from tool_desc_prompt becomes the objective
            "name": tool.tool_name,
            "user_comment": human_input["context"],
            "input": [], # Inputs/outputs could be further refined or extracted by tool_desc_prompt
            "output": [],
            "name_toolkit": "", # To be determined by sdk_production_node in tool_infograph
            "code": ""          # To be generated by code_production_node in tool_infograph
        }
        initial_tool_states.append(initial_tool_state)
    
    results = tool_infograph.batch(initial_tool_states)
    
    for result in results:
        generated_tool_codes.append(result["code"])

    return {
        "messages": [AIMessage(content="Tool identification and individual code generation complete.")],
        "total_code": generated_tool_codes, # List of code strings for each tool
        "compiled_code": current_code # Pass along the original compiled code
    }

def compile_tool_code_node(state: ToolCollectorState): # Renamed for clarity
    """
    LangGraph node to combine the main agent code with the generated tool function codes.
    Uses an LLM with `tool_compile_prompt` to perform the merge.
    """
    tool_code_list = state['total_code']
    main_agent_code = state['compiled_code'] # Renamed for clarity
    
    if not tool_code_list:
        return {
            "messages": [AIMessage(content="No new tool functions to compile.")],
            "python_code": main_agent_code # Return original if no tools were generated
        }

    full_tool_code = "\n\n".join(tool_code_list) # Join tool codes with newlines
    
    # Use LLM to merge the main agent code with the generated tool definitions
    response = llm.invoke([HumanMessage(content=tool_compile_prompt.format(
        complete_code=main_agent_code,
        functions=full_tool_code
    ))])
    
    # The response from this LLM call is expected to be the final, complete Python code
    return {
        "messages": [AIMessage(content="code generated!")], # Storing the LLM's final code as a message for now
        "python_code": response.content, # Update compiled_code with the final merged code
    }

# --- Tool Compilation Graph Definition (`tool_compile_graph`) ---
# This graph manages the process of finding @tool placeholders in the main generated code,
# generating their implementations, and then merging them back.
tool_compile_workflow = StateGraph(ToolCollectorState)

# Add nodes to the tool compilation graph
tool_compile_workflow.add_node("get_tool_descriptions", get_tool_description_node) # Node to extract tool descriptions
tool_compile_workflow.add_node("graph_map_step", graph_map_step)
tool_compile_workflow.add_node("compile_tools", compile_tool_code_node) # Renamed node

# Define edges for the tool compilation graph
tool_compile_workflow.add_edge(START, "get_tool_descriptions")
tool_compile_workflow.add_edge("get_tool_descriptions", "graph_map_step")
tool_compile_workflow.add_edge("graph_map_step", "compile_tools")
tool_compile_workflow.add_edge("compile_tools", END)

# Compile the tool compilation graph
# This graph doesn't seem to require a checkpointer in this setup,
# but adding one if state needs to be inspected or persisted.
tool_compile_graph = tool_compile_workflow.compile()


