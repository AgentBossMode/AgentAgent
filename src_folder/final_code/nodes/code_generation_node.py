from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.llms.model_factory import get_model
# from tests.test_utils.nutrition_llm.json_schema_nutrition import json_schema_nutrition
from copilotkit.langgraph import copilotkit_customize_config
from langchain_core.runnables import RunnableConfig
from final_code.states.NodesAndEdgesSchemas import JSONSchema, get_tools_info,get_nodes_and_edges_info
from final_code.prompt_lib.node_info.graph_state import graph_state
from final_code.prompt_lib.node_info.node_structure import node_structure
from final_code.prompt_lib.node_info.tool_calling import tool_calling
from final_code.prompt_lib.node_info.struct_output import struct_output
from final_code.prompt_lib.node_info.interrupt_info import interrupt_info
from final_code.prompt_lib.node_info.multi_pattern import multi_pattern
from final_code.prompt_lib.edge_info.edge_info import edge_info
from final_code.utils.get_filtered_file import get_filtered_file
from final_code.ast_visitors_lib.validation_script import run_detailed_validation
from final_code.utils.copilotkit_emit_status import append_in_progress_to_list, update_last_status
import traceback
from langchain_core.messages import AIMessage
from langgraph.types import Command
from typing import Literal


llm = get_model()

JSON_WRAPPER = """
<JSON>                                              
{json_schema}
</JSON>

<tools_info>
{tools_info}
</tools_info>

<tools_code>
{tools_code}
</tools_code>
"""
CODE_GEN_PROMPT = PromptTemplate.from_template("""
You are an expert Python programmer specializing in AI agent development via the Langgraph and Langchain SDK. 
Your primary task is to generate compilable, logical, and complete Python code for a LangGraph state graph based on user provided 'JSON' section. You must prioritize LLM-based implementations for relevant tasks.
You will also be provided a 'tools_code.py' which contains all the tools required for the langgraph agent.
You will also be provided with information stating which tool is used by which node in the JSON.
                                               
You must follow the 'INSTRUCTIONS' section carefully to ensure the generated code meets all requirements.

<NODE_IMPLEMENTATION_INSTRUCTIONS>

<COMMONINSTRUCTION>
1. {graph_state}
2. {node_structure}
</COMMONINSTRUCTION>
                                            
## Pattern 1: Tool-calling react agent
1. if the Id of the node is linked to any of the tool in the tools list, you will follow the below format:
2. You will find the tool definition in the 'tools_code.py' provided by user
{tool_calling}
## Pattern 2: LLM with Structured Output
{struct_output}
## Pattern 3: Human-in-the-Loop with Interrupt
{interrupt_info}
## Pattern 4: Multi-Step LLM Processing
{multi_pattern}
                                               
Code produced for each node needs to adhere to the following:
**Important:** Every node's return dictionary **must** include a \"messages\" key, even if it just contains a AiMessage for status.
**State Input Consistency**: Node correctly accesses state properties according to the state schema
**State Output Updates**: Node properly updates state with correct data types and structure
**Type Safety**: All state reads/writes maintain consistent data types
**Required Fields**: Node handles missing or optional state fields appropriately
**Return Format**: Node returns dictionary with proper state updates

</NODE_IMPLEMENTATION_INSTRUCTIONS>

<EDGE_IMPLEMENTATION_INSTRUCTIONS>
{edge_info}
</EDGE_IMPLEMENTATION_INSTRUCTIONS>

<CODE_GENERATION_INSTRUCTIONS>
Generate a compilable Python script following this structure:

### 1. Imports and Setup
Below are some common langggraph imports, you need note necessarily add them, refer to them when writing the code.
```python
from typing import Dict, Any, List, Optional, Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver, InMemorySaver
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
import re
import json
```
                                               
(IMPORTANT) Importing different tool definitions in 'tools_code.py' file.
EXAMPLE:
IF THE tools_code.py looks like:
```python
from composio import Composio
from composio_langchain import LangchainProvider
composio = Composio(provider=LangchainProvider())

tool_name_1 = composio.tools.get(user_id=os.environ(\"USER_ID\"), tools=[\"RANDOM\"]) 
tool_name_2 = composio.tools.get(user_id=os.environ(\"USER_ID\"), tools=[\"composio_tool_name_xyz\"])
@tool
def tool_name_3():
    # method definition

@tool
def tool_name_4():
    # method definition
```
                                               
The import statement you will add to final output is (FOLLOW THIS STRICTLY)
```python
from tools_code import tool_name_1, tool_name_2, tool_name_3, tool_name_4
```                                          
### 2. Create llm definition
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

### 3. If there is mention of ReactAgentState in create_react_agent method calls, add this after the imports section:

class ReactAgentState(MessagesState):
    remaining_steps: int
    structured_response: any
                                                                                              
### 4. Final Graph Compilation
```python
checkpointer = InMemorySaver()
app = workflow.compile(
    checkpointer=checkpointer
)
```                        

DONOT ADD '__main__' block or any other boilerplate code, the code should be self-contained and compilable.

<QUALITY_CHECKLIST>
Before finalizing your code, verify:
- [ ] All imports are included and correct, no duplicate imports.
- [ ] GraphState properly extends MessagesState 
- [ ] LLM calls include proper error handling
- [ ] Structured output uses proper Pydantic models
- [ ] Inside any Pydantic model, do not use 'dict' or 'Dict' as a type hint for any field. you can instead use str and then store a serialized JSON
- [ ] Conditional edges handle all possible routing outcomes
- [ ] Code is compilable and logically consistent
- [ ] No unterminated string literals or syntax errors
- [ ] The code needs to be production ready, which means there is no place for any placeholder code, no assumptions, and no incomplete sections.
- [ ] Ensure that the code does not access graphstate like an object attribute, it needs be accessed like a dict
- [ ] Donot use os.environs for anything, you can instead put it as attribute in GraphState and access it from there, assume the user will pass it when the workflow starts.
- [ ] **Every node's return dictionary includes a "messages" key.**
- [ ] **The first LLM call/node appropriately utilizes `state["messages"]` as part of its input.**
- [ ] The user provided code should not be copied to final_output, only the imports as mentioned in the CODE_GENERATION_INSTRUCTIONS section.
- [ ] **Exit Conditions**: Clear termination conditions and END nodes
                                               
</QUALITY_CHECKLIST>

</CODE_GENERATION_INSTRUCTIONS>

                                                                                       
<INSTRUCTIONS>
1. start analyzing the nodes, refer to <NODE_IMPLEMENTATION_INSTRUCTIONS>
   - For the first LLM call or first node, ensure the LLM's input collects information contains `state["messages"]` to incorporate the conversation history.
3. Now create the edges, refer to the <EDGE_IMPLEMENTATION_INSTRUCTIONS> section.
4. Now to piece it all together follow <CODE_GENERATION_INSTRUCTIONS>
</INSTRUCTIONS>



Please return only complete and compilable langgraph python code
""")

def generate_code_gen_prompt():
    return CODE_GEN_PROMPT.format(
            graph_state=graph_state,
            node_structure=node_structure,
            tool_calling=tool_calling,
            struct_output=struct_output,
            interrupt_info=interrupt_info,
            multi_pattern=multi_pattern,
            edge_info=edge_info)

def get_schema_info(json_schema: JSONSchema, tools_code: str):
    return JSON_WRAPPER.format(
            json_schema=get_nodes_and_edges_info(json_schema),
            tools_info=get_tools_info(json_schema.tools),
            tools_code=tools_code)

async def generate_python_code(modifiedConfig: RunnableConfig, json_schema: JSONSchema, tools_code, answers: dict) -> str:
    try:
        llm_model = get_model()
        messages_list = [ SystemMessage(content=generate_code_gen_prompt()),
            HumanMessage(content=get_schema_info(json_schema, tools_code))]
        if answers is not None:
            #[{'question': 'Could you please provide the Notion database ID(s) where the food intake and exercise activity logs should be stored and retrieved from? This is needed to configure the User Data Storage tools properly for logging and querying data.', 'answer': '[Marked as irrelevant]', 'isIrrelevant': True}]
            try:
                relevant_answers = [answer for answer in answers if not answer.get("isIrrelevant")]
                if relevant_answers:
                    messages_list.append(HumanMessage(content=f"<additional_information>{relevant_answers}</additional_information>"))
            except Exception as e:
                pass
            
        response = await llm_model.ainvoke(messages_list, config=modifiedConfig)
        py_code = get_filtered_file(response.content)
        return py_code
    except Exception as e:
        raise e

async def code_node(state: AgentBuilderState, config: RunnableConfig) -> Command[Literal["code_analyzer_node", "__end__"]]:
    """
    LangGraph node to generate the final Python code for the agent.
    It uses the gathered agent_instructions and the CODE_GEN_PROMPT.
    """
    try:
        modifiedConfig = copilotkit_customize_config(
            config,
            emit_messages=False
        )
        #json_schema_final = json_schema_nutrition
        state["current_tab"] = "code"
        await append_in_progress_to_list(modifiedConfig, state, "Generating Python code...")
        response  = await generate_python_code(modifiedConfig, state["json_schema"], state["tools_code"], state["answers"])
        await update_last_status(modifiedConfig, state, "Python code generated successfully", True)
        # Return the generated Python code and an AI message
        return Command(
            goto="code_analyzer_node",
            update={
                "current_tab": "code",
                "python_code": response,
                "agent_status_list": state["agent_status_list"],
            }
        )
    except Exception as e:
        return Command(
            goto="__end__",
            update={
                "exception_caught": f"{e}\n{traceback.format_exc()}",
                "messages": [AIMessage(content="An error occurred during generating Python code. Please try again.")]
            }
        ) 

async def code_analyzer_node(state: AgentBuilderState, config: RunnableConfig) -> Command[Literal["mock_tools_writer", "__end__"]]:
    try:
        FIX_PROMPT = """
You are a langgraph expert, user will provide you with a python code and a list of errors/warning with fixes. Your job is to make the fixes.
"""
        PYTHON_PROMPT = """
<python_code>
{python_code}
</python_code>
"""
        ERROR_REPORT = """
<error_report>
{fixes}
</error_report>
"""
        modifiedConfig = copilotkit_customize_config(
            config,
            emit_messages=False
        )
        python_file = get_filtered_file(state["python_code"])
        error_report = run_detailed_validation(python_file)
        if len(error_report["errors"])>0:
            await append_in_progress_to_list(modifiedConfig, state, "Analyzing code for correctness...")
            llm_model = get_model()
            response = await llm_model.ainvoke(input=[SystemMessage(content=FIX_PROMPT),
                                                 HumanMessage(content=PYTHON_PROMPT.format(python_code=python_file)),
                                                 HumanMessage(content=ERROR_REPORT.format(fixes=str(error_report["errors"])))], config=modifiedConfig)
            await update_last_status(modifiedConfig, state, "Code analysis complete", True)
            return Command(
                goto="mock_tools_writer",
                update={"python_code": response.content, "agent_status_list": state["agent_status_list"]}
            )
        return Command(
            goto="mock_tools_writer",
            update={"python_code": state["python_code"]}
        )
    except Exception as e:
        return Command(
            goto="__end__",
            update={
                "exception_caught": f"{e}\n{traceback.format_exc()}",
                "messages": [AIMessage(content="An error occurred during analyzing code. Please try again.")]
            }
        )
