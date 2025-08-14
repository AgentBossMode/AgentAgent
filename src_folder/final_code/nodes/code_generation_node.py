from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.llms.model_factory import get_model
# from tests.test_utils.nutrition_llm.json_schema_nutrition import json_schema_nutrition
from copilotkit.langgraph import copilotkit_customize_config
from langchain_core.runnables import RunnableConfig
import uuid
from pydantic import BaseModel, Field
from final_code.states.NodesAndEdgesSchemas import JSONSchema, get_tools_info,get_nodes_and_edges_info
from final_code.prompt_lib.node_info.graph_state import graph_state
from final_code.prompt_lib.node_info.node_structure import node_structure
from final_code.prompt_lib.node_info.tool_calling import tool_calling
from final_code.prompt_lib.node_info.struct_output import struct_output
from final_code.prompt_lib.node_info.interrupt_info import interrupt_info
from final_code.prompt_lib.node_info.multi_pattern import multi_pattern
from final_code.prompt_lib.edge_info.edge_info import edge_info
from copilotkit.langgraph import copilotkit_emit_state 
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
def tool_name_3():
    # method definition
```
                                               
The import statement you will add to final output is (FOLLOW THIS STRICTLY)
```python
from tools_code import tool_name_1, tool_name_2, tool_name_3, tool_name_3 
```                                          
### 2. Create llm definition
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

###3: Final Graph Compilation
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
- [ ] Conditional edges handle all possible routing outcomes
- [ ] Code is compilable and logically consistent
- [ ] No unterminated string literals or syntax errors
- [ ] The code needs to be production ready, which means there is no place for any placeholder code, no assumptions, and no incomplete sections.
- [ ] Ensure that the code does not access graphstate like an object attribute, it needs be accessed like a dict
- [ ] Assume any API keys(e.g., OPENAI_API_KEY, GOOGLE_API_KEY) are part of the environment variables and all environment variables are to be defined using the os.environs notation
- [ ] **Every node's return dictionary includes a "messages" key.**
- [ ] **The first LLM call/node appropriately utilizes `state["messages"]` as part of its input.**
- [ ] The user provided code should not be copied to final_output, only the imports as mentioned in the CODE_GENERATION_INSTRUCTIONS section.
- [ ] **Exit Conditions**: Clear termination conditions and END nodes
                                               
</QUALITY_CHECKLIST>

<KEY_EXTRACTION_INSTRUCTIONS>
After generating the complete Python script, add a section titled:

## Required Keys and Credentials

List all environment variables, API keys, and external dependencies needed as comment :
- Environment variables (e.g., OPENAI_API_KEY)
- Tool-specific credentials 
- External service configurations
- Database connection strings (if applicable)

If no external keys are needed, state: "No external API keys required for this implementation."
</KEY_EXTRACTION_INSTRUCTIONS>
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

class PythonCode(BaseModel):
    code: str = Field(description="complete and compilable langgraph python code")

async def code_node(state: AgentBuilderState, config: RunnableConfig):
    """
    LangGraph node to generate the final Python code for the agent.
    It uses the gathered agent_instructions and the CODE_GEN_PROMPT.
    """

    modifiedConfig = copilotkit_customize_config(
        config,
        emit_messages=False
    )
    #json_schema_final = json_schema_nutrition
    state["current_tab"] = "code"
    state["current_status"] = {"inProcess":True ,"status": "Generating Python code.."} 
    await copilotkit_emit_state(config=modifiedConfig, state=state)
    response  = await generate_python_code(modifiedConfig, state["json_schema"], state["tools_code"])
    state["current_status"] = {"inProcess":False ,"status": "Python code generated successfully."} 
    await copilotkit_emit_state(config=modifiedConfig, state=state)
    # Return the generated Python code and an AI message
    return {
        "python_code": response,
        "messages":[AIMessage(content="Generated the agent code, check main.py in the code editor.")]
    } 

def get_schema_info(json_schema: JSONSchema, tools_code: str):
    return JSON_WRAPPER.format(
            json_schema=get_nodes_and_edges_info(json_schema),
            tools_info=get_tools_info(json_schema.tools),
            tools_code=tools_code)

async def generate_python_code(modifiedConfig: RunnableConfig, json_schema: JSONSchema, tools_code) -> str:
    llm = get_model()
    response = await llm.ainvoke([
        SystemMessage(content=generate_code_gen_prompt()),
        HumanMessage(content=get_schema_info(json_schema, tools_code))],
        config=modifiedConfig)
                                                                      
    return response.content
