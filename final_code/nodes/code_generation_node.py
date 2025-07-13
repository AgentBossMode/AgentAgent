from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.llms.model_factory import get_model
from final_code.utils.MockJsonSchema import json_schema_str
from copilotkit.langgraph import copilotkit_customize_config
from langchain_core.runnables import RunnableConfig
import uuid
from pydantic import BaseModel, Field

from final_code.prompt_lib.node_info.graph_state import graph_state
from final_code.prompt_lib.node_info.node_structure import node_structure
from final_code.prompt_lib.node_info.tool_calling import tool_calling
from final_code.prompt_lib.node_info.struct_output import struct_output
from final_code.prompt_lib.node_info.interrupt_info import interrupt_info
from final_code.prompt_lib.node_info.multi_pattern import multi_pattern
from final_code.prompt_lib.edge_info.edge_info import edge_info
llm = get_model()

CODE_GEN_PROMPT = PromptTemplate.from_template("""
You are an expert Python programmer specializing in AI agent development via the Langgraph and Langchain SDK. Your primary task is to generate compilable, logical, and complete Python code for a LangGraph state graph based on user 'JSON' section below. You must prioritize LLM-based implementations for relevant tasks.

<JSON>                                              
{json_schema}
</JSON>
<TOOL_BINDING_INSTRUCTIONS>
1. From the toolset field in the json schema identify the tool schema which might look like the following:
Example 1 (composio tool)
{{
          "name": "CRM_Tool",
          "description": "Tool to interact with CRM systems to update lead information, log activities, and retrieve lead statuses.",
          "is_composio_tool": true,
          "composio_toolkit_name": "HubSpot",
          "composio_tool_name": "DO_ABC_ACTIVITY",
          "py_code": null,
          "node_ids": [
            "node_a",
            "node_c"
          ]
}}

Example 2 (non-comosio tool)
{{
          "name": "search_customer_database",
          "description": "Tool to search customer database",
          "is_composio_tool": false,
          "composio_toolkit_name": "None",
          "composio_tool_name": "None",
          "py_code": "The python code to implement this tool ....",
          "node_ids": [
            "node_a",
            "node_b"
          ]
}}
                                               
2. IF is_composio_tool is true (Example 1), THEN: 
```python
from composio import Composio
from composio_langchain import LangchainProvider
composio = Composio(provider=LangchainProvider())
tools = composio.tools.get(user_id=os.environ(\"USER_ID\"), tools=[\"composio_tool_name\"])  # Replace with actual tool name (in this example DO_ABC_ACTIVITY)

```                                               
3. ELSE IF the tools corresponding to a node are non-composio tools  (Example 2) use the py_code field in the json schema:
``` python
                                               
from langchain_core.tools import tool

#py_code goes here. For example:
@tool
def search_customer_database(customer_id: str) -> str:
    '''Search for customer information by ID.'''
    # Direct implementation - no nested LLM calls
    return f"Customer {{customer_id}} data retrieved"
```
4. Follow the above for all the different tools in the Tool List.
</TOOL_BINDING_INSTRUCTIONS>

<NODE_IMPLEMENTATION_INSTRUCTIONS>

<COMMONINSTRUCTION>
1. {graph_state}
2. {node_structure}
</COMMONINSTRUCTION>
                                            
## Pattern 1: Tool-calling react agent
1. if the Id of the node is linked to any of the tool in the tools list, you will follow the below format:
{tool_calling}
## Pattern 2: LLM with Structured Output
{struct_output}
## Pattern 3: Human-in-the-Loop with Interrupt
{interrupt_info}
## Pattern 4: Multi-Step LLM Processing
{multi_pattern}
</NODE_IMPLEMENTATION_INSTRUCTIONS>

<EDGE_IMPLEMENTATION_INSTRUCTIONS>
{edge_info}
</EDGE_IMPLEMENTATION_INSTRUCTIONS>

                                                                                       
<INSTRUCTIONS>
1. First create the tools, refer to <TOOLBINDINGINSTRUCTIONS> section.
2. Now start analyzing the nodes, refer to <NODE_IMPLEMENTATION_INSTRUCTIONS>
3. Now create the edges, refer to the <EDGE_IMPLEMENTATION_INSTRUCTIONS> section.
4. Now to piece it all together follow <CODE_GENERATION_INSTRUCTIONS>
</INSTRUCTIONS>


<CODE_GENERATION_INSTRUCTIONS>

Generate a single, self-contained, and compilable Python script following this structure:

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



###2: Final Graph Compilation
```python
checkpointer = InMemorySaver()
app = workflow.compile(
    checkpointer=checkpointer
)
                        

DONOT ADD '__main__' block or any other boilerplate code, the code should be self-contained and compilable.
```
<QUALITY_CHECKLIST>
Before finalizing your code, verify:
- [ ] All imports are included and correct, no duplicate imports.
- [ ] GraphState properly extends MessagesState  
- [ ] LLM calls include proper error handling
- [ ] Tools are self-contained (no nested LLM calls)
- [ ] Structured output uses proper Pydantic models
- [ ] Conditional edges handle all possible routing outcomes
- [ ] Code is compilable and logically consistent
- [ ] The code needs to be production ready, which means there is no place for any placeholder code, no assumptions, and no incomplete sections.
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



Please return only complete and compilable langgraph python code
""")

class PythonCode(BaseModel):
    code: str = Field(description="complete and compilable langgraph python code")

def code_node(state: AgentBuilderState, config: RunnableConfig):
    """
    LangGraph node to generate the final Python code for the agent.
    It uses the gathered agent_instructions and the CODE_GEN_PROMPT.
    """

    modifiedConfig = copilotkit_customize_config(
        config,
        emit_intermediate_state=[{
            "state_key": "python_code",
            "tool": "PythonCode",
            "tool_argument": "code",
        }],
    )

    json_schema_final = state["json_schema"].model_dump_json(indent=2)
    #json_schema_final = json_schema_str
    response: PythonCode  = generate_python_code(modifiedConfig, json_schema_final)
    # Return the generated Python code and an AI message
    return {
        "python_code": response.code,
    }

def generate_python_code(modifiedConfig, json_schema_final) -> PythonCode:
    code_llm_writer = llm.with_structured_output(PythonCode)
    response: PythonCode = code_llm_writer.invoke([HumanMessage(content=
                                                                CODE_GEN_PROMPT.format(
                                                                    json_schema=json_schema_final,
                                                                    # snippets
                                                                    graph_state=graph_state,
                                                                    node_structure=node_structure,
                                                                    tool_calling=tool_calling,
                                                                    struct_output=struct_output,
                                                                    interrupt_info=interrupt_info,
                                                                    multi_pattern=multi_pattern,
                                                                    edge_info=edge_info))],
                                                                      config=modifiedConfig)
                                                                      
    return response
