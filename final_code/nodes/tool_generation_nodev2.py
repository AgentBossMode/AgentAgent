from typing import List
from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.states.NodesAndEdgesSchemas import JSONSchema, Tool
from final_code.nodes.native_tool_builder import native_tool_builder
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt
from final_code.llms.model_factory import get_model
workflow = StateGraph(AgentBuilderState)
def get_composio_tools_node(state: AgentBuilderState):
    json_schema: JSONSchema = state["json_schema"]
    tool_list = []

    for tool in json_schema.tools:
        tool_list.append({
            "tool_name": tool.name,
            "tool_description": tool.description,
        }) 
    tools_selection_data = interrupt({
        "type": "select_composio_tools",
        "payload": tool_list
    }) 
    tool_set = set()
    for tool_selected in tools_selection_data["completed"]:
        tool_set.add(tool_selected["toolkit_slug"])

    tools_to_update: List[Tool] = json_schema.tools 

    for tool_selected in tools_selection_data["completed"]:
        for tool_to_update in tools_to_update:
            if tool_selected["tool_name"] == tool_to_update.name:
                tool_to_update.is_composio_tool = True
                tool_to_update.composio_toolkit_slug = tool_selected["toolkit_slug"]
                tool_to_update.composio_tool_slug = tool_selected["tool_slug"]
                tool_to_update.py_code = None # Composio tools don't have py_code directly
                break


    json_schema.tools = tools_to_update
    print(json_schema.tools)
    return {"tool_set": tool_set, "json_schema": json_schema, "messages": "Composio tools selected successfully"}


async def process_non_composio_tools(state: AgentBuilderState):
    json_schema: JSONSchema = state["json_schema"]
    tool_list = ""

    for tool in json_schema.tools:
        if not tool.is_composio_tool:
            tool_list += f"tool_name: {tool.name}, tool_description: {tool.description}\n" 
    if tool_list == "":
        return 
    updated_json_schema = await native_tool_builder.ainvoke({"messages": [HumanMessage(content=tool_list)], "json_schema": json_schema})
    return {"json_schema": updated_json_schema["json_schema"], "messages": updated_json_schema["messages"]}

async def generate_tools_code(state: AgentBuilderState):
    TOOL_FILE_GENERATION_PROMPT = """
Follow the tool_binding instructions, for the given json provided by the user.
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
CRM_Tool = composio.tools.get(user_id=os.environ(\"USER_ID\"), tools=[\"DO_ABC_ACTIVITY\"]) 
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
5. Generate a final file without markdowns.
6. Make sure that if you are generating in methods, then make it with proper docstrings, which means detailed description, the inputs, and what it returns.
7. Add imports properly at the top.
</TOOL_BINDING_INSTRUCTIONS>
"""
    json_schema: JSONSchema = state["json_schema"]
    tools: List[Tool] = json_schema.tools
    tools_info_list = "\n".join(tool.model_dump_json() for tool in tools)
    llm = get_model()
    tools_code = llm.invoke([SystemMessage(TOOL_FILE_GENERATION_PROMPT), HumanMessage(content=tools_info_list)])
    return {"tools_code": tools_code.content}

workflow.add_node("get_composio_tools", get_composio_tools_node)
workflow.add_node("process_non_composio_tools", process_non_composio_tools)
workflow.add_node("generate_tools_code", generate_tools_code)

workflow.add_edge(START, "get_composio_tools")
workflow.add_edge("get_composio_tools", "process_non_composio_tools")
workflow.add_edge("process_non_composio_tools", "generate_tools_code")
workflow.add_edge("generate_tools_code", END)

tool_graph = workflow.compile()