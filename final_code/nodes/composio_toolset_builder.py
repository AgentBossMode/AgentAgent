from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from composio_langchain import ComposioToolSet
from composio.client.collections import AppModel
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Literal
from pydantic import Field, BaseModel
from langgraph.types import interrupt, Command
from final_code.states.ToolBuilderState import ToolBuilderState
from final_code.states.NodesAndEdgesSchemas import JSONSchema
from final_code.pydantic_models.EndOrContinue import EndOrContinue

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

TOOL_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant, user will provide you with a list of tool names along with their descriptions
You are also provided with a list of Composio apps along with their descriptions below.
<COMPOSIO_APPS>
{app_list}                                               
</COMPOSIO_APPS>

1. Respond with suggestions of an app for each tool_name provided by the user. Provide the suggestion in a structured manner.
2. User may respond mentioning which app do they require corresponding to a tool description
2. Check which Apps in the COMPOSIO_APPS are best matching the user response.
3. For only the Apps retrieved in above step, check the list of actions corresponding to the app using 'get_actions_for_given_app' tool. The tool may return with an empty list, it is fine.
4. check which of the action schema best suits the requirements, donot suggest actions which are remotely connected to the task at hand.
5. Your job is not to forcefully find a composio app if it is not available, do not impose on user. Understand what they say, if you donot have the app they require in the COMPOSIO_APPS list, just go and provide the final response.
FINAL RESPONSE:                                               
Corresponding to each input tool description:

Example:

Input:
tool_name_1 and tool_description_1
tool_name_2 and tool_description_2
tool_name3 and tool_description_3

Final Output:

*For tool_name_1, the follow composio Action was identified:

**action1 --> justification why you think that this action will satisfy tool_description_1

*For tool_name_2, the follow composio Action was identified::

**action2323 --> justification why you think that this action will satisfy tool_description_2

*For tool_name_3, There is no composio_action found which satisfies the requirements.
""")


composio_toolset = ComposioToolSet()

def get_apps():
    """
    Retrieve a list of applications using the Composio Toolset.
    This function fetches applications with the following options:
    Returns:
        str: Returns a well formatted string of all the apps found. It performs an operation
        using the `composio_toolset.get_apps` method.
    """
    apps: List[AppModel] = composio_toolset.get_apps(no_auth=False, include_local=True)

    app_list : str = ""
    for app in apps:
        app_list += f"""
App name: {app.name}
Description: {app.description}
Categories: {app.categories}
\n\n
"""
    return app_list

def get_actions_for_given_app(app_name: str):
    """
    Retrieves a list of action schemas for a given application.

    Args:
        app_name (str): The name of the application for which action schemas are to be retrieved.

    Returns:
        str: A formatted string containing the names and descriptions of the action schemas 
             associated with the specified application. Each action schema is listed on a 
             new line, separated by a blank line.
    """
    action_schemas_list = "" 
    action_schemas = composio_toolset.get_action_schemas(apps=[app_name], check_connected_accounts=False)
    for action_schema in action_schemas:
        action_schemas_list += f"""{action_schema.name} - {action_schema.description}\n\n"""
    return action_schemas_list

tools =  [get_actions_for_given_app]




def get_human_review(state: ToolBuilderState) -> Command[Literal["composio_tool_fetch", "select_final_tool"]]:
    llm_with_struct = llm.with_structured_output(EndOrContinue)
    should_continue: EndOrContinue = llm_with_struct.invoke([state["messages"][-1]])
    if should_continue.should_end_conversation:
        return Command(goto="select_final_tool")
    value = interrupt(state["messages"][-1].content)
    return Command(goto="composio_tool_fetch", update={"messages": [HumanMessage(content=value)]} ) 


def select_final_tool(state: ToolBuilderState):
    json_schema = state["json_schema"]
    llm_with_struct = llm.with_structured_output(JSONSchema)
    prompt= "User will provide the representation of the JSONSchema object, and also provide a list of functions, along with the commposio action that corresponds to the function, in some cases there might not be any composio_action provided. You are to generate a JSONSchema object with the updated information. "
    updated_json_schema: JSONSchema = llm_with_struct.invoke([SystemMessage(content=prompt)] + [HumanMessage(content=json_schema.model_dump_json())] + [state["messages"][-1]])
    return {"json_schema": updated_json_schema}

composio_tool_fetch_app = create_react_agent(llm, prompt=TOOL_PROMPT.format(app_list=get_apps()), tools=tools, name="composio_tool_fetch")

workflow = StateGraph(ToolBuilderState)
workflow.add_node("human_review", get_human_review)
workflow.add_node("composio_tool_fetch", composio_tool_fetch_app)
workflow.add_node("select_final_tool", select_final_tool)
workflow.add_edge(START, "composio_tool_fetch")
workflow.add_edge("composio_tool_fetch", "human_review")
workflow.add_edge("select_final_tool", END)
composio_tool_builder= workflow.compile()
