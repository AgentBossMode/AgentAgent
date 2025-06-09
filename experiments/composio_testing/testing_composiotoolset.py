from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from composio_langchain import ComposioToolSet, Action, App
from composio.client.collections import AppModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from typing import List

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tool_prompt = """
You are responsible for getting the right tool given a tool description from Composio

Follow the following instructions:
1. Get the list of apps present in Composio using 'get_app' tool.
2. Check which apps are best matching the user requirements based on their description.
3. For only the apps retrieved in above step, check the list of actions corresponding to the app using 'get_actions_for_given_app' tool. The tool may return with an empty list, it is fine.
4. check which of the action schema best suits the requirement, donot suggest actions which are remotely connected to the task at hand.
                                               
Respond with the following: 

App (there can be multiple entries) - Corresponding to each app, if found provide the exact ActionModel name (donot make human readable), if no actions were found it is okay/

"""




composio_toolset = ComposioToolSet()

def simplify_list_schemas(schema: dict) -> dict:
    """Removes recipient_email and attachment params from the schema."""
    params = schema.get("parameters", {}).get("properties", {})
    params.pop("category", None)
    params.pop("enabled_only", None)
    # We could also modify descriptions here, e.g.:
    # schema["description"] = "Sends an email using Gmail (recipient managed separately)."
    return schema


tools = composio_toolset.get_tools(actions=[Action.COMPOSIO_LIST_APPS],
                                   processors={
                                       "schema": {Action.COMPOSIO_LIST_APPS: simplify_list_schemas}
                                   })

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
    Retrieve action schemas for a specified app_name.

    This function uses the `composio_toolset` to fetch action schemas
    associated with the given app_name.

    Args:
        app_name (str): The name of the application for which to retrieve action schemas. DONOT call with empty parameters.

    Returns:
        list: A list of action schemas for the specified application.
    """
    return composio_toolset.get_action_schemas(apps=[app_name], check_connected_accounts=False)

tools =  [get_apps, get_actions_for_given_app]

def agent_node(state: MessagesState):
    """
    Process messages through the LLM and return the response
    """
    llm_tools = llm.bind_tools(tools)
    response = llm_tools.invoke([SystemMessage(content=tool_prompt)] + state["messages"])
    return {"messages": [response]}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

app = workflow.compile()
#task = "find a tool that sends emails"
#task = "track my workout"
#task = "raise a pull request"
#task = "create ad"
task = "create notion page"
for output in app.stream({"messages": [HumanMessage(content=task)]}, stream_mode="updates"):
    try:
        print(output["agent"]["messages"][-1].pretty_print())
    except:
        print(output["tools"]["messages"][-1].pretty_print())
