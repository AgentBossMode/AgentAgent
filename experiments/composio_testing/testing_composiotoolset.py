from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from composio_langchain import ComposioToolSet, Action
from composio.client.collections import AppModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import List, Literal
from pydantic import Field, BaseModel
from langgraph.types import interrupt, Command

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tool_prompt = """
You are responsible for getting the right tools from Composio given description of the requirements by user.

Follow the following instructions:
1. Respond to the user to understand if they already have any apps/tools that they would like to use for the given requirements. 
2. Get the list of apps present in Composio using 'get_app' tool.
3. Check which apps are best matching the user requirements based on their descriptions.
4. For only the apps retrieved in above step, check the list of actions corresponding to the app using 'get_actions_for_given_app' tool. The tool may return with an empty list, it is fine.
5. check which of the action schema best suits the requirements, donot suggest actions which are remotely connected to the task at hand.
6. Post initial analysis respond with the following: 

Corresponding to each input tool description:
App (there can be multiple entries) - Corresponding to each app, if found provide the exact ActionModel name (donot make human readable), if no actions were found it is okay/

Example:

Input:
tool_name_1 and tool_description 1
tool_name_2 and tool_description 2

Output:

*For tool_name_1, here are the following proposed tools:

**App1:
***action1 --> justification why you think that this action will satisfy tool_description1

**App2:
***action3 --> justification why you think that this action will satisfy tool_description1

*For tool_name_2, here are the following proposed tools:

**App5:
***action12 --> justification why you think that this action will satisfy tool_description2

**App6:
***action2323 --> justification why you think that this action will satisfy tool_description2


For both tool_name_1 and tool_name_2, now generate a bunch of questions which would better help understand which tool might help select between the bunch of actions. Most likely the user already knows what tool he needs to integrate.


7. If you get human input, responding to the above, now finally provide the final tools to be used corresponding to each user requirement.
"""

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

tools =  [get_apps, get_actions_for_given_app]

class ToolBuilderState(MessagesState):
    """
    State for the graph that builds the agent.
    """
    is_tool_fetched_via_composio: bool = Field(description="is tool fetched by composio")
    selected_tool_via_composio: str = Field(description="the tool selected via composio")


def agent_node(state: MessagesState):
    """
    Process messages through the LLM and return the response
    """
    llm_tools = llm.bind_tools(tools)
    response = llm_tools.invoke([SystemMessage(content=tool_prompt)] + state["messages"])
    return {"messages": [response]}

class EndOrContinue(BaseModel):
    should_end_conversation : bool = Field(description="true if the AI response does not indicate that it needs any human input.")

def get_human_review(state: MessagesState) -> Command[Literal["composio_tool_fetch", "select_final_tool"]]:
    llm_with_struct = llm.with_structured_output(EndOrContinue)
    should_continue: EndOrContinue = llm_with_struct.invoke([state["messages"][-1]])
    if should_continue.should_end_conversation:
        return Command(goto="select_final_tool")
    value = interrupt(state["messages"][-1].content)
    return Command(goto="composio_tool_fetch", update={"messages": [HumanMessage(content=value)]} ) 

def select_final_tool(state: ToolBuilderState):
    prompt = "You have a list of Apps-Actions and the user has provided which tool to use. Your job is to just respond the final action."
    response = llm.invoke([SystemMessage(content=prompt)] + state["messages"])
    return {"messages": [response], "selected_tool_via_composio": response.content, "is_tool_fetched_via_composio": True}

def generate_tool_node(state: ToolBuilderState):
    prompt = ChatPromptTemplate.from_template("""
from composio_langchain import ComposioToolSet, Action
composio_toolset = ComposioToolSet()

tools = composio_toolset.get_tools(actions=[Action.{tool_name}])
tool_node = ToolNode(tools)

def call_model(state: MessagesState):
    \"\"\"
    Process messages through the LLM and return the response
    \"\"\"
    messages = state["messages"]
    model_with_tools = model.bind_tools(tools)
    response = model_with_tools.invoke(messages)
    return {{"messages": [response]}}

workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge("__start__", "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
)
workflow.add_edge("tools", "agent")
""")
    return {"messages": [AIMessage(content=prompt.format(tool_name=state["selected_tool_via_composio"]))]} 

composio_tool_fetch_graph = StateGraph(MessagesState)
composio_tool_fetch_graph.add_node("agent", agent_node)
composio_tool_fetch_graph.add_node("tools", ToolNode(tools))
composio_tool_fetch_graph.add_edge(START, "agent")
composio_tool_fetch_graph.add_conditional_edges("agent", tools_condition)
composio_tool_fetch_graph.add_edge("tools", "agent")
composio_tool_fetch_app = composio_tool_fetch_graph.compile()

workflow = StateGraph(MessagesState)
workflow.add_node("human_review", get_human_review)
workflow.add_node("composio_tool_fetch", composio_tool_fetch_app)
workflow.add_node("select_final_tool", select_final_tool)
workflow.add_node("generate_tool_node", generate_tool_node)
workflow.add_edge(START, "composio_tool_fetch")
workflow.add_edge("composio_tool_fetch", "human_review")
workflow.add_edge("select_final_tool", "generate_tool_node")
workflow.add_edge("generate_tool_node", END)
app = workflow.compile()


if __name__ == "__main__":
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
