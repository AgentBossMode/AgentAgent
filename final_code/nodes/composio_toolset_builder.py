from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from composio_langchain import ComposioToolSet
from composio.client.collections import AppModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import List, Literal
from pydantic import Field, BaseModel
from langgraph.types import interrupt, Command
from final_code.states.NodesAndEdgesSchemas import JSONSchema

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

TOOL_PROMPT = ChatPromptTemplate.from_template("""
You are responsible for getting the right tools from Composio given description of the requirements by user.
<COMPOSIO_TOOLSET>
{app_list}                                               
</COMPOSIO_TOOLSET>
Follow the following instructions:
1. Respond to the user to understand if they already have any apps/tools that they would like to use for the given requirements. 
2. Check which apps in the COMPOSIO_TOOLSET are best matching the user requirements based on their descriptions.
3. For only the apps retrieved in above step, check the list of actions corresponding to the app using 'get_actions_for_given_app' tool. The tool may return with an empty list, it is fine.
4. check which of the action schema best suits the requirements, donot suggest actions which are remotely connected to the task at hand.
5. Post initial analysis respond with the following: 

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


class ToolBuilderState(MessagesState):
    """
    State for the graph that builds the agent.
    """
    json_schema: JSONSchema = Field(description="The JSON schema that will be used to build the agent. This is the schema that will be used to build the agent.")

class EndOrContinue(BaseModel):
    should_end_conversation : bool = Field(description="true if the AI response does not indicate that it needs any human input.")

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
    prompt= "User will provide the representation of the JSONSchema object, and also provide a list of functions, along with the commposio action that corresponds to the function. You are to generate a JSONSchema object with the updated information. "
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
