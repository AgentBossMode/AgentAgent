from composio import Composio
from composio.client.types import Tool
from composio_langchain import LangchainProvider
composio = Composio(provider=LangchainProvider())



def get_all_toolkits():
    """
    Retrieve a list of applications using the Composio Toolset.
    This function fetches applications with the following options:
    Returns:
        str: Returns a well formatted string of all the apps found. It performs an operation
        using the `composio_toolset.get_apps` method.
    """
    apps = composio.toolkits.get()

    app_list : str = ""
    for app in apps:
        app_list += f"""
toolkit name: {app.name}
toolkit slug: {app.slug}
Description: {app.meta.description}
\n\n
"""
    return app_list

def get_all_raw_tool_schemas_for_a_toolkit(toolkit_name: str):
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
    tool_list: list[Tool] = composio.tools.get_raw_composio_tools(toolkits=[toolkit_name])
    for tool in tool_list:
        action_schemas_list += f"""
TOOL SLUG: {tool.slug}
TOOL NAME: {tool.name}
TOOL DESCRIPTION: {tool.description}

"""
    return action_schemas_list

def get_raw_tool_schema(tool_name: str) -> Tool:
    """
        Retrieves the schema for a given action name from Composio.

        Args:
            action_name (str): The name of the action to retrieve the schema for.

        Returns:
            Tool: The schema for the given action, as a Tool object.
        """
    return composio.tools.get_raw_composio_tools(tools=[tool_name])[0]