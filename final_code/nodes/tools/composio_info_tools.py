from composio import Composio
from composio.client.types import Tool
from composio_langchain import LangchainProvider
import concurrent.futures
composio = Composio(provider=LangchainProvider())


def _get_toolkits_with_tools_batch(apps_batch) -> list[str]:
    """
    Helper function to get toolkit info for a batch of apps.
    Returns a list of formatted strings for apps that have tools.
    """
    slugs = [app.slug for app in apps_batch]
    tools = composio.tools.get_raw_composio_tools(toolkits=slugs, limit=2000)
    # Get unique toolkit slugs from the returned tools
    toolkits_with_tools = set(tool.toolkit.slug for tool in tools)
    results = []
    for app in apps_batch:
        if app.slug in toolkits_with_tools:
            results.append(f"""\
toolkit name: {app.name}
toolkit slug: {app.slug}
Description: {app.meta.description}
\n\n
""")
    return results

def get_all_toolkits():
    """
    Retrieve a list of applications using the Composio Toolset.
    This function fetches applications in batches and checks for tools in parallel.
    Returns:
        str: Returns a well-formatted string of all the apps found that have tools.
    """
    apps = composio.toolkits.get()

    # Batch apps into batches of 10
    batches = [apps[i:i+10] for i in range(0, len(apps), 10)]
    app_list_parts = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(_get_toolkits_with_tools_batch, batches)
        for batch_result in results:
            app_list_parts.extend(batch_result)
    return "\n".join(app_list_parts)

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