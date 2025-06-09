from composio_openai import ComposioToolSet, Action, App

# Initialize ToolSet (assuming API key is in env)
toolset = ComposioToolSet()

# Fetch only the tool for starring a GitHub repo
github_star_tool = toolset.get_tools(
    actions=[Action.GITHUB_STAR_A_REPOSITORY_FOR_THE_AUTHENTICATED_USER]
)

print(github_star_tool)
# Output will contain the schema for the specified action.

# Fetch default tools for the connected GitHub app
github_tools = toolset.get_tools(apps=[App.GITHUB])  # only actions marked as important are picked up 

print(f"Fetched {len(github_tools)} tools for GitHub.")
# Output contains schemas for 'important' GitHub tools.


# FIND A TOOL BY SEMANTIC SEARCH

## DOES NOT WORK!!!!!!!!!
# Describe the task
# query = "create a new page in notion"

# # Find relevant action ENUMS (Python-specific helper)
# relevant_actions = toolset.find_actions_by_use_case(
#     use_case=query,

#     # advanced=True # Use for complex queries needing multiple tools
# )

# print(f"Found relevant actions")

# # Fetch the actual tool schemas for the found actions
# if relevant_actions:
#     notion_tools = toolset.get_tools(actions=relevant_actions)
#     print(f"Fetched {len(notion_tools)} tool(s) for the use case.")
# else:
#     print("No relevant actions found for the use case.")



from composio import ComposioToolSet, Action, App # Use base ComposioToolSet for schema inspection

# Initialize base ToolSet
base_toolset = ComposioToolSet()

# Get the raw schema for a specific Google Calendar action
# Bypass the check for an active Google Calendar connection
calendar_schemas = base_toolset.get_action_schemas(
    actions=[Action.GOOGLECALENDAR_LIST_CALENDARS],
    check_connected_accounts=False
)

if calendar_schemas:
    import json
    print("Raw Schema for GOOGLECALENDAR_LIST_CALENDARS:")
    # calendar_schemas is a list, access the first element
    print(json.dumps(calendar_schemas[0].model_dump(), indent=2))
else:
    print("Schema not found.")

# You can also fetch schemas by app or tags similarly
# github_schemas = base_toolset.get_action_schemas(
#    apps=[App.GITHUB], check_connected_accounts=False
# )
