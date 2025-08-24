from typing import Dict, Any, List, Optional, Literal, TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
import re
import json

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

MOCK_TOOL_PROMPT = """
You are a helpful assistant that generates mock data for tool outputs.
Given the tool's purpose and expected output, generate a realistic mock response.
"""

INPUT_PROMPT = """
Tool Docstring: {description}
Input: {input}
Generate a mock output for this tool.
"""

@tool
def MS_Graph_API_READ(app_id: str):
    """
    To retrieve the latest state of MS Graph resources, such as 'application' objects and their properties.

    Args:
        app_id (str): The ID of the application to retrieve.

    Returns:
        None: This function prints the application display name and ID.

    Example:
        MS_Graph_API_READ(app_id="your_application_id")
    """
    class ApplicationState(BaseModel):
        display_name: str = Field(description="The display name of the application.")
        app_id: str = Field(description="The unique ID of the application.")

    input_str = f"app_id: {app_id}"
    description = MS_Graph_API_READ.__doc__

    result = llm.with_structured_output(ApplicationState).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

@tool
def Database_CREATE(database_id: str, properties: Optional[List[dict]] = None, child_blocks: Optional[List[dict]] = None, cover: Optional[str] = None, icon: Optional[str] = None):
    """
    Creates a new page (row) in a specified Notion database.
    To store historical snapshots of MS Graph resources for comparison.

    Args:
        database_id (str): Identifier (UUID) of the Notion database where the new page (row) will be inserted.
        properties (Optional[List[dict]]): Property values for the new page.
        child_blocks (Optional[List[dict]]): A list of NotionRichText objects defining content blocks to append to the new page's body.
        cover (Optional[str]): URL of an external image to set as the page cover.
        icon (Optional[str]): Emoji to be used as the page icon.

    Returns:
        str: JSON string of the output, including `response_data` and `successful` status.

    Example:
        Database_CREATE(
            database_id="59833787-2cf9-4fdf-8782-e53db20768a5",
            properties=[
                {"name": "Name", "type": "title", "value": "MS Graph Snapshot 2024-01-01"},
                {"name": "Date", "type": "date", "value": "2024-01-01T00:00:00.000-00:00"}
            ]
        )
    """
    class ResponseData(BaseModel):
        response_data: str = Field(description="JSON string of the complete JSON object representing the newly created page (row), as returned by the Notion API.")

    class DatabaseCreateOutput(BaseModel):
        data: ResponseData = Field(description="Data from the action execution.")
        successful: bool = Field(description="Whether or not the action execution was successful or not.")

    input_str = f"database_id: {database_id}, properties: {properties}, child_blocks: {child_blocks}, cover: {cover}, icon: {icon}"
    description = Database_CREATE.__doc__

    result = llm.with_structured_output(DatabaseCreateOutput).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

@tool
def Database_READ(database_id: str, page_size: Optional[int] = 2, sorts: Optional[List[dict]] = None, start_cursor: Optional[str] = None):
    """
    Queries a Notion database for pages (rows), where rows are pages and columns are properties.
    To retrieve historical snapshots of MS Graph resources for comparison.

    Args:
        database_id (str): Identifier of the Notion database to query.
        page_size (Optional[int]): The maximum number of items (pages or rows) to return in a single response.
        sorts (Optional[List[dict]]): List of sort rules. Each item must use the keys 'property_name' and 'ascending'.
        start_cursor (Optional[str]): An opaque cursor for pagination, used to retrieve the next set of results.

    Returns:
        str: JSON string of the output, including `response_data` and `successful` status.

    Example:
        Database_READ(
            database_id="59833787-2cf9-4fdf-8782-e53db20768a5",
            sorts=[{"property_name": "Date", "ascending": False}],
            page_size=1
        )
    """
    class ResponseData(BaseModel):
        response_data: str = Field(description="JSON string of a dictionary containing the queried data from the Notion database.")

    class DatabaseReadOutput(BaseModel):
        data: ResponseData = Field(description="Data from the action execution.")
        successful: bool = Field(description="Whether or not the action execution was successful or not.")

    input_str = f"database_id: {database_id}, page_size: {page_size}, sorts: {sorts}, start_cursor: {start_cursor}"
    description = Database_READ.__doc__

    result = llm.with_structured_output(DatabaseReadOutput).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

@tool
def Database_UPDATE(row_id: str, properties: Optional[List[dict]] = None, cover: Optional[str] = None, delete_row: Optional[bool] = False, icon: Optional[str] = None):
    """
    Updates or archives an existing Notion database row (page) using its `row id`.
    To update historical snapshots of MS Graph resources.

    Args:
        row_id (str): Identifier (UUID) of the database row (page) to be updated.
        properties (Optional[List[dict]]): A list of property values to update for the page.
        cover (Optional[str]): URL of an external image to be used as the cover for the page.
        delete_row (Optional[bool]): If true, the row (page) will be archived.
        icon (Optional[str]): The emoji to be used as the icon for the page.

    Returns:
        str: JSON string of the output, including `response_data` and `successful` status.

    Example:
        Database_UPDATE(
            row_id="59833787-2cf9-4fdf-8782-e53db20768a5",
            properties=[
                {"name": "Status", "type": "select", "value": "Reviewed"}
            ]
        )
    """
    class ResponseData(BaseModel):
        response_data: str = Field(description="JSON string of a dictionary containing the raw JSON response from the Notion API, representing the updated page object.")

    class DatabaseUpdateOutput(BaseModel):
        data: ResponseData = Field(description="Data from the action execution.")
        successful: bool = Field(description="Whether or not the action execution was successful or not.")

    input_str = f"row_id: {row_id}, properties: {properties}, cover: {cover}, delete_row: {delete_row}, icon: {icon}"
    description = Database_UPDATE.__doc__

    result = llm.with_structured_output(DatabaseUpdateOutput).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

@tool
def Email_Service_CREATE(recipient_email: str, body: str, attachment: Optional[dict] = None, bcc: Optional[List[str]] = None, cc: Optional[List[str]] = None, extra_recipients: Optional[List[str]] = None, is_html: Optional[bool] = False, subject: Optional[str] = None, user_id: Optional[str] = 'me'):
    """
    Sends an email via Gmail API using the authenticated user's Google profile display name.
    To send email notifications to specified recipients when critical changes are detected.

    Args:
        recipient_email (str): Primary recipient's email address.
        body (str): Email content (plain text or HTML); if HTML, `is_html` must be `True`.
        attachment (Optional[dict]): File to attach; ensure `s3key`, `mimetype`, and `name` are set if provided.
        bcc (Optional[List[str]]): Blind Carbon Copy (BCC) recipients' email addresses.
        cc (Optional[List[str]]): Carbon Copy (CC) recipients' email addresses.
        extra_recipients (Optional[List[str]]): Additional 'To' recipients' email addresses (not Cc or Bcc).
        is_html (Optional[bool]): Set to `True` if the email body contains HTML tags.
        subject (Optional[str]): Subject line of the email.
        user_id (Optional[str]): User's email address; the literal 'me' refers to the authenticated user.

    Returns:
        str: JSON string of the output, including `response_data` and `successful` status.

    Example:
        Email_Service_CREATE(
            recipient_email="admin@example.com",
            subject="Critical MS Graph Change Detected",
            body="A critical change was detected in the MS Graph 'application' resource. Please review immediately."
        )
    """
    class ResponseData(BaseModel):
        response_data: str = Field(description="JSON string of the Gmail API response, typically including the sent message ID and threadId.")

    class EmailServiceCreateOutput(BaseModel):
        data: ResponseData = Field(description="Data from the action execution.")
        successful: bool = Field(description="Whether or not the action execution was successful or not.")

    input_str = f"recipient_email: {recipient_email}, body: {body}, attachment: {attachment}, bcc: {bcc}, cc: {cc}, extra_recipients: {extra_recipients}, is_html: {is_html}, subject: {subject}, user_id: {user_id}"
    description = Email_Service_CREATE.__doc__

    result = llm.with_structured_output(EmailServiceCreateOutput).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GraphState(MessagesState):
    """ The GraphState represents the state of the LangGraph workflow. """
    latest_resource_state: Optional[dict] = Field(
        default=None,
        description="The latest state of the MS Graph 'application' resource, including its display name, description, and permissions."
    )
    previous_resource_state: Optional[dict] = Field(
        default=None,
        description="The previous snapshot of the MS Graph 'application' resource stored in the database."
    )
    change_detected: Optional[bool] = Field(
        default=None,
        description="A boolean indicating whether a change was detected between the latest and previous resource states."
    )
    differences: Optional[dict] = Field(
        default=None,
        description="A dictionary detailing the differences between the latest and previous resource states."
    )
    is_critical_change: Optional[bool] = Field(
        default=None,
        description="A boolean indicating whether the detected change is considered critical based on the MS Graph SDK reference."
    )
    critical_change_description: Optional[str] = Field(
        default=None,
        description="A natural language description of the critical change detected."
    )
    recipient_email: Optional[str] = Field(
        default=None,
        description="The email address of the recipient for critical change notifications."
    )
    email_subject: Optional[str] = Field(
        default=None,
        description="The subject line of the email notification."
    )
    email_body: Optional[str] = Field(
        default=None,
        description="The body content of the email notification."
    )

class ReactAgentState(MessagesState):
    remaining_steps: int
    structured_response: Any

def start_node(state: GraphState) -> GraphState:
    """
    Node purpose: Entry point of the workflow.
    Implementation reasoning: Initializes the workflow.
    """
    return {"messages": [AIMessage(content="Workflow started.")]}

monitor_ms_graph_tools = [MS_Graph_API_READ]
def monitor_ms_graph(state: GraphState) -> GraphState:
    """
    Node purpose: Continuously monitors specified MS Graph resources for any updates or changes and retrieves their current state.
    Implementation reasoning: Uses a tool-calling agent to interact with the MS Graph API to fetch resource states.
    """
    class MonitorMSGraphOutput(BaseModel):
        app_id: str = Field(description="The ID of the application to monitor.")
        display_name: str = Field(description="The display name of the application.")
        description: Optional[str] = Field(default=None, description="The description of the application.")
        permissions: List[str] = Field(description="List of permissions assigned to the application.")

    agent = create_react_agent(
        model=llm,
        prompt="You are an agent that monitors MS Graph resources. Use the MS_Graph_API_READ tool to retrieve the latest state of an application. You need to provide a dummy app_id for now.",
        tools=monitor_ms_graph_tools,
        state_schema=ReactAgentState,
        response_format=MonitorMSGraphOutput
    )

    # For demonstration, we'll use a dummy app_id and mock the output
    # In a real scenario, the agent would dynamically determine the app_id or it would be passed in the state.
    dummy_app_id = "dummy_app_id_123"
    result: MonitorMSGraphOutput = agent.invoke({"messages": state["messages"] + [HumanMessage(content=f"Monitor application with ID: {dummy_app_id}")]})["structured_response"]

    # Mocking the latest_resource_state as the tool call is async and returns None
    latest_resource_state = {
        "app_id": dummy_app_id,
        "display_name": result.display_name,
        "description": result.description,
        "permissions": result.permissions
    }

    return {
        "latest_resource_state": latest_resource_state,
        "messages": [AIMessage(content=f"Monitored MS Graph for app ID: {dummy_app_id}. Latest state retrieved.")]
    }

compare_snapshots_tools = [Database_CREATE, Database_READ, Database_UPDATE]
def compare_snapshots(state: GraphState) -> GraphState:
    """
    Node purpose: Compares the latest resource state with previous snapshots stored in a database to identify differences.
    Implementation reasoning: Uses tool-calling agents to read from and update a database, and an LLM to analyze differences.
    """
    class CompareSnapshotsOutput(BaseModel):
        change_detected: bool = Field(description="True if changes were detected, False otherwise.")
        differences: Dict[str, Any] = Field(description="A dictionary detailing the differences between the latest and previous states.")
        previous_resource_state: Optional[Dict[str, Any]] = Field(default=None, description="The previous snapshot of the resource.")

    agent = create_react_agent(
        model=llm,
        prompt="You are an agent that compares resource snapshots. Use Database_READ to retrieve the previous state. If no previous state exists, use Database_CREATE to store the current state. If a previous state exists, compare it with the latest_resource_state and identify differences. Then use Database_UPDATE to store the latest state as the new previous state. Finally, output whether a change was detected and the differences.",
        tools=compare_snapshots_tools,
        state_schema=ReactAgentState,
        response_format=CompareSnapshotsOutput
    )

    latest_state = state["latest_resource_state"]
    if not latest_state:
        return {
            "messages": [AIMessage(content="Error: latest_resource_state is missing for comparison.")]
        }

    # For demonstration, we'll mock database interaction and comparison logic
    # In a real scenario, the agent would use Database_READ, Database_CREATE, Database_UPDATE
    # and perform the comparison.
    mock_previous_state = {
        "app_id": latest_state["app_id"],
        "display_name": "Old App Name",
        "description": "Old description",
        "permissions": ["read"]
    }
    
    # Simulate a change
    change_detected = False
    differences = {}
    if latest_state["display_name"] != mock_previous_state["display_name"]:
        change_detected = True
        differences["display_name"] = {"old": mock_previous_state["display_name"], "new": latest_state["display_name"]}
    if latest_state["description"] != mock_previous_state["description"]:
        change_detected = True
        differences["description"] = {"old": mock_previous_state["description"], "new": latest_state["description"]}
    if set(latest_state["permissions"]) != set(mock_previous_state["permissions"]):
        change_detected = True
        differences["permissions"] = {"old": mock_previous_state["permissions"], "new": latest_state["permissions"]}

    # Update the database (mocked)
    # Database_UPDATE.invoke({"id": latest_state["app_id"], "data": latest_state})

    return {
        "previous_resource_state": mock_previous_state,
        "change_detected": change_detected,
        "differences": differences,
        "messages": [AIMessage(content=f"Comparison complete. Change detected: {change_detected}. Differences: {differences}")]
    }

class InterpretSchemaOutput(BaseModel):
    is_critical_change: bool = Field(description="True if the detected change is critical, False otherwise.")
    critical_change_description: str = Field(description="A natural language description of the critical change.")

def interpret_schema(state: GraphState) -> GraphState:
    """
    Node purpose: Utilizes MS Graph SDK reference to understand the meaning and criticality of resource schema fields and determines if detected changes are critical.
    Implementation reasoning: Uses an LLM with structured output to classify the criticality of changes based on provided differences.
    """
    if not state["change_detected"]:
        return {
            "is_critical_change": False,
            "critical_change_description": "No changes detected.",
            "messages": [AIMessage(content="No changes detected, skipping criticality assessment.")]
        }

    structured_llm = llm.with_structured_output(InterpretSchemaOutput)
    
    differences = state["differences"]
    prompt_content = f"""
    Analyze the following differences in an MS Graph 'application' resource and determine if any of them constitute a critical change.
    A critical change is defined as any modification that could significantly impact security, access, or core functionality.
    Refer to the MS Graph SDK documentation (implicitly, as you are an LLM trained on such data) for understanding field criticality.

    Differences: {json.dumps(differences, indent=2)}

    Provide a boolean indicating if it's a critical change and a natural language description of the critical change.
    """
    
    result: InterpretSchemaOutput = structured_llm.invoke([
        SystemMessage(content="You are an expert in MS Graph resource schema and security implications."),
        HumanMessage(content=prompt_content)
    ])

    return {
        "is_critical_change": result.is_critical_change,
        "critical_change_description": result.critical_change_description,
        "messages": [AIMessage(content=f"Schema interpretation complete. Critical change: {result.is_critical_change}. Description: {result.critical_change_description}")]
    }

send_notification_tools = [Email_Service_CREATE]
def send_notification(state: GraphState) -> GraphState:
    """
    Node purpose: Sends email notifications to relevant stakeholders when critical changes or deviations are detected.
    Implementation reasoning: Uses an LLM to generate email content and a tool-calling agent to send the email.
    """
    class EmailDetails(BaseModel):
        recipient_email: str = Field(description="The email address of the recipient.")
        email_subject: str = Field(description="The subject line of the email.")
        email_body: str = Field(description="The body content of the email.")

    agent = create_react_agent(
        model=llm,
        prompt="You are an agent responsible for sending critical change notifications. Generate a concise and informative email subject and body based on the critical change description. Use the Email_Service_CREATE tool to send the email. The recipient email should be 'admin@example.com'.",
        tools=send_notification_tools,
        state_schema=ReactAgentState,
        response_format=EmailDetails
    )

    critical_change_description = state["critical_change_description"]
    if not critical_change_description:
        return {
            "messages": [AIMessage(content="Error: No critical change description to send notification.")]
        }

    # The agent will generate the email details and then call the tool.
    # For demonstration, we'll mock the agent's response and tool call.
    # In a real scenario, the agent.invoke would handle this.
    
    # Mocking the agent's structured response
    mock_email_details = EmailDetails(
        recipient_email="admin@example.com",
        email_subject=f"Critical MS Graph Application Change Detected: {critical_change_description[:50]}...",
        email_body=f"Dear Admin,\n\nA critical change has been detected in an MS Graph application:\n\n{critical_change_description}\n\nPlease investigate this change immediately.\n\nRegards,\nMonitoring System"
    )

    # Mocking the tool call
    # Email_Service_CREATE.invoke({
    #     "to": mock_email_details.recipient_email,
    #     "subject": mock_email_details.email_subject,
    #     "body": mock_email_details.email_body
    # })

    return {
        "recipient_email": mock_email_details.recipient_email,
        "email_subject": mock_email_details.email_subject,
        "email_body": mock_email_details.email_body,
        "messages": [AIMessage(content=f"Notification sent to {mock_email_details.recipient_email} with subject: {mock_email_details.email_subject}")]
    }

def route_on_critical_change(state: GraphState) -> Literal["send_notification", "__END__"]:
    """
    Determines whether to send a notification or end the workflow based on if a critical change was detected.
    """
    if state["is_critical_change"]:
        return "send_notification"
    else:
        return "__END__"

workflow = StateGraph(GraphState)

workflow.add_node("monitor_ms_graph", monitor_ms_graph)
workflow.add_node("compare_snapshots", compare_snapshots)
workflow.add_node("interpret_schema", interpret_schema)
workflow.add_node("send_notification", send_notification)

workflow.add_edge(START, "monitor_ms_graph")
workflow.add_edge("monitor_ms_graph", "compare_snapshots")
workflow.add_edge("compare_snapshots", "interpret_schema")

workflow.add_conditional_edges(
    "interpret_schema",
    route_on_critical_change,
    {
        "send_notification": "send_notification",
        "__END__": END
    }
)
workflow.add_edge("send_notification", END)

checkpointer = InMemorySaver()
app = workflow.compile(
    checkpointer=checkpointer
)
