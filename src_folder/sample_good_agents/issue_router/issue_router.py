from typing import Dict, Any, List, Optional, Literal, TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
import os

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
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

def GitHub_Issue_Creator_PromptiusWeb(owner: str, repo: str, title: str, assignee: Optional[str] = None, assignees: Optional[List[str]] = None, body: Optional[str] = None, labels: Optional[List[str]] = None, milestone: Optional[str] = None) -> str:
    """
    Creates a new issue in a github repository, requiring the repository to exist and have issues enabled; specific fields like assignees, milestone, or labels may require push access.

    Args:
        owner (str): The GitHub account owner of the repository (case-insensitive).
        repo (str): The name of the repository, without the `.git` extension (case-insensitive).
        title (str): The title for the new issue.
        assignee (Optional[str]): Login for the user to whom this issue should be assigned. NOTE: Only users with push access can set the assignee; it is silently dropped otherwise. **This field is deprecated in favor of `assignees`.**
        assignees (Optional[List[str]]): GitHub login names for users to assign to this issue. NOTE: Only users with push access can set assignees; they are silently dropped otherwise.
        body (Optional[str]): The detailed textual contents of the new issue.
        labels (Optional[List[str]]): Label names to associate with this issue (generally case-insensitive). NOTE: Only users with push access can set labels; they are silently dropped otherwise. Pass an empty list to clear all labels.
        milestone (Optional[str]): The ID of the milestone to associate this issue with (e.g., "5"). NOTE: Only users with push access can set the milestone; it is silently dropped otherwise.

    Returns:
        str: JSON string of the output.

    Example:
        GitHub_Issue_Creator_PromptiusWeb(owner='octocat', repo='PromptiusWeb', title='Bug: Login button not working', body='The login button on the homepage does not respond when clicked.', labels=['bug', 'UI'])
    """
    class Data(BaseModel):
        data: str = Field(description="A dictionary containing the full data representation of the newly created GitHub issue, including its ID, title, body, state, assignees, labels, etc. (JSON string)")
        successful: bool = Field(description="Whether or not the action execution was successful or not")
        error: Optional[str] = Field(description="Error if any occurred during the execution of the action")

    input_str = f"owner: {owner}, repo: {repo}, title: {title}, assignee: {assignee}, assignees: {assignees}, body: {body}, labels: {labels}, milestone: {milestone}"
    description = GitHub_Issue_Creator_PromptiusWeb.__doc__

    result = llm.with_structured_output(Data).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

def GitHub_Issue_Creator_AgentAgent(owner: str, repo: str, title: str, assignee: Optional[str] = None, assignees: Optional[List[str]] = None, body: Optional[str] = None, labels: Optional[List[str]] = None, milestone: Optional[str] = None) -> str:
    """
    Creates a new issue in a github repository, requiring the repository to exist and have issues enabled; specific fields like assignees, milestone, or labels may require push access.

    Args:
        owner (str): The GitHub account owner of the repository (case-insensitive).
        repo (str): The name of the repository, without the `.git` extension (case-insensitive).
        title (str): The title for the new issue.
        assignee (Optional[str]): Login for the user to whom this issue should be assigned. NOTE: Only users with push access can set the assignee; it is silently dropped otherwise. **This field is deprecated in favor of `assignees`.**
        assignees (Optional[List[str]]): GitHub login names for users to assign to this issue. NOTE: Only users with push access can set assignees; they are silently dropped otherwise.
        body (Optional[str]): The detailed textual contents of the new issue.
        labels (Optional[List[str]]): Label names to associate with this issue (generally case-insensitive). NOTE: Only users with push access can set labels; they are silently dropped otherwise. Pass an empty list to clear all labels.
        milestone (Optional[str]): The ID of the milestone to associate this issue with (e.g., "5"). NOTE: Only users with push access can set the milestone; it is silently dropped otherwise.

    Returns:
        str: JSON string of the output.

    Example:
        GitHub_Issue_Creator_AgentAgent(owner='octocat', repo='AgentAgent', title='Bug: Agent not responding', body='The agent fails to respond to certain prompts, leading to a deadlock.', labels=['bug', 'backend'])
    """
    class Data(BaseModel):
        data: str = Field(description="A dictionary containing the full data representation of the newly created GitHub issue, including its ID, title, body, state, assignees, labels, etc. (JSON string)")
        successful: bool = Field(description="Whether or not the action execution was successful or not")
        error: Optional[str] = Field(description="Error if any occurred during the execution of the action")

    input_str = f"owner: {owner}, repo: {repo}, title: {title}, assignee: {assignee}, assignees: {assignees}, body: {body}, labels: {labels}, milestone: {milestone}"
    description = GitHub_Issue_Creator_AgentAgent.__doc__

    result = llm.with_structured_output(Data).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

def Zendesk_Ticket_Creator(subject: str, description: str, priority: Optional[str] = 'normal', requester_email: Optional[str] = None, requester_name: Optional[str] = None) -> str:
    """
    Create a ticket in zendesk.

    Args:
        subject (str): Short summary of the issue. Keep it concise (e.g. 'Cannot log in').
        description (str): Long-form description / steps to reproduce.
        priority (Optional[str]): Zendesk priority. Must be one of: 'urgent', 'high', 'normal', 'low'.
        requester_email (Optional[str]): Email of the requester. Must accompany `requester_name`.
        requester_name (Optional[str]): Name of the requester. If you supply this, you MUST also supply `requester_email`. Leave both blank to default to the authenticated user.

    Returns:
        str: JSON string of the output.

    Example:
        Zendesk_Ticket_Creator(subject='Unable to log in', description='Customer cannot log in to their account after password reset.', priority='high', requester_email='customer@example.com', requester_name='John Doe')
    """
    class Data(BaseModel):
        data: str = Field(description="Data from the action execution (JSON string)")
        successful: bool = Field(description="Whether or not the action execution was successful or not")
        error: Optional[str] = Field(description="Error if any occurred during the execution of the action")

    input_str = f"subject: {subject}, description: {description}, priority: {priority}, requester_email: {requester_email}, requester_name: {requester_name}"
    description = Zendesk_Ticket_Creator.__doc__

    result = llm.with_structured_output(Data).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

def Email_Sender(recipient_email: str, body: str, attachment: Optional[str] = None, bcc: Optional[List[str]] = None, cc: Optional[List[str]] = None, extra_recipients: Optional[List[str]] = None, is_html: Optional[bool] = False, subject: Optional[str] = None, user_id: Optional[str] = 'me') -> str:
    """
    Sends an email via gmail api using the authenticated user's google profile display name, requiring `is html=true` if the body contains html and valid `s3key`, `mimetype`, `name` for any attachment.

    Args:
        recipient_email (str): Primary recipient's email address.
        body (str): Email content (plain text or HTML); if HTML, `is_html` must be `True`.
        attachment (Optional[str]): File to attach; ensure `s3key`, `mimetype`, and `name` are set if provided. Omit or set to null for no attachment. (JSON string of the FileUploadable object)
        bcc (Optional[List[str]]): Blind Carbon Copy (BCC) recipients' email addresses.
        cc (Optional[List[str]]): Carbon Copy (CC) recipients' email addresses.
        extra_recipients (Optional[List[str]]): Additional 'To' recipients' email addresses (not Cc or Bcc).
        is_html (Optional[bool]): Set to `True` if the email body contains HTML tags.
        subject (Optional[str]): Subject line of the email.
        user_id (Optional[str]): User's email address; the literal 'me' refers to the authenticated user.

    Returns:
        str: JSON string of the output.

    Example:
        Email_Sender(recipient_email='user@example.com', subject='Issue Created', body='A new GitHub issue has been created for your request.', is_html=False)
    """
    class Data(BaseModel):
        data: str = Field(description="Data from the action execution (JSON string)")
        successful: bool = Field(description="Whether or not the action execution was successful or not")
        error: Optional[str] = Field(description="Error if any occurred during the execution of the action")

    input_str = f"recipient_email: {recipient_email}, body: {body}, attachment: {attachment}, bcc: {bcc}, cc: {cc}, extra_recipients: {extra_recipients}, is_html: {is_html}, subject: {subject}, user_id: {user_id}"
    description = Email_Sender.__doc__

    result = llm.with_structured_output(Data).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

def Email_Reader(api_key: str, from_email: str, from_name: str, to_email: str, to_name: str, subject: str, html_content: str) -> str:
    """
    To receive and read incoming emails from customers.

    Args:
        api_key (str): Your MailerSend API key.
        from_email (str): The sender's email address.
        from_name (str): The sender's name.
        to_email (str): The recipient's email address.
        to_name (str): The recipient's name.
        subject (str): The subject of the email.
        html_content (str): The HTML content of the email.

    Returns:
        str: The response from the MailerSend API.

    Example:
        Email_Reader(api_key='your_api_key', from_email='sender@example.com', from_name='Sender Name', to_email='your_email@example.com', to_name='Your Name', subject='Website Issue', html_content='<p>The website login is broken.</p>')
    """
    class EmailReadResponse(BaseModel):
        response: str = Field(description="The response from the MailerSend API.")

    input_str = f"api_key: {api_key}, from_email: {from_email}, from_name: {from_name}, to_email: {to_email}, to_name: {to_name}, subject: {subject}, html_content: {html_content}"
    description = Email_Reader.__doc__

    result = llm.with_structured_output(EmailReadResponse).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

def GitHub_Issue_Search(token: str, owner: str, repo: str) -> str:
    """
    To search for existing issues within a specified GitHub repository.

    Args:
        token (str): Your GitHub personal access token.
        owner (str): The owner of the repository.
        repo (str): The name of the repository (e.g., 'PromptiusWeb' or 'AgentAgent').

    Returns:
        str: A formatted string of found issues or a failure message.

    Example:
        GitHub_Issue_Search(token='your_github_token', owner='octocat', repo='PromptiusWeb')
    """
    class GitHubIssueSearchResponse(BaseModel):
        issues_found: str = Field(description="A formatted string of found issues or a failure message.")

    input_str = f"token: {token}, owner: {owner}, repo: {repo}"
    description = GitHub_Issue_Search.__doc__

    result = llm.with_structured_output(GitHubIssueSearchResponse).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GraphState(MessagesState):
    """
    The GraphState represents the state of the LangGraph workflow.
    """
    email_content: str
    email_type: Optional[str] = None
    issue_found: Optional[bool] = None
    issue_details: Optional[str] = None
    issue_created: Optional[bool] = None
    ticket_created: Optional[bool] = None
    ticket_details: Optional[str] = None
    email_sent: Optional[bool] = None

class EmailClassification(BaseModel):
    """Structured output for email classification."""
    email_type: Literal["website_issue", "general_complaint", "agentic_backend_issue"] = Field(description="Classified email type")
    reasoning: str = Field(description="Brief explanation of classification")

class GitHubIssueType(BaseModel):
    """Structured output for GitHub issue type classification."""
    issue_type: Literal["promptiusweb_ux", "agentagent_backend"] = Field(description="Classified GitHub issue type")
    reasoning: str = Field(description="Brief explanation of classification")

def read_and_analyze_email(state: GraphState) -> GraphState:
    """
    Node purpose: Reads and analyzes the incoming email content to determine if it's a website issue or a general customer complaint.
    Implementation reasoning: Uses an LLM with structured output to classify the email content for conditional routing.
    """
    structured_llm = llm.with_structured_output(EmailClassification)
    
    # Assuming email content is passed in the initial state or from a previous node
    email_content = state["messages"][-1].content
    
    prompt = f"""Analyze the following email content and classify it as either 'website_issue', 'general_complaint', or 'agentic_backend_issue'.
    Email content: {email_content}
    """
    
    classification_result: EmailClassification = structured_llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "email_type": classification_result.email_type,
        "email_content": email_content, # Store the extracted email content
        "messages": [AIMessage(content=f"Email classified as: {classification_result.email_type}. Reasoning: {classification_result.reasoning}")]
    }

def search_github_issues(state: GraphState) -> GraphState:
    """
    Node purpose: Searches for existing issues in the PromptiusWeb and AgentAgent GitHub repositories based on the email content.
    Implementation reasoning: Uses the GitHub_Issue_Search tool to query both repositories and determine if a relevant issue exists.
    """
    search_github_issues_tools = [GitHub_Issue_Search]

    class CustomStateForReact(MessagesState):
        remaining_steps: int
        structured_response: Any

    class SearchResult(BaseModel):
        issue_found: bool = Field(description="True if a relevant issue was found, False otherwise.")
        issue_details: Optional[str] = Field(description="Details of the found issue, or a message indicating no issue found.")

    agent = create_react_agent(
        model=llm,
        prompt="You are an agent that searches for GitHub issues. Use the 'GitHub_Issue_Search' tool to search for issues related to the provided email content in both 'PromptiusWeb' and 'AgentAgent' repositories. Prioritize finding existing issues before suggesting new ones. If an issue is found, provide its details. If not, state that no relevant issue was found.",
        tools=search_github_issues_tools,
        state_schema=CustomStateForReact,
        response_format=SearchResult
    )

    email_content = state["email_content"]
    
    # Invoke the agent to search for issues in both repositories
    # Assuming owner is 'your_github_username' or an organization name
    prompt_message = f"Search for GitHub issues related to: '{email_content}' in 'PromptiusWeb' and 'AgentAgent' repositories. Use the GitHub_Issue_Search tool with owner='your_github_username' and repo='PromptiusWeb' or 'AgentAgent'. Your GitHub token is {os.environ.get('GITHUB_TOKEN')}."
    
    result: SearchResult = agent.invoke({"messages": [HumanMessage(content=prompt_message)]})["structured_response"]

    return {
        "issue_found": result.issue_found,
        "issue_details": result.issue_details,
        "messages": [AIMessage(content=f"GitHub issue search completed. Issue found: {result.issue_found}. Details: {result.issue_details}")]
    }
def create_promptiusweb_issue(state: GraphState) -> GraphState:
    """
    Node purpose: Creates a new issue in the PromptiusWeb GitHub repository for website UX issues.
    Implementation reasoning: Uses the GitHub_Issue_Creator_PromptiusWeb tool to create a new issue.
    """
    create_promptiusweb_issue_tools = [GitHub_Issue_Creator_PromptiusWeb]

    class CustomStateForReact(MessagesState):
        remaining_steps: int
        structured_response: Any

    class IssueCreationResult(BaseModel):
        issue_created: bool = Field(description="True if the issue was successfully created, False otherwise.")
        issue_details: str = Field(description="Details of the created issue (e.g., URL, title) or an error message.")

    agent = create_react_agent(
        model=llm,
        prompt="You are an agent that creates GitHub issues. Use the 'GitHub_Issue_Creator_PromptiusWeb' tool to create a new issue in the 'PromptiusWeb' repository based on the email content. Provide a concise title and detailed description. Your GitHub token is {os.environ.get('GITHUB_TOKEN')}.",
        tools=create_promptiusweb_issue_tools,
        state_schema=CustomStateForReact,
        response_format=IssueCreationResult
    )

    email_content = state["email_content"]
    issue_title = f"Website UX Issue: {email_content[:50]}..."
    issue_body = f"User reported the following website UX issue:\n\n{email_content}"
    
    prompt_message = f"Create a GitHub issue in 'PromptiusWeb' repository with title '{issue_title}' and body '{issue_body}'. Use the GitHub_Issue_Creator_PromptiusWeb tool with owner='your_github_username', repo='PromptiusWeb', title='{issue_title}', body='{issue_body}', token='{os.environ.get('GITHUB_TOKEN')}'."
    
    result: IssueCreationResult = agent.invoke({"messages": [HumanMessage(content=prompt_message)]})["structured_response"]

    return {
        "issue_created": result.issue_created,
        "issue_details": result.issue_details,
        "messages": [AIMessage(content=f"PromptiusWeb issue creation completed. Issue created: {result.issue_created}. Details: {result.issue_details}")]
    }

def create_agentagent_issue(state: GraphState) -> GraphState:
    """
    Node purpose: Creates a new issue in the AgentAgent GitHub repository for agentic related backend issues.
    Implementation reasoning: Uses the GitHub_Issue_Creator_AgentAgent tool to create a new issue.
    """
    create_agentagent_issue_tools = [GitHub_Issue_Creator_AgentAgent]

    class CustomStateForReact(MessagesState):
        remaining_steps: int
        structured_response: Any

    class IssueCreationResult(BaseModel):
        issue_created: bool = Field(description="True if the issue was successfully created, False otherwise.")
        issue_details: str = Field(description="Details of the created issue (e.g., URL, title) or an error message.")

    agent = create_react_agent(
        model=llm,
        prompt="You are an agent that creates GitHub issues. Use the 'GitHub_Issue_Creator_AgentAgent' tool to create a new issue in the 'AgentAgent' repository based on the email content. Provide a concise title and detailed description. Your GitHub token is {os.environ.get('GITHUB_TOKEN')}.",
        tools=create_agentagent_issue_tools,
        state_schema=CustomStateForReact,
        response_format=IssueCreationResult
    )

    email_content = state["email_content"]
    issue_title = f"AgentAgent Backend Issue: {email_content[:50]}..."
    issue_body = f"User reported the following agentic backend issue:\n\n{email_content}"
    
    prompt_message = f"Create a GitHub issue in 'AgentAgent' repository with title '{issue_title}' and body '{issue_body}'. Use the GitHub_Issue_Creator_AgentAgent tool with owner='your_github_username', repo='AgentAgent', title='{issue_title}', body='{issue_body}', token='{os.environ.get('GITHUB_TOKEN')}'."
    
    result: IssueCreationResult = agent.invoke({"messages": [HumanMessage(content=prompt_message)]})["structured_response"]

    return {
        "issue_created": result.issue_created,
        "issue_details": result.issue_details,
        "messages": [AIMessage(content=f"AgentAgent issue creation completed. Issue created: {result.issue_created}. Details: {result.issue_details}")]
    }

def create_zendesk_ticket(state: GraphState) -> GraphState:
    """
    Node purpose: Creates a new support ticket in Zendesk for general customer complaints.
    Implementation reasoning: Uses the Zendesk_Ticket_Creator tool to create a new ticket.
    """
    create_zendesk_ticket_tools = [Zendesk_Ticket_Creator]

    class CustomStateForReact(MessagesState):
        remaining_steps: int
        structured_response: Any

    class TicketCreationResult(BaseModel):
        ticket_created: bool = Field(description="True if the ticket was successfully created, False otherwise.")
        ticket_details: str = Field(description="Details of the created ticket (e.g., ID, subject) or an error message.")

    agent = create_react_agent(
        model=llm,
        prompt="You are an agent that creates Zendesk tickets. Use the 'Zendesk_Ticket_Creator' tool to create a new support ticket based on the email content. Provide a concise subject and detailed description. Your Zendesk API key is {os.environ.get('ZENDESK_API_KEY')}.",
        tools=create_zendesk_ticket_tools,
        state_schema=CustomStateForReact,
        response_format=TicketCreationResult
    )

    email_content = state["email_content"]
    ticket_subject = f"Customer Complaint: {email_content[:50]}..."
    ticket_description = f"User reported the following complaint:\n\n{email_content}"
    
    prompt_message = f"Create a Zendesk ticket with subject '{ticket_subject}' and description '{ticket_description}'. Use the Zendesk_Ticket_Creator tool with subject='{ticket_subject}', description='{ticket_description}', api_key='{os.environ.get('ZENDESK_API_KEY')}'."
    
    result: TicketCreationResult = agent.invoke({"messages": [HumanMessage(content=prompt_message)]})["structured_response"]

    return {
        "ticket_created": result.ticket_created,
        "ticket_details": result.ticket_details,
        "messages": [AIMessage(content=f"Zendesk ticket creation completed. Ticket created: {result.ticket_created}. Details: {result.ticket_details}")]
    }

def send_email_existing_issue(state: GraphState) -> GraphState:
    """
    Node purpose: Sends an email to the user informing them about an existing GitHub issue.
    Implementation reasoning: Uses the Email_Sender tool to send a confirmation email.
    """
    send_email_existing_issue_tools = [Email_Sender]

    class CustomStateForReact(MessagesState):
        remaining_steps: int
        structured_response: Any

    class EmailSendResult(BaseModel):
        email_sent: bool = Field(description="True if the email was successfully sent, False otherwise.")
        message: str = Field(description="Confirmation message or error details.")

    agent = create_react_agent(
        model=llm,
        prompt="You are an agent that sends emails. Use the 'Email_Sender' tool to inform the user about an existing GitHub issue. The email should include details about the issue. Your MailerSend API key is {os.environ.get('MAILERSEND_API_KEY')}.",
        tools=send_email_existing_issue_tools,
        state_schema=CustomStateForReact,
        response_format=EmailSendResult
    )

    email_content = state["email_content"]
    issue_details = state["issue_details"]
    
    # Assuming we can extract recipient email from initial email_content or it's a fixed value
    recipient_email = "user@example.com" # Placeholder, replace with actual extraction logic
    email_subject = "Regarding your inquiry: Existing GitHub Issue Found"
    email_body = f"Dear User,\n\nThank you for contacting us. We found an existing GitHub issue related to your inquiry:\n\n{issue_details}\n\nWe will keep you updated on its progress.\n\nBest regards,\nSupport Team"
    
    prompt_message = f"Send an email to '{recipient_email}' with subject '{email_subject}' and HTML content '{email_body}'. Use the Email_Sender tool with api_key='{os.environ.get('MAILERSEND_API_KEY')}', from_email='support@example.com', from_name='Support Team', to_email='{recipient_email}', to_name='User', subject='{email_subject}', html_content='{email_body}'."
    
    result: EmailSendResult = agent.invoke({"messages": [HumanMessage(content=prompt_message)]})["structured_response"]

    return {
        "email_sent": result.email_sent,
        "messages": [AIMessage(content=f"Email for existing issue sent. Status: {result.email_sent}. Message: {result.message}")]
    }

def send_email_new_issue(state: GraphState) -> GraphState:
    """
    Node purpose: Sends an email to the user informing them that a new GitHub issue has been created.
    Implementation reasoning: Uses the Email_Sender tool to send a confirmation email.
    """
    send_email_new_issue_tools = [Email_Sender]

    class CustomStateForReact(MessagesState):
        remaining_steps: int
        structured_response: Any

    class EmailSendResult(BaseModel):
        email_sent: bool = Field(description="True if the email was successfully sent, False otherwise.")
        message: str = Field(description="Confirmation message or error details.")

    agent = create_react_agent(
        model=llm,
        prompt="You are an agent that sends emails. Use the 'Email_Sender' tool to inform the user that a new GitHub issue has been created. The email should include details about the new issue. Your MailerSend API key is {os.environ.get('MAILERSEND_API_KEY')}.",
        tools=send_email_new_issue_tools,
        state_schema=CustomStateForReact,
        response_format=EmailSendResult
    )

    email_content = state["email_content"]
    issue_details = state["issue_details"]
    
    # Assuming we can extract recipient email from initial email_content or it's a fixed value
    recipient_email = "user@example.com" # Placeholder, replace with actual extraction logic
    email_subject = "Regarding your inquiry: New GitHub Issue Created"
    email_body = f"Dear User,\n\nThank you for contacting us. A new GitHub issue has been created for your inquiry:\n\n{issue_details}\n\nWe will keep you updated on its progress.\n\nBest regards,\nSupport Team"
    
    prompt_message = f"Send an email to '{recipient_email}' with subject '{email_subject}' and HTML content '{email_body}'. Use the Email_Sender tool with api_key='{os.environ.get('MAILERSEND_API_KEY')}', from_email='support@example.com', from_name='Support Team', to_email='{recipient_email}', to_name='User', subject='{email_subject}', html_content='{email_body}'."
    
    result: EmailSendResult = agent.invoke({"messages": [HumanMessage(content=prompt_message)]})["structured_response"]

    return {
        "email_sent": result.email_sent,
        "messages": [AIMessage(content=f"Email for new issue sent. Status: {result.email_sent}. Message: {result.message}")]
    }

def send_email_ticket_confirmation(state: GraphState) -> GraphState:
    """
    Node purpose: Sends an email to the user confirming the creation of a new support ticket.
    Implementation reasoning: Uses the Email_Sender tool to send a confirmation email.
    """
    send_email_ticket_confirmation_tools = [Email_Sender]

    class CustomStateForReact(MessagesState):
        remaining_steps: int
        structured_response: Any

    class EmailSendResult(BaseModel):
        email_sent: bool = Field(description="True if the email was successfully sent, False otherwise.")
        message: str = Field(description="Confirmation message or error details.")

    agent = create_react_agent(
        model=llm,
        prompt="You are an agent that sends emails. Use the 'Email_Sender' tool to inform the user about the creation of a new support ticket. The email should include details about the ticket. Your MailerSend API key is {os.environ.get('MAILERSEND_API_KEY')}.",
        tools=send_email_ticket_confirmation_tools,
        state_schema=CustomStateForReact,
        response_format=EmailSendResult
    )

    email_content = state["email_content"]
    ticket_details = state["ticket_details"]
    
    # Assuming we can extract recipient email from initial email_content or it's a fixed value
    recipient_email = "user@example.com" # Placeholder, replace with actual extraction logic
    email_subject = "Regarding your inquiry: Support Ticket Created"
    email_body = f"Dear User,\n\nThank you for contacting us. A new support ticket has been created for your inquiry:\n\n{ticket_details}\n\nWe will keep you updated on its progress.\n\nBest regards,\nSupport Team"
    
    prompt_message = f"Send an email to '{recipient_email}' with subject '{email_subject}' and HTML content '{email_body}'. Use the Email_Sender tool with api_key='{os.environ.get('MAILERSEND_API_KEY')}', from_email='support@example.com', from_name='Support Team', to_email='{recipient_email}', to_name='User', subject='{email_subject}', html_content='{email_body}'."
    
    result: EmailSendResult = agent.invoke({"messages": [HumanMessage(content=prompt_message)]})["structured_response"]

    return {
        "email_sent": result.email_sent,
        "messages": [AIMessage(content=f"Email for ticket confirmation sent. Status: {result.email_sent}. Message: {result.message}")]
    }

def route_email_type(state: GraphState) -> Literal["search_github_issues", "create_zendesk_ticket"]:
    """
    Routing function: Determines whether to search GitHub issues or create a Zendesk ticket based on email type.
    """
    if state["email_type"] in ["website_issue", "agentic_backend_issue"]:
        return "search_github_issues"
    elif state["email_type"] == "general_complaint":
        return "create_zendesk_ticket"
    else:
        # Default or error handling, though classification should cover all cases
        return "create_zendesk_ticket" # Fallback

def route_github_issue_status(state: GraphState) -> Literal["send_email_existing_issue", "create_promptiusweb_issue", "create_agentagent_issue"]:
    """
    Routing function: Determines whether to inform about an existing issue or create a new one.
    If no issue is found, it further classifies the issue type to route to the correct GitHub repository.
    """
    if state["issue_found"]:
        return "send_email_existing_issue"
    else:
        # If no issue found, classify the type of issue to create
        structured_llm = llm.with_structured_output(GitHubIssueType)
        email_content = state["email_content"]
        prompt = f"""Analyze the following email content and classify it as either 'promptiusweb_ux' (for website user experience issues) or 'agentagent_backend' (for agentic related backend issues).
        Email content: {email_content}
        """
        classification_result: GitHubIssueType = structured_llm.invoke([HumanMessage(content=prompt)])
        
        if classification_result.issue_type == "promptiusweb_ux":
            return "create_promptiusweb_issue"
        elif classification_result.issue_type == "agentagent_backend":
            return "create_agentagent_issue"
        else:
            # Fallback, though classification should cover all cases
            return "create_promptiusweb_issue"


workflow = StateGraph(GraphState)

workflow.add_node("read_email", read_and_analyze_email)
workflow.add_node("search_github_issues", search_github_issues)
workflow.add_node("create_promptiusweb_issue", create_promptiusweb_issue)
workflow.add_node("create_agentagent_issue", create_agentagent_issue)
workflow.add_node("create_zendesk_ticket", create_zendesk_ticket)
workflow.add_node("send_email_existing_issue", send_email_existing_issue)
workflow.add_node("send_email_new_issue", send_email_new_issue)
workflow.add_node("send_email_ticket_confirmation", send_email_ticket_confirmation)

workflow.add_edge(START, "read_email")

workflow.add_conditional_edges(
    "read_email",
    route_email_type,
    {
        "search_github_issues": "search_github_issues",
        "create_zendesk_ticket": "create_zendesk_ticket"
    }
)

workflow.add_conditional_edges(
    "search_github_issues",
    route_github_issue_status,
    {
        "send_email_existing_issue": "send_email_existing_issue",
        "create_promptiusweb_issue": "create_promptiusweb_issue",
        "create_agentagent_issue": "create_agentagent_issue"
    }
)

workflow.add_edge("send_email_existing_issue", END)
workflow.add_edge("create_promptiusweb_issue", "send_email_new_issue")
workflow.add_edge("create_agentagent_issue", "send_email_new_issue")
workflow.add_edge("create_zendesk_ticket", "send_email_ticket_confirmation")
workflow.add_edge("send_email_new_issue", END)
workflow.add_edge("send_email_ticket_confirmation", END)

app = workflow.compile(
)