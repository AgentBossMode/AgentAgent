github_mock_tools = r'''
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class EmailClientReadOutput(BaseModel):
    """
    Output schema for Email_Client_Read tool.
    """
    output: str = Field(description="JSON string of the output from the Email_Client_Read tool.")

class GitHubCreateIssueOutput(BaseModel):
    """
    Output schema for GitHub_Create_PromptiusWeb and GitHub_Create_AgentAgent tools.
    """
    output: str = Field(description="JSON string of the output from the GitHub_Create_Issue tool.")

class TicketingSystemCreateOutput(BaseModel):
    """
    Output schema for Ticketing_System_Create tool.
    """
    output: str = Field(description="JSON string of the output from the Ticketing_System_Create tool.")

class EmailClientSendOutput(BaseModel):
    """
    Output schema for Email_Client_Send tool.
    """
    output: str = Field(description="JSON string of the output from the Email_Client_Send tool.")

class GitHubSearchOutput(BaseModel):
    """
    Output schema for GitHub_Search tool.
    """
    output: str = Field(description="A string containing information about the found issues.")


MOCK_TOOL_PROMPT = """
You are a helpful assistant that generates mock data for tool outputs.
Given the tool's purpose and expected output, generate a realistic mock response.
"""

INPUT_PROMPT = """
Tool Docstring: {description}
Input: {input}
Generate a mock output for this tool.
"""

def Email_Client_Read(
    user_id: Optional[str] = 'me',
    ids_only: Optional[bool] = False,
    include_payload: Optional[bool] = True,
    include_spam_trash: Optional[bool] = False,
    label_ids: Optional[List[str]] = None,
    max_results: Optional[int] = 1,
    page_token: Optional[str] = None,
    query: Optional[str] = None,
    verbose: Optional[bool] = True
) -> str:
    """
    Fetches a list of email messages from a gmail account, supporting filtering, pagination, and optional full content retrieval.

    Args:
        user_id (str, optional): User's email address or 'me' for the authenticated user. Defaults to 'me'.
        ids_only (bool, optional): If true, only returns message IDs from the list API without fetching individual message details. Fastest option for getting just message IDs and thread IDs. Defaults to False.
        include_payload (bool, optional): Set to true to include full message payload (headers, body, attachments); false for metadata only. Defaults to True.
        include_spam_trash (bool, optional): Set to true to include messages from 'SPAM' and 'TRASH'. Defaults to False.
        label_ids (List[str], optional): Filter by label IDs; only messages with all specified labels are returned. Defaults to None.
        max_results (int, optional): Maximum number of messages to retrieve per page. Defaults to 1.
        page_token (str, optional): Token for retrieving a specific page, obtained from a previous response's `nextPageToken`. Omit for the first page. Defaults to None.
        query (str, optional): Gmail advanced search query (e.g., 'from:user subject:meeting'). Defaults to None.
        verbose (bool, optional): If false, uses optimized concurrent metadata fetching for faster performance (~75% improvement). If true, uses standard detailed message fetching. When false, only essential fields (subject, sender, recipient, time, labels) are guaranteed. Defaults to True.

    Returns:
        str: JSON string of the output from the Email_Client_Read tool.

    Example:
        Email_Client_Read(query='is:unread', max_results=1)
    """
    input_str = f"user_id: {user_id}, ids_only: {ids_only}, include_payload: {include_payload}, include_spam_trash: {include_spam_trash}, label_ids: {label_ids}, max_results: {max_results}, page_token: {page_token}, query: {query}, verbose: {verbose}"
    description = Email_Client_Read.__doc__
    
    result = llm.with_structured_output(EmailClientReadOutput).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.output

def GitHub_Create_PromptiusWeb(
    owner: str,
    repo: str,
    title: str,
    assignee: Optional[str] = None,
    assignees: Optional[List[str]] = None,
    body: Optional[str] = None,
    labels: Optional[List[str]] = None,
    milestone: Optional[str] = None
) -> str:
    """
    Creates a new issue in a github repository, requiring the repository to exist and have issues enabled; specific fields like assignees, milestone, or labels may require push access.
    This tool is specifically for creating new issues in the 'PromptiusWeb' GitHub repository.

    Args:
        owner (str): The GitHub account owner of the repository (case-insensitive).
        repo (str): The name of the repository, without the `.git` extension (case-insensitive).
        title (str): The title for the new issue.
        assignee (str, optional): Login for the user to whom this issue should be assigned. Defaults to None.
        assignees (List[str], optional): GitHub login names for users to assign to this issue. Defaults to None.
        body (str, optional): The detailed textual contents of the new issue. Defaults to None.
        labels (List[str], optional): Label names to associate with this issue. Defaults to None.
        milestone (str, optional): The ID of the milestone to associate this issue with. Defaults to None.

    Returns:
        str: JSON string of the output from the GitHub_Create_Issue tool.

    Example:
        GitHub_Create_PromptiusWeb(owner='Promptius', repo='PromptiusWeb', title='Website UI Bug', body='The contact form is not submitting correctly.')
    """
    input_str = f"owner: {owner}, repo: {repo}, title: {title}, assignee: {assignee}, assignees: {assignees}, body: {body}, labels: {labels}, milestone: {milestone}"
    description = GitHub_Create_PromptiusWeb.__doc__
    
    result = llm.with_structured_output(GitHubCreateIssueOutput).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.output

def GitHub_Create_AgentAgent(
    owner: str,
    repo: str,
    title: str,
    assignee: Optional[str] = None,
    assignees: Optional[List[str]] = None,
    body: Optional[str] = None,
    labels: Optional[List[str]] = None,
    milestone: Optional[str] = None
) -> str:
    """
    Creates a new issue in a github repository, requiring the repository to exist and have issues enabled; specific fields like assignees, milestone, or labels may require push access.
    This tool is specifically for creating new issues in the 'AgentAgent' GitHub repository.

    Args:
        owner (str): The GitHub account owner of the repository (case-insensitive).
        repo (str): The name of the repository, without the `.git` extension (case-insensitive).
        title (str): The title for the new issue.
        assignee (str, optional): Login for the user to whom this issue should be assigned. Defaults to None.
        assignees (List[str], optional): GitHub login names for users to assign to this issue. Defaults to None.
        body (str, optional): The detailed textual contents of the new issue. Defaults to None.
        labels (List[str], optional): Label names to associate with this issue. Defaults to None.
        milestone (str, optional): The ID of the milestone to associate this issue with. Defaults to None.

    Returns:
        str: JSON string of the output from the GitHub_Create_Issue tool.

    Example:
        GitHub_Create_AgentAgent(owner='AgentCo', repo='AgentAgent', title='Agent Backend Error', body='The agent is failing to process requests due to a database connection issue.')
    """
    input_str = f"owner: {owner}, repo: {repo}, title: {title}, assignee: {assignee}, assignees: {assignees}, body: {body}, labels: {labels}, milestone: {milestone}"
    description = GitHub_Create_AgentAgent.__doc__
    
    result = llm.with_structured_output(GitHubCreateIssueOutput).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.output

def Ticketing_System_Create(
    subject: str,
    description: str,
    attachments: Optional[List] = None,
    cc_emails: Optional[List] = None,
    company_id: Optional[int] = None,
    custom_fields: Optional[dict] = None,
    due_by: Optional[str] = None,
    email: Optional[str] = None,
    email_config_id: Optional[int] = None,
    facebook_id: Optional[str] = None,
    fr_due_by: Optional[str] = None,
    group_id: Optional[int] = None,
    internal_agent_id: Optional[int] = None,
    internal_group_id: Optional[int] = None,
    lookup_parameter: Optional[str] = None,
    name: Optional[str] = None,
    phone: Optional[str] = None,
    priority: Optional[int] = 1,
    product_id: Optional[int] = None,
    requester_id: Optional[int] = None,
    responder_id: Optional[int] = None,
    source: Optional[int] = 2,
    status: Optional[int] = 2,
    tags: Optional[List] = None,
    twitter_id: Optional[str] = None,
    unique_external_id: Optional[str] = None
) -> str:
    """
    Creates a new ticket in Freshdesk.
    This tool is used to create support tickets for customer complaints that are not website or agent backend issues.

    Args:
        subject (str): Subject of the ticket.
        description (str): HTML content of the ticket.
        attachments (List, optional): Ticket attachments. Defaults to None.
        cc_emails (List, optional): Email addresses added in the 'cc' field of the incoming ticket email. Defaults to None.
        company_id (int, optional): Company ID of the requester. Defaults to None.
        custom_fields (dict, optional): Key value pairs containing the names and values of custom fields. Defaults to None.
        due_by (str, optional): Timestamp that denotes when the ticket is due to be resolved. Defaults to None.
        email (str, optional): Email address of the requester. Defaults to None.
        email_config_id (int, optional): ID of email config which is used for this ticket. Defaults to None.
        facebook_id (str, optional): Facebook ID of the requester. Defaults to None.
        fr_due_by (str, optional): Timestamp that denotes when the first response is due. Defaults to None.
        group_id (int, optional): ID of the group to which the ticket has been assigned. Defaults to None.
        internal_agent_id (int, optional): ID of the internal agent which the ticket should be assigned with. Defaults to None.
        internal_group_id (int, optional): ID of the internal group to which the ticket should be assigned with. Defaults to None.
        lookup_parameter (str, optional): Lookup field for custom objects. Defaults to None.
        name (str, optional): Name of the requester. Defaults to None.
        phone (str, optional): Phone number of the requester. Defaults to None.
        priority (int, optional): Priority of the ticket. Defaults to 1.
        product_id (int, optional): ID of the product linked to the ticket. Defaults to None.
        requester_id (int, optional): User ID of the requester. Defaults to None.
        responder_id (int, optional): ID of the agent to whom the ticket has been assigned. Defaults to None.
        source (int, optional): The channel through which the ticket was created. Defaults to 2.
        status (int, optional): Status of the ticket. Defaults to 2.
        tags (List, optional): Tags that have been associated with the ticket. Defaults to None.
        twitter_id (str, optional): Twitter handle of the requester. Defaults to None.
        unique_external_id (str, optional): External ID of the requester. Defaults to None.

    Returns:
        str: JSON string of the output from the Ticketing_System_Create tool.

    Example:
        Ticketing_System_Create(subject='General Inquiry', description='The user has a question about billing.')
    """
    input_str = f"subject: {subject}, description: {description}, attachments: {attachments}, cc_emails: {cc_emails}, company_id: {company_id}, custom_fields: {custom_fields}, due_by: {due_by}, email: {email}, email_config_id: {email_config_id}, facebook_id: {facebook_id}, fr_due_by: {fr_due_by}, group_id: {group_id}, internal_agent_id: {internal_agent_id}, internal_group_id: {internal_group_id}, lookup_parameter: {lookup_parameter}, name: {name}, phone: {phone}, priority: {priority}, product_id: {product_id}, requester_id: {requester_id}, responder_id: {responder_id}, source: {source}, status: {status}, tags: {tags}, twitter_id: {twitter_id}, unique_external_id: {unique_external_id}"
    description = Ticketing_System_Create.__doc__
    
    result = llm.with_structured_output(TicketingSystemCreateOutput).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.output

def Email_Client_Send(
    recipient_email: str,
    body: str,
    attachment: Optional[dict] = None,
    bcc: Optional[List[str]] = None,
    cc: Optional[List[str]] = None,
    extra_recipients: Optional[List[str]] = None,
    is_html: Optional[bool] = False,
    subject: Optional[str] = None,
    user_id: Optional[str] = 'me'
) -> str:
    """
    Sends an email via gmail api using the authenticated user's google profile display name, requiring `is html=true` if the body contains html and valid `s3key`, `mimetype`, `name` for any attachment.
    This tool is used to send automated email responses to users regarding the status of their reported issues.

    Args:
        recipient_email (str): Primary recipient's email address.
        body (str): Email content (plain text or HTML); if HTML, `is_html` must be `True`.
        attachment (dict, optional): File to attach; ensure `s3key`, `mimetype`, and `name` are set if provided. Defaults to None.
        bcc (List[str], optional): Blind Carbon Copy (BCC) recipients' email addresses. Defaults to None.
        cc (List[str], optional): Carbon Copy (CC) recipients' email addresses. Defaults to None.
        extra_recipients (List[str], optional): Additional 'To' recipients' email addresses (not Cc or Bcc). Defaults to None.
        is_html (bool, optional): Set to `True` if the email body contains HTML tags. Defaults to False.
        subject (str, optional): Subject line of the email. Defaults to None.
        user_id (str, optional): User's email address; the literal 'me' refers to the authenticated user. Defaults to 'me'.

    Returns:
        str: JSON string of the output from the Email_Client_Send tool.

    Example:
        Email_Client_Send(recipient_email='user@example.com', subject='Issue Update', body='Your issue #123 has been resolved.')
    """
    input_str = f"recipient_email: {recipient_email}, body: {body}, attachment: {attachment}, bcc: {bcc}, cc: {cc}, extra_recipients: {extra_recipients}, is_html: {is_html}, subject: {subject}, user_id: {user_id}"
    description = Email_Client_Send.__doc__
    
    result = llm.with_structured_output(EmailClientSendOutput).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.output

@tool
def GitHub_Search(owner: str, repo_name: str, search_terms: str = None) -> str:
    """
    Searches for existing issues within a specified GitHub repository.
    This tool is used to search for existing issues within the PromptiusWeb and AgentAgent repositories.

    Args:
        owner (str): The owner of the GitHub repository.
        repo_name (str): The name of the GitHub repository.
        search_terms (str, optional): Specific terms to search for within issues. Defaults to None.

    Returns:
        str: A string containing information about the found issues.

    Example:
        GitHub_Search(owner='Promptius', repo_name='PromptiusWeb', search_terms='bug in login')
    """
    input_str = f"owner: {owner}, repo_name: {repo_name}, search_terms: {search_terms}"
    description = GitHub_Search.__doc__
    
    result = llm.with_structured_output(GitHubSearchOutput).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.output'''