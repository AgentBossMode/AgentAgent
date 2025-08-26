#!/usr/bin/env python3
"""
Pytest-based testing script for LangGraph Validator.

This script runs comprehensive tests for the LangGraph validator functions.
"""

import pytest
from typing import Dict, Any, List
import time

# Import our validator modules
from src_folder.final_code.ast_visitors_lib.langgraph_parser import validate_langgraph_code
from src_folder.final_code.ast_visitors_lib.validation_script import run_detailed_validation, check_environment_variables, suggest_improvements


github_code = r'''
from typing import Dict, Any, List, Optional, Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
import re
import json
import os

from tools_code import Receive_Email, Search_Issues, Create_Issue, Create_Ticket, Send_Email

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GraphState(MessagesState):
    """ The GraphState represents the state of the LangGraph workflow. """
    email_content: str
    email_category: Optional[str] = None
    issue_keywords: Optional[List[str]] = None
    search_results: Optional[List[dict]] = None
    duplicate_issue_found: Optional[bool] = None
    issue_id: Optional[str] = None
    repository_name: Optional[str] = None
    issue_title: Optional[str] = None
    issue_body: Optional[str] = None
    ticket_title: Optional[str] = None
    ticket_body: Optional[str] = None
    email_response_content: Optional[str] = None

class ReactAgentState(MessagesState):
    remaining_steps: int
    structured_response: Any

class EmailAnalysisOutput(BaseModel):
    email_category: Literal["Website UX Issue", "Agentic Backend Issue", "General Inquiry"] = Field(
        description="The classified category of the email."
    )
    issue_keywords: List[str] = Field(
        description="Keywords extracted from the email for issue searching."
    )
    repository_name: Optional[Literal["PromptiusWeb", "AgentAgent"]] = Field(
        description="The determined repository for issue creation, if applicable. Only 'PromptiusWeb' or 'AgentAgent'."
    )
    issue_title: Optional[str] = Field(
        description="The suggested title for a new issue."
    )
    issue_body: Optional[str] = Field(
        description="The suggested body for a new issue."
    )
    ticket_title: Optional[str] = Field(
        description="The suggested title for a new support ticket."
    )
    ticket_body: Optional[str] = Field(
        description="The suggested body for a new support ticket."
    )

# Define a specific Pydantic model for an individual search result
class SearchResultItem(BaseModel):
    id: str = Field(description="The ID of the issue.")
    title: str = Field(description="The title of the issue.")
    url: str = Field(description="The URL of the issue.")
    # Add other relevant fields that your Search_Issues tool might return
    # For example:
    # state: str = Field(description="The state of the issue (e.g., 'open', 'closed').")
    # created_at: str = Field(description="Timestamp when the issue was created.")

class SearchIssuesOutput(BaseModel):
    search_results: List[SearchResultItem] = Field(
        description="Results from searching existing issues."
    )
    duplicate_issue_found: bool = Field(
        description="Indicates if a duplicate issue was found."
    )
    issue_id: Optional[str] = Field(
        description="The ID of the found duplicate issue, if any."
    )

class CreateIssueOutput(BaseModel):
    issue_id: str = Field(description="The ID of the newly created issue.")

class CreateTicketOutput(BaseModel):
    issue_id: str = Field(description="The ID of the newly created support ticket.")

class EmailResponseContent(BaseModel):
    email_response_content: str = Field(description="The content of the automated email response.")

def analyze_email_content(state: GraphState) -> GraphState:
    """
    Node purpose: Analyzes the incoming email to categorize it, extract relevant keywords, and determine the appropriate repository or action.
    Implementation reasoning: Uses an LLM with structured output to ensure consistent and type-safe extraction of email details for downstream processing.
    """
    structured_llm = llm.with_structured_output(EmailAnalysisOutput)
    
    prompt = f"""Analyze the following email content to:
    1. Categorize it into one of 'Website UX Issue', 'Agentic Backend Issue', or 'General Inquiry'.
    2. Extract relevant keywords for searching existing issues.
    3. If it's a 'Website UX Issue', suggest 'PromptiusWeb' as the repository.
    4. If it's an 'Agentic Backend Issue', suggest 'AgentAgent' as the repository.
    5. If it's an issue, suggest a concise title and a detailed body for a new issue.
    6. If it's a 'General Inquiry', suggest a concise title and a detailed body for a new support ticket.

    Email content:
    {state["email_content"]}
    """
    
    result: EmailAnalysisOutput = structured_llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "email_category": result.email_category,
        "issue_keywords": result.issue_keywords,
        "repository_name": result.repository_name,
        "issue_title": result.issue_title,
        "issue_body": result.issue_body,
        "ticket_title": result.ticket_title,
        "ticket_body": result.ticket_body,
        "messages": [AIMessage(content=f"Email analyzed. Category: {result.email_category}")]
    }

search_issues_tools = [Search_Issues]
def search_existing_issues(state: GraphState) -> GraphState:
    """
    Node purpose: Searches existing issues in the specified repositories (PromptiusWeb, AgentAgent) to avoid duplicates.
    Implementation reasoning: Uses a ReAct agent to interact with the Search_Issues tool, allowing for dynamic search queries based on extracted keywords.
    """
    class CustomSearchOutput(BaseModel):
        search_results: List[SearchResultItem] = Field(description="List of search results from the issue tracker.")
        duplicate_issue_found: bool = Field(description="True if a duplicate issue is found, False otherwise.")
        issue_id: Optional[str] = Field(description="The ID of the duplicate issue if found, otherwise None.")

    agent = create_react_agent(
        model=llm,
        prompt="You are an agent that searches for existing issues. Use the 'Search_Issues' tool to find issues in the specified repository based on keywords. Determine if a duplicate issue exists. If a duplicate is found, provide its ID.",
        tools=search_issues_tools,
        state_schema=ReactAgentState,
        response_format=CustomSearchOutput
    )

    # Assuming GitHub token is available as an environment variable
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        raise ValueError("GITHUB_TOKEN environment variable not set.")

    # The Search_Issues tool expects owner, repo, and token.
    # For simplicity, we'll assume a fixed owner for PromptiusWeb and AgentAgent.
    owner = "your_github_username" # Replace with actual GitHub owner

    # Construct a prompt for the agent to use the Search_Issues tool
    agent_prompt = f"Search for issues in the '{state['repository_name']}' repository related to keywords: {', '.join(state['issue_keywords'])}. The owner is '{owner}'. Use the provided GitHub token for authentication. After searching, determine if any of the results indicate a duplicate of the issue described by the keywords. If a duplicate is found, extract its ID."
    
    result: CustomSearchOutput = agent.invoke({
        "messages": [HumanMessage(content=agent_prompt)],
        "repository_name": state["repository_name"],
        "issue_keywords": state["issue_keywords"],
        "owner": owner,
        "token": github_token
    })["structured_response"]

    return {
        "search_results": [item.model_dump() for item in result.search_results], # Convert Pydantic models back to dicts for GraphState
        "duplicate_issue_found": result.duplicate_issue_found,
        "issue_id": result.issue_id,
        "messages": [AIMessage(content=f"Issue search completed. Duplicate found: {result.duplicate_issue_found}")]
    }

create_issue_tools = [Create_Issue]
def create_new_issue(state: GraphState) -> GraphState:
    """
    Node purpose: Creates a new issue in the appropriate repository based on the email content.
    Implementation reasoning: Uses a ReAct agent to interact with the Create_Issue tool, ensuring the issue is created with the correct details.
    """
    agent = create_react_agent(
        model=llm,
        prompt="You are an agent that creates new issues. Use the 'Create_Issue' tool to create an issue in the specified repository with the given title and body.",
        tools=create_issue_tools,
        state_schema=ReactAgentState,
        response_format=CreateIssueOutput
    )

    # Assuming GitHub token is available as an environment variable
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        raise ValueError("GITHUB_TOKEN environment variable not set.")

    # The Create_Issue tool expects owner, repo, title, body, and token.
    owner = "your_github_username" # Replace with actual GitHub owner

    agent_prompt = f"Create an issue in the '{state['repository_name']}' repository. The owner is '{owner}'. The title is '{state['issue_title']}' and the body is '{state['issue_body']}'. Use the provided GitHub token for authentication. Return the ID of the newly created issue."

    result: CreateIssueOutput = agent.invoke({
        "messages": [HumanMessage(content=agent_prompt)],
        "owner": owner,
        "repo": state["repository_name"],
        "title": state["issue_title"],
        "body": state["issue_body"],
        "token": github_token
    })["structured_response"]

    return {
        "issue_id": result.issue_id,
        "messages": [AIMessage(content=f"New issue created with ID: {result.issue_id}")]
    }

create_ticket_tools = [Create_Ticket]
def create_support_ticket(state: GraphState) -> GraphState:
    """
    Node purpose: Creates a support ticket for general customer complaints not related to website issues.
    Implementation reasoning: Uses a ReAct agent to interact with the Create_Ticket tool, ensuring the ticket is created with the correct details.
    """
    agent = create_react_agent(
        model=llm,
        prompt="You are an agent that creates support tickets. Use the 'Create_Ticket' tool to create a support ticket with the given title and body.",
        tools=create_ticket_tools,
        state_schema=ReactAgentState,
        response_format=CreateTicketOutput
    )

    # The Create_Ticket tool expects subject, description, and email.
    # Assuming a default requester email for the ticket.
    requester_email = "support@example.com" # Replace with actual support email

    agent_prompt = f"Create a support ticket with the subject '{state['ticket_title']}' and description '{state['ticket_body']}'. The requester email is '{requester_email}'."

    result: CreateTicketOutput = agent.invoke({
        "messages": [HumanMessage(content=agent_prompt)],
        "subject": state["ticket_title"],
        "description": state["ticket_body"],
        "email": requester_email
    })["structured_response"]

    return {
        "issue_id": result.issue_id,
        "messages": [AIMessage(content=f"New support ticket created with ID: {result.issue_id}")]
    }

def generate_automated_email_response(state: GraphState) -> GraphState:
    """
    Node purpose: Generates an automated email response to the user based on the outcome of the issue/ticket creation process.
    Implementation reasoning: Uses an LLM with structured output to generate a consistent and appropriate email response.
    """
    structured_llm = llm.with_structured_output(EmailResponseContent)
    
    if state["duplicate_issue_found"]:
        response_prompt = f"""Generate an email response to the user.
        A duplicate issue (ID: {state['issue_id']}) was found for their '{state['email_category']}' report.
        Inform them that we are aware of the issue and are working on it.
        """
    elif state["email_category"] in ["Website UX Issue", "Agentic Backend Issue"] and state["issue_id"]:
        response_prompt = f"""Generate an email response to the user.
        A new issue (ID: {state['issue_id']}) has been created for their '{state['email_category']}' report.
        Thank them for their report and inform them that we will investigate.
        """
    elif state["email_category"] == "General Inquiry" and state["issue_id"]:
        response_prompt = f"""Generate an email response to the user.
        A new support ticket (ID: {state['issue_id']}) has been created for their '{state['email_category']}'.
        Thank them for their inquiry and inform them that our support team will get back to them shortly.
        """
    else:
        response_prompt = f"""Generate a generic email response to the user for their '{state['email_category']}' inquiry.
        Apologize for any inconvenience and state that we have received their message and will get back to them.
        """

    result: EmailResponseContent = structured_llm.invoke([HumanMessage(content=response_prompt)])
    
    return {
        "email_response_content": result.email_response_content,
        "messages": [AIMessage(content="Automated email response generated.")]
    }

send_email_response_tools = [Send_Email]
def send_automated_email(state: GraphState) -> GraphState:
    """
    Node purpose: Sends the automated email response to the user.
    Implementation reasoning: Uses a ReAct agent to interact with the Send_Email tool to dispatch the generated email.
    """
    class SendEmailOutput(BaseModel):
        status: str = Field(description="Status of the email sending operation.")

    agent = create_react_agent(
        model=llm,
        prompt="You are an agent that sends emails. Use the 'Send_Email' tool to send an email with the given content. Assume the recipient is the original sender of the email.",
        tools=send_email_response_tools,
        state_schema=ReactAgentState,
        response_format=SendEmailOutput
    )

    # Assuming the original sender's email is available in the state or can be inferred.
    # For this example, we'll use a placeholder. In a real scenario, Receive_Email would provide this.
    recipient_email = "user@example.com" # Placeholder: In a real app, this would come from the initial email.
    subject = "Regarding your recent inquiry" # Placeholder: Could be generated by LLM or derived from category.

    agent_prompt = f"Send an email to '{recipient_email}' with the subject '{subject}' and the following content: '{state['email_response_content']}'."

    result: SendEmailOutput = agent.invoke({
        "messages": [HumanMessage(content=agent_prompt)],
        "to": recipient_email,
        "subject": subject,
        "body": state["email_response_content"]
    })["structured_response"]

    return {
        "messages": [AIMessage(content=f"Automated email response sent. Status: {result.status}")]
    }

workflow = StateGraph(GraphState)

workflow.add_node("analyze_email", analyze_email_content)
workflow.add_node("search_issues", search_existing_issues)
workflow.add_node("create_issue", create_new_issue)
workflow.add_node("create_ticket", create_support_ticket)
workflow.add_node("generate_email_response", generate_automated_email_response)
workflow.add_node("send_email_response", send_automated_email)

workflow.add_edge(START, "analyze_email")

def route_email_category(state: GraphState) -> str:
    """
    Routing function: Determines the next node based on the classified email category.
    Implementation reasoning: This function acts as a conditional router, directing the workflow
                              to either search for issues or create a support ticket based on the 'email_category' field in the state.
    """
    if state["email_category"] in ["Website UX Issue", "Agentic Backend Issue"]:
        return "search_issues"
    elif state["email_category"] == "General Inquiry":
        return "create_ticket"
    return "__END__" # Fallback for unhandled categories

workflow.add_conditional_edges(
    "analyze_email",
    route_email_category,
    {
        "search_issues": "search_issues",
        "create_ticket": "create_ticket",
        "__END__": END
    }
)

def route_duplicate_status(state: GraphState) -> str:
    """
    Routing function: Determines the next node based on whether a duplicate issue was found.
    Implementation reasoning: This function routes the workflow to either generate an email response
                              (if a duplicate is found) or proceed to create a new issue.
    """
    if state["duplicate_issue_found"] == True:
        return "generate_email_response"
    elif state["duplicate_issue_found"] == False:
        return "create_issue"
    return "__END__" # Fallback

workflow.add_conditional_edges(
    "search_issues",
    route_duplicate_status,
    {
        "generate_email_response": "generate_email_response",
        "create_issue": "create_issue",
        "__END__": END
    }
)

workflow.add_edge("create_issue", "generate_email_response")
workflow.add_edge("create_ticket", "generate_email_response")
workflow.add_edge("generate_email_response", "send_email_response")
workflow.add_edge("send_email_response", END)

checkpointer = InMemorySaver()
app = workflow.compile(
    checkpointer=checkpointer
)'''

def create_test_cases() -> Dict[str, Dict[str, Any]]:
    """Create comprehensive test cases covering various scenarios"""
    return {
        "perfect_code": {
            "description": "Well-structured LangGraph code following all requirements",
            "code": '''
from typing import Optional, List, Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.types import Command
import os

# LLM Definition
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Graph State
class GraphState(MessagesState):
    """Workflow state extending MessagesState"""
    query: str
    intent: Optional[str] = None
    response: Optional[str] = None
    status: str = "initialized"

# Pydantic Models
class IntentResult(BaseModel):
    """Intent classification result"""
    intent: Literal["question", "request"] = Field(description="User intent")
    confidence: float = Field(description="Confidence score", ge=0, le=1)

# Node Functions
def classify_intent(state: GraphState) -> GraphState:
    """
    Node purpose: Classify user intent from messages
    Implementation reasoning: Uses structured output for consistent intent detection
    """
    user_msg = state["messages"][-1].content
    structured_llm = llm.with_structured_output(IntentResult)
    
    result = structured_llm.invoke(f"Classify intent: {user_msg}")
    
    return {
        "messages": [AIMessage(content=f"Intent: {result.intent}")],
        "intent": result.intent,
        "status": "classified"
    }

def generate_response(state: GraphState) -> GraphState:
    """
    Node purpose: Generate final response
    Implementation reasoning: Uses Command for routing to END
    """
    query = state["messages"][-1].content
    intent = state["intent"]
    
    response = llm.invoke(f"Respond to {intent}: {query}").content
    
    return {
            "messages": [AIMessage(content=response)],
            "response": response,
            "status": "completed"
        }

# Graph Construction
workflow = StateGraph(GraphState)
workflow.add_node("classify", classify_intent)
workflow.add_node("respond", generate_response)
workflow.add_edge(START, "classify")
workflow.add_edge("classify", "respond")
workflow.add_edge("respond", END)

# Compilation
checkpointer = InMemorySaver()
app = workflow.compile(checkpointer=checkpointer)
            ''',
            "expected_errors": 0,
            "expected_key_accesses": 0,
            "expected_key_accesses_list":[],
            "expected_warnings": 0,
            "expected_errors_list": [],
            "expected_warnings_list": []
        },
        "state_schema_issues": {
            "description": "Code with GraphState definition problems",
            "code": '''
from langgraph.graph import MessagesState
from pydantic import BaseModel

# ERROR: Explicitly defines messages field
class BadGraphState(MessagesState):
    messages: str
    data: any  # ERROR: Vague type

# ERROR: Doesn't inherit from MessagesState
class AnotherBadState:
    query: str

# ERROR: Uses dict in Pydantic model
class BadModel(BaseModel):
    info: dict
    items: List[dict]  # ERROR: List[dict] usage
            ''',
            "expected_errors": 8,
            "expected_warnings": 0,
            "expected_key_accesses": 0,
            "expected_key_accesses_list": [],
            "expected_warnings_list": [],
            "expected_errors_list": [
                "State class 'BadGraphState' should not explicitly define 'messages' field when inheriting from MessagesState. Fix: Remove the explicit messages field definition.",
                "Pydantic model 'BadModel' field 'info' uses a 'dict' type in annotation 'dict'. Fix: Use a specific Pydantic model for structured data, make changes in the code referencing this field to accomodate the change in type",
                "Pydantic model 'BadModel' field 'items' uses a 'dict' type in annotation 'List[dict]'. Fix: Use a specific Pydantic model for structured data, make changes in the code referencing this field to accomodate the change in type" ,                
                "Missing required imports: ['AIMessage', 'ChatOpenAI', 'Field', 'HumanMessage', 'StateGraph', 'SystemMessage']. Fix: Add appropriate import statements for these components.",
                'No LLM definition found. Fix: Add \'llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)\'',
                'No edge definitions detected', 
                "No graph compilation found. Fix: Add 'app = workflow.compile(checkpointer=checkpointer)'", 
                'No checkpointer definition found. Consider adding: checkpointer = InMemorySaver()']
        },
        "convoluted_pydantic_model": {
            "description": "Code with GraphState definition problems",
            "code": '''
from pydantic import BaseModel

class RequestClassification(BaseModel):
    """Structured output for classifying user requests."""
    request_type: Literal["add_investment", "portfolio_value", "set_alert", "unknown"] = Field(description="The classified type of request from the user.")
    
    investment_details: Optional[Dict[str, Any]] = Field(default=None, description="Details of the investment to be added, including stock symbol, shares, and purchase price. Required if request_type is 'add_investment'.")
    
    alert_details: Optional[Dict[str, Any]] = Field(default=None, description="Details of the price alert to be set, including stock symbol and threshold price. Required if request_type is 'set_alert'.")
            ''',
            "expected_errors": 8,
            "expected_warnings": 0,
            "expected_key_accesses": 0,
            "expected_key_accesses_list": [],
            "expected_warnings_list": [],
            "expected_errors_list": [
                "Pydantic model 'RequestClassification' field 'investment_details' uses a 'dict' type in annotation 'Optional[Dict[str, Any]]'. Fix: Use a specific Pydantic model for structured data, make changes in the code referencing this field to accomodate the change in type",
                "Pydantic model 'RequestClassification' field 'alert_details' uses a 'dict' type in annotation 'Optional[Dict[str, Any]]'. Fix: Use a specific Pydantic model for structured data, make changes in the code referencing this field to accomodate the change in type",
                "Missing required imports: ['AIMessage', 'ChatOpenAI', 'Field', 'HumanMessage', 'MessagesState', 'StateGraph', 'SystemMessage']. Fix: Add appropriate import statements for these components.",
                'No GraphState class found. Fix: Define a class that inherits from MessagesState.', 'No LLM definition found. Fix: Add \'llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)\'',
                'No edge definitions detected', "No graph compilation found. Fix: Add 'app = workflow.compile(checkpointer=checkpointer)'",
                'No checkpointer definition found. Consider adding: checkpointer = InMemorySaver()']
        },
        "node_function_issues": {
            "description": "Code with node function implementation problems",
            "code": '''
from langgraph.graph import MessagesState
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GraphState(MessagesState):
    result: str

# ERROR: Missing docstring, wrong state access, no messages in return
def bad_node(state):
    data = state.result  # ERROR: Should use state["result"]
    return {"result": "done"}  # ERROR: Missing messages key

# ERROR: No state parameter
def another_bad_node():
    return {"messages": [], "result": "test"}

# GOOD: Proper node function
def good_node(state: GraphState) -> GraphState:
    """
    Node purpose: Process data correctly
    Implementation reasoning: Follows all required patterns
    """
    result = state["result"]  # Correct dict access
    return {
        "messages": [AIMessage(content="Processed")],  # Required messages key
        "result": f"Processed: {result}"
    }
            ''',
            "expected_errors": 4,
            "expected_warnings": 0,
            "expected_key_accesses_list": [],
            "expected_key_accesses": 0,
            "expected_warnings_list": [],
            "expected_errors_list": [
                "Missing required imports: ['AIMessage', 'BaseModel', 'Field', 'HumanMessage', 'StateGraph', 'SystemMessage']. Fix: Add appropriate import statements for these components.",
                'No edge definitions detected',
                "No graph compilation found. Fix: Add 'app = workflow.compile(checkpointer=checkpointer)'", 
                'No checkpointer definition found. Consider adding: checkpointer = InMemorySaver()']         },
        "react_agent_issues": {
            "description": "Code with react agent implementation problems",
            "code": '''
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import List

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class SearchResult(BaseModel):
    results: List[str]

def bad_agent_node(state):
    # ERROR: Missing required parameters
    agent = create_react_agent(model=llm)
    
    # ERROR: Wrong response_format usage
    agent2 = create_react_agent(
        model=llm,
        tools=[],
        state_schema=state,
        response_format=List[SearchResult]  # ERROR: Should not use List[Model]
    )
    
    return {"messages": []}
            ''',
            "expected_errors": 9,
            "expected_warnings": 0,
            "expected_key_accesses_list": [],
            "expected_key_accesses": 0,
            "expected_warnings_list": [],
            "expected_errors_list": [
                "create_react_agent missing required parameters: ['response_format', 'state_schema', 'tools']. Fix: Add the missing parameters to your create_react_agent call.",
                'create_react_agent state_schema must be ReactAgentState and not state. \n \n<STATE_SCHEMA>\nclass ReactAgentState(MessagesState):\n    remaining_steps: int\n    structured_response: any\n</STATE_SCHEMA>\n\n<CALLING_PATTERN>\n\nFollow this example:\n```python\n\nnode_name_tools = [list_of_tools]\ndef node_name(state: GraphState) -> GraphState:\n    # define what kind of result you need from the agent.\n    class CustomClass(BaseModel):\n        attr1: type = Field(description="What is the field")\n        attr2: type = Field(description="What is the field")\n\n    agent = create_react_agent(\n      model=llm,\n      prompt="The prompt for the agent to follow, also mention which tools to use, if any.",\n      tools=node_name_tools,\n      state_schema=ReactAgentState\n      response_format=CustomClass)\n\n    result: CustomClass = agent.invoke({"messages":state["messages"]})["structured_response"] #or whatever content you wish to put as per the state.\n    # Logic that either updates the state variable with result.attr1/result.attr2\n    # DO NOT do string parsing or regex parsing\n```\n\nPlease ensure that the code produced for a tool node follows:\n1. **Tool Registration**: Tools are properly defined and registered in the node\n2. **Schema Adherence**: Tool inputs/outputs match their defined schemas exactly\n\n<NOTE>\nMake sure to use proper pydantic models for structured output.\nDONOT use dict!!!!\nthe response_format should be a pydantic model with the required fields. Should not be a List[PydanticModel] or something like that.\n</NOTE>\n\n</CALLING_PATTERN>\n         ',
                'create_react_agent state_schema \'state\' is not a defined class. \n<STATE_SCHEMA>\nclass ReactAgentState(MessagesState):\n    remaining_steps: int\n    structured_response: any\n</STATE_SCHEMA>\n\n<CALLING_PATTERN>\n\nFollow this example:\n```python\n\nnode_name_tools = [list_of_tools]\ndef node_name(state: GraphState) -> GraphState:\n    # define what kind of result you need from the agent.\n    class CustomClass(BaseModel):\n        attr1: type = Field(description="What is the field")\n        attr2: type = Field(description="What is the field")\n\n    agent = create_react_agent(\n      model=llm,\n      prompt="The prompt for the agent to follow, also mention which tools to use, if any.",\n      tools=node_name_tools,\n      state_schema=ReactAgentState\n      response_format=CustomClass)\n\n    result: CustomClass = agent.invoke({"messages":state["messages"]})["structured_response"] #or whatever content you wish to put as per the state.\n    # Logic that either updates the state variable with result.attr1/result.attr2\n    # DO NOT do string parsing or regex parsing\n```\n\nPlease ensure that the code produced for a tool node follows:\n1. **Tool Registration**: Tools are properly defined and registered in the node\n2. **Schema Adherence**: Tool inputs/outputs match their defined schemas exactly\n\n<NOTE>\nMake sure to use proper pydantic models for structured output.\nDONOT use dict!!!!\nthe response_format should be a pydantic model with the required fields. Should not be a List[PydanticModel] or something like that.\n</NOTE>\n\n</CALLING_PATTERN>\n        ',
                'create_react_agent response_format should not be List[Model]. Fix: Use a single Pydantic model instead of a list.',
                "Missing required imports: ['AIMessage', 'Field', 'HumanMessage', 'MessagesState', 'StateGraph', 'SystemMessage']. Fix: Add appropriate import statements for these components.", 
                'No GraphState class found. Fix: Define a class that inherits from MessagesState.',
                'No edge definitions detected', "No graph compilation found. Fix: Add 'app = workflow.compile(checkpointer=checkpointer)'", 
                'No checkpointer definition found. Consider adding: checkpointer = InMemorySaver()']
        },
        "missing_components": {
            "description": "Code missing essential LangGraph components",
            "code": '''
# Missing most required components
from typing import Dict

def some_function():
    return {"result": "test"}

class SomeClass:
    pass
            ''',
            "expected_errors": 6,
            "expected_warnings": 0,
            "expected_key_accesses": 0,
            "expected_warnings_list": [],
            "expected_key_accesses_list": [],
            "expected_errors_list": [
                "Missing required imports: ['AIMessage', 'BaseModel', 'ChatOpenAI', 'Field', 'HumanMessage', 'MessagesState', 'StateGraph', 'SystemMessage']. Fix: Add appropriate import statements for these components.",
                'No GraphState class found. Fix: Define a class that inherits from MessagesState.',
                'No LLM definition found. Fix: Add \'llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)\'',
                'No edge definitions detected',
                "No graph compilation found. Fix: Add 'app = workflow.compile(checkpointer=checkpointer)'", 
                'No checkpointer definition found. Consider adding: checkpointer = InMemorySaver()']
        },
        "edge_and_compilation_issues": {
            "description": "Code with edge definition and compilation problems",
            "code": '''
from langgraph.graph import StateGraph, MessagesState
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GraphState(MessagesState):
    status: str

def node1(state: GraphState) -> GraphState:
    """Node 1"""
    return {"messages": [], "status": "done"}

workflow = StateGraph(GraphState)
workflow.add_node("node1", node1)

# ERROR: Wrong number of arguments for add_edge
workflow.add_edge("node1")

# ERROR: Missing conditional edge parameters
workflow.add_conditional_edges("node1")

# Missing compilation - this will be caught by overall validation
            ''',
            "expected_errors": 5,
            "expected_warnings": 0,
            "expected_key_accesses": 0,
            "expected_key_accesses_list": [],
            "expected_errors_list": [
                "add_edge requires exactly 2 arguments (source, target). Fix: graph.add_edge('source_node', 'target_node')",
                "add_conditional_edges requires at least 2 arguments (source, routing_function). Fix: graph.add_conditional_edges('node', routing_function) or graph.add_conditional_edges('node', routing_function, mapping_dict)",
                "Missing required imports: ['AIMessage', 'BaseModel', 'Field', 'HumanMessage', 'SystemMessage']. Fix: Add appropriate import statements for these components.",
                "No graph compilation found. Fix: Add 'app = workflow.compile(checkpointer=checkpointer)'",
                'No checkpointer definition found. Consider adding: checkpointer = InMemorySaver()'],
            "expected_warnings_list": []
        },
        "state_flow_sequential_valid": {
            "description": "Valid sequential flow of state",
            "code": '''
from langgraph.graph import MessagesState, START, END, StateGraph
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

llm = ChatOpenAI()
checkpointer = InMemorySaver()

class MyState(MessagesState):
    a: str
    b: str

def node_a(state):
    """docstring"""
    return {"a": "a", "messages": [AIMessage(content="")]}

def node_b(state):
    """docstring"""
    x = state["a"]
    return {"b": "b", "messages": [AIMessage(content="")]}

def node_c(state):
    """docstring"""
    y = state["b"]
    return {"messages": [AIMessage(content="")]}

workflow = StateGraph(MyState)
workflow.add_node("node_a", node_a)
workflow.add_node("node_b", node_b)
workflow.add_node("node_c", node_c)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", "node_c")
app = workflow.compile(checkpointer=checkpointer)
''',
            "expected_errors": 0,
            "expected_warnings": 0,
            "expected_key_accesses": 0,
            "expected_errors_list": [],
            "expected_warnings_list": [],
            "expected_key_accesses_list": []
        },
        "state_flow_sequential_invalid": {
            "description": "Invalid sequential flow of state",
            "code": '''
from langgraph.graph import MessagesState, START, END, StateGraph
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

llm = ChatOpenAI()
checkpointer = InMemorySaver()

class MyState(MessagesState):
    a: str
    b: str

def node_a(state):
    """docstring"""
    return {"messages": [AIMessage(content="")]}

def node_b(state):
    """docstring"""
    x = state["a"]
    return {"b": "b", "messages": [AIMessage(content="")]}

workflow = StateGraph(MyState)
workflow.add_node("node_a", node_a)
workflow.add_node("node_b", node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
app = workflow.compile(checkpointer=checkpointer)
''',
            "expected_errors": 0,
            "expected_warnings": 0,
            "expected_key_accesses": 1,
            "expected_key_accesses_list": ["In node 'node_b', state variable 'a' is accessed but might not be assigned. There is a path from START to 'node_b' that does not guarantee 'a' is assigned: START -> node_a -> node_b"],
            "expected_errors_list": [],
            "expected_warnings_list": []
        },
        "state_flow_conditional_invalid": {
            "description": "Invalid conditional flow of state",
            "code": '''
from langgraph.graph import MessagesState, START, END, StateGraph
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

llm = ChatOpenAI()
checkpointer = InMemorySaver()

class MyState(MessagesState):
    a: str
    b: str
    c: str

def node_a(state):
    """docstring"""
    return {"a": "a", "messages": [AIMessage(content="")]}

def router(state):
    """docstring"""
    return "node_b"

def node_b(state):
    """docstring"""
    return {"b": "b", "messages": [AIMessage(content="")]}

def node_c(state):
    """docstring"""
    return {"messages": [AIMessage(content="")]}

def node_d(state):
    """docstring"""
    x = state["a"]
    y = state["b"]
    return {"c": "d", "messages": [AIMessage(content="")]}

workflow = StateGraph(MyState)
workflow.add_node("node_a", node_a)
workflow.add_node("node_b", node_b)
workflow.add_node("node_c", node_c)
workflow.add_node("node_d", node_d)
workflow.add_edge(START, "node_a")
workflow.add_conditional_edges(
    "node_a",
    router,
    {"node_b": "node_b", "node_c": "node_c"}
)
workflow.add_edge("node_b", "node_d")
workflow.add_edge("node_c", "node_d")
app = workflow.compile(checkpointer=checkpointer)
''',
            "expected_errors": 0,
            "expected_key_accesses": 1,
            "expected_warnings": 0,
            "expected_errors_list":[],
            "expected_key_accesses_list": ["In node 'node_d', state variable 'b' is accessed but might not be assigned. There is a path from START to 'node_d' that does not guarantee 'b' is assigned: START -> node_a -> node_c -> node_d"],
            "expected_warnings_list": []
        },
        "state_flow_loop_valid": {
            "description": "Valid state flow in a loop",
            "code": '''
from langgraph.graph import MessagesState, START, END, StateGraph
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

llm = ChatOpenAI()
checkpointer = InMemorySaver()

class MyState(MessagesState):
    counter: int = 0

def node_a(state):
    """docstring"""
    count = state["counter"]
    return {"counter": count + 1, "messages": [AIMessage(content="")]}

def should_continue(state):
    """docstring"""
    if state["counter"] > 2:
        return "end"
    return "continue"

workflow = StateGraph(MyState)
workflow.add_node("node_a", node_a)
workflow.add_edge(START, "node_a")
workflow.add_conditional_edges(
    "node_a",
    should_continue,
    {"continue": "node_a", "end": END}
)
app = workflow.compile(checkpointer=checkpointer)
''',
            "expected_errors": 0,
            "expected_warnings": 0,
            "expected_key_accesses": 1,
            "expected_errors_list": [],
            "expected_warnings_list": [],
            "expected_key_accesses_list":["In node 'node_a', state variable 'counter' is accessed but might not be assigned. There is a path from START to 'node_a' that does not guarantee 'counter' is assigned: START -> node_a"]
        },
        "state_flow_parallel_invalid": {
            "description": "Invalid parallel flow with cross-dependency",
            "code": '''
from langgraph.graph import MessagesState, START, END, StateGraph
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

llm = ChatOpenAI()
checkpointer = InMemorySaver()

class MyState(MessagesState):
    a: str
    b: str

def node_a(state):
    """docstring"""
    x = state["b"]
    return {"a": "a", "messages": [AIMessage(content="")]}

def node_b(state):
    """docstring"""
    return {"b": "b", "messages": [AIMessage(content="")]}

workflow = StateGraph(MyState)
workflow.add_node("node_a", node_a)
workflow.add_node("node_b", node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge(START, "node_b")
app = workflow.compile(checkpointer=checkpointer)
''',
            "expected_errors": 0,
            "expected_key_accesses":1,
            "expected_warnings": 0,
            "expected_errors_list": [],
            "expected_key_accesses_list": ["In node 'node_a', state variable 'b' is accessed but might not be assigned. There is a path from START to 'node_a' that does not guarantee 'b' is assigned: START -> node_a"],
            "expected_warnings_list": []
        },
        "try_new": {
            "description": "Try new",
            "code": github_code,
            "expected_errors": 0,
            "expected_key_accesses":2,
            "expected_warnings": 0,
            "expected_key_accesses_list": [
                "In node 'analyze_email_content', state variable 'email_content' is accessed but might not be assigned. There is a path from START to 'analyze_email_content' that does not guarantee 'email_content' is assigned: START -> analyze_email_content",
                "In node 'generate_automated_email_response', state variable 'duplicate_issue_found' is accessed but might not be assigned. There is a path from START to 'generate_automated_email_response' that does not guarantee 'duplicate_issue_found' is assigned: START -> analyze_email_content -> create_support_ticket -> generate_automated_email_response"],
            "expected_errors_list": [],
            "expected_warnings_list": []
        }
    }

test_cases = create_test_cases()

@pytest.mark.parametrize("test_name, test_data", test_cases.items())
@pytest.mark.asyncio
def test_langgraph_validation_cases(test_name, test_data):
    """
    Runs validation for a specific test case and asserts the number of errors.
    """
    description = test_data['description']
    code = test_data['code']
    expected_errors = test_data['expected_errors']
    expected_warnings = test_data['expected_warnings']
    expected_key_accesses = test_data['expected_key_accesses']

    print(f"\n--- Running Test: {test_name} ---")
    print(f"Description: {description}")
    
    report = run_detailed_validation(code)
    actual_errors = len(report.get('errors', []))
    
    assert actual_errors == expected_errors, (
        f"Test '{test_name}' FAILED: "
        f"Expected {expected_errors} errors, but found {actual_errors}.\n"
        f"Errors found: {report.get('errors', [])}"
    )

    assert report.get('errors',[]) == test_data['expected_errors_list'],(
        f"Test '{test_name}' FAILED: "
        f"Expected {test_data['expected_errors_list']} errors, but found {report.get('errors', [])}.\n"
    )


    actual_warnings = len(report.get('warnings', []))
    
    assert actual_warnings == expected_warnings, (
        f"Test '{test_name}' FAILED: "
        f"Expected {expected_warnings} warnings, but found {actual_warnings}.\n"
        f"Warnings found: {report.get('warnings', [])}"
    )

    assert report.get('warnings',[]) == test_data['expected_warnings_list'],(
        f"Test '{test_name}' FAILED: "
        f"Expected {test_data['expected_warnings_list']} errors, but found {report.get('warnings', [])}.\n"
    )

    actual_key_accesses = len(report.get('key_accesses', []))
    
    assert actual_key_accesses == expected_key_accesses, (
        f"Test '{test_name}' FAILED: "
        f"Expected {expected_key_accesses} key_accesses, but found {actual_key_accesses}.\n"
        f"key_accesses found: {report.get('key_accesses', [])}"
    )

    assert sorted(report.get('key_accesses',[])) == test_data['expected_key_accesses_list'],(
        f"Test '{test_name}' FAILED: "
        f"Expected {test_data['expected_key_accesses_list']} key_accesses, but found {report.get('key_accesses', [])}.\n"
    )
