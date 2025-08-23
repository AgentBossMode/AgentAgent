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
            "expected_errors": 9,
            "expected_warnings": 0,
            "expected_warnings_list": [],
            "expected_errors_list": [
                "State class 'BadGraphState' should not explicitly define 'messages' field when inheriting from MessagesState. Fix: Remove the explicit messages field definition.",
                "State class 'AnotherBadState' must inherit from MessagesState",
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
            "expected_warnings_list": [],
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
            "expected_errors_list": [
                "add_edge requires exactly 2 arguments (source, target). Fix: graph.add_edge('source_node', 'target_node')",
                "add_conditional_edges requires at least 2 arguments (source, routing_function). Fix: graph.add_conditional_edges('node', routing_function) or graph.add_conditional_edges('node', routing_function, mapping_dict)",
                "Missing required imports: ['AIMessage', 'BaseModel', 'Field', 'HumanMessage', 'SystemMessage']. Fix: Add appropriate import statements for these components.",
                "No graph compilation found. Fix: Add 'app = workflow.compile(checkpointer=checkpointer)'",
                'No checkpointer definition found. Consider adding: checkpointer = InMemorySaver()'],
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
