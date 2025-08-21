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

def generate_response(state: GraphState) -> Command[Literal["END"]]:
    """
    Node purpose: Generate final response
    Implementation reasoning: Uses Command for routing to END
    """
    query = state["messages"][-1].content
    intent = state["intent"]
    
    response = llm.invoke(f"Respond to {intent}: {query}").content
    
    return Command(
        update={
            "messages": [AIMessage(content=response)],
            "response": response,
            "status": "completed"
        },
        goto="END"
    )

# Graph Construction
workflow = StateGraph(GraphState)
workflow.add_node("classify", classify_intent)
workflow.add_node("respond", generate_response)
workflow.add_edge(START, "classify")
workflow.add_edge("classify", "respond")

# Compilation
checkpointer = InMemorySaver()
app = workflow.compile(checkpointer=checkpointer)
            ''',
            "expected_errors": 0,
            "expected_warnings": 0
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
            "expected_errors": 2,
            "expected_warnings": 0
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
            "expected_errors": 2,
            "expected_warnings": 2
        },
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
            "expected_errors": 3,
            "expected_warnings": 0
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
            "expected_errors": 4,
            "expected_warnings": 3
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
            "expected_errors": 3,
            "expected_warnings": 1
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
    
    print(f"\n--- Running Test: {test_name} ---")
    print(f"Description: {description}")
    
    report = run_detailed_validation(code)
    actual_errors = len(report.get('errors', []))
    
    assert actual_errors >= expected_errors, (
        f"Test '{test_name}' FAILED: "
        f"Expected at least {expected_errors} errors, but found {actual_errors}.\n"
        f"Errors found: {report.get('errors', [])}"
    )
    
    print(f"Test '{test_name}' PASSED: Found {actual_errors} errors (expected >= {expected_errors}).")

