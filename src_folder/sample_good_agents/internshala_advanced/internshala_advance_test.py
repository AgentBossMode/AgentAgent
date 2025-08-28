
import json
import pytest
from uuid import uuid4
from app import app
from typing import Any
import collections
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from agentevals.graph_trajectory.llm import create_graph_trajectory_llm_as_judge
from agentevals.graph_trajectory.utils import (
    extract_langgraph_trajectory_from_thread,
)


DEFAULT_REF_COMPARE_PROMPT = """You are an expert data labeler.
Your task is to grade the accuracy of an AI agent's internal steps in resolving a user queries.

<Rubric>
  An accurate trajectory:
  - Makes logical sense between steps
  - Shows clear progression
  - Is relatively efficient, though it does not need to be perfectly efficient
  - Is semantically equivalent to the provided reference trajectory, if present
</Rubric>

<Instructions>
  Grade the following thread, evaluating whether the agent's overall steps are logical and relatively efficient.
  Check the list of "messages" passed, read each of them and see if the progression of ai messages is logical.
  For the trajectory, "__start__" denotes an initial entrypoint to the agent, and "__interrupt__" corresponds to the agent
  interrupting to await additional data from another source ("human-in-the-loop"):
</Instructions>

<thread>
{thread}
</thread>

{reference_outputs}
"""


def serialize_value(value: Any) -> Any:
    """
    Recursively serialize any value that may contain BaseModel instances.
    Handles nested structures, collections, and Optional types.
    """
    # Handle None
    if value is None:
        return None
    
    # Handle BaseModel instances
    if isinstance(value, BaseModel):
        return value.model_dump()
    
    # Handle dictionaries (including nested ones)
    if isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    
    # Handle all iterable collections (list, tuple, set, frozenset, etc.)
    # but exclude strings and bytes which are also iterable
    if isinstance(value, collections.abc.Iterable) and not isinstance(value, (str, bytes)):
        # Convert to list for JSON serialization (sets become lists)
        if isinstance(value, (set, frozenset)):
            return [serialize_value(item) for item in value]
        # Preserve tuple structure by converting back to tuple
        elif isinstance(value, tuple):
            return tuple(serialize_value(item) for item in value)
        # Handle other iterables (list, deque, etc.)
        else:
            return [serialize_value(item) for item in value]
    
    # Return primitive types as-is
    return value

@pytest.mark.parametrize(
    "input_query, expected_tool_call_names",
    [
     (json.dumps({"messages": [{"role": "user", "content": "Please process internship listings from the following website: https://example-internships.com/listings"}], "initial_url": "https://example-internships.com/listings"}), ['start', 'parse_initial_url', 'Web_Scraper_Parse', 'extract_internship_listing_urls', 'read_existing_data', 'Google_Sheets_Read', 'filter_new_listings', 'extract_job_details', 'Web_Scraper_Extract', 'parse_company_url', 'URL_Parser_Extract_Base_URL', 'scrape_company_website', 'Web_Scraper_Parse', 'extract_email_address', 'Email_Extractor_Extract', 'update_google_sheet', 'Google_Sheets_Write', 'end'])
    ]
)
def test_full_workflow_trajectory(input_query: str, expected_tool_call_names: list[str]):   
    thread_config = {"configurable": {"thread_id": uuid4()}}
    result = app.invoke(json.loads(input_query), config=thread_config)
    
    print(result)
    # Extract the trajectory from the first two thread runs
    extracted_trajectory = extract_langgraph_trajectory_from_thread(app, thread_config)
    
     
    messages_as_dicts = [msg.model_dump(include={"content", "type"}) for msg in result["messages"]]
    result["messages"] = messages_as_dicts
    new_result = result

    for key,value in result.items():
        if key == "messages":
            continue
        new_result[key] = serialize_value(value)
   
    extracted_trajectory["outputs"]["results"]= [new_result]
    graph_trajectory_evaluator = create_graph_trajectory_llm_as_judge(prompt=DEFAULT_REF_COMPARE_PROMPT, model="openai:gpt-4o-mini")
    print(extracted_trajectory)
    res = graph_trajectory_evaluator(inputs=extracted_trajectory["inputs"], outputs=extracted_trajectory["outputs"])
    assert res["score"] == True, res["comment"]

