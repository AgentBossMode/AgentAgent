from typing import List
from final_code.states.DryRunState import UseCaseAnalysis
from final_code.nodes.tools.pytest_writing_tools import write_trajectory_pytest_code, TRAJECTORY_STR
from final_code.llms.model_factory import get_model, ModelName
from langchain_core.messages import HumanMessage, AIMessage
from final_code.pydantic_models.UtGen import UtGeneration
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_customize_config, copilotkit_emit_state
from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.states.ReqAnalysis import DryRuns
PYTEST_WRITER_PROMPT = """
You are a python code writing expert, your job is to write test case inputs given the langgraph code and use cases.
<python_code.py>
{python_code}
</python_code.py>

<tools_code.py>
{mock_tools_code}
</tools_code.py>

You are given the use cases for a workflow graph along with dry runs.
<USE_CASES>
{use_cases}
</USE_CASES>
2. you will now write pytest code, use the 'USE_CASES' to generate test cases for the code in 'python_code.py' section.The tests should cover the following:
    a. Trajectory: refer to <TRAJECTORY> section
3. The list size of the TrajectoryUts should be equal to the number of use cases.

<TRAJECTORY>
{TRAJECTORY_STR}
</TRAJECTORY>
"""

async def pytest_writer(state: AgentBuilderState, config: RunnableConfig):
    modified_config = copilotkit_customize_config(config, emit_messages=False)
    state["current_status"] = {"inProcess":True ,"status": "Generating pytest code.."}
    await copilotkit_emit_state(state=state, config=modified_config)
    python_code = state["python_code"]
    mock_tools_code = state["mock_tools_code"]
    utgenerated: UtGeneration = await generate_ut_llm_call(state["dry_runs"], python_code, mock_tools_code)

    inputs_trajectory = []
    responses_trajectory = []
    for ut in utgenerated.trajectory_uts:
        inputs_trajectory.append(ut.input)
        responses_trajectory.append(ut.expected_trajectory)

    final_trajectory_code = write_trajectory_pytest_code(inputs_trajectory, responses_trajectory)

    PYTEST = """
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

{final_trajectory_code}
"""
    state["current_status"] = {"inProcess":False ,"status": "Pytest code generated"}
    await copilotkit_emit_state(state=state, config=modified_config)
    return {
        "utGeneration": utgenerated,
        "current_tab":"console",
            "pytest_code": PYTEST.format(final_trajectory_code=final_trajectory_code),
            "messages": [AIMessage(content="Unit tests have been generated")]}

async def generate_ut_llm_call(dry_runs: DryRuns, python_code, mock_tools_code):
    pytest_llm = get_model(ModelName.GEMINI25FLASH).with_structured_output(UtGeneration)
    utgenerated: UtGeneration = await pytest_llm.ainvoke([HumanMessage(content=PYTEST_WRITER_PROMPT.format(python_code=python_code, use_cases=dry_runs.model_dump_json(indent=2), mock_tools_code=mock_tools_code, TRAJECTORY_STR=TRAJECTORY_STR))])
    return utgenerated
