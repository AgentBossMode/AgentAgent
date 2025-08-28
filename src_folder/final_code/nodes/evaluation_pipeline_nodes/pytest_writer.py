from typing import List
from final_code.states.DryRunState import UseCaseAnalysis
from final_code.nodes.tools.pytest_writing_tools import write_trajectory_pytest_code, TRAJECTORY_STR
from final_code.llms.model_factory import get_model, ModelName
from langchain_core.messages import HumanMessage, AIMessage
from final_code.pydantic_models.UtGen import UtGeneration
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_customize_config
from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.states.ReqAnalysis import DryRuns
from final_code.utils.copilotkit_emit_status import append_in_progress_to_list, update_last_status
from final_code.ast_visitors_lib.validation_script import run_detailed_validation
from final_code.utils.get_filtered_file import get_filtered_file


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
    b. If user has also passed a message which looks like: "In node 'node_name', state variable 'x' is accessed but might not be assigned. There is a path from START to 'node_name' that does not guarantee 'x' is assigned: START -> A -> B -> node_name",
        look at the graph, verify if this is a necessary piece of additional information, and then add it to the final dict.

3. The list size of the TrajectoryUts should be equal to the number of use cases.

The input_query dict should contain atleast the 'messages' key. Make sure 
{{
    "messages": [{{ "role": "user", "content": input_content }}]
}}

<TRAJECTORY>
{TRAJECTORY_STR}
</TRAJECTORY>
"""

async def pytest_writer(state: AgentBuilderState, config: RunnableConfig):
    modified_config = copilotkit_customize_config(config, emit_messages=False)
    await append_in_progress_to_list(config, state, "Generating pytest code...")
    python_code = state["python_code"]
    mock_tools_code = state["mock_tools_code"]

    python_file = get_filtered_file(state["python_code"])
    validation_report = run_detailed_validation(python_file)
    utgenerated: UtGeneration = await generate_ut_llm_call(state["dry_runs"], python_code, mock_tools_code, validation_report["key_accesses"])

    inputs_trajectory = []
    responses_trajectory = []
    for ut in utgenerated.trajectory_uts:
        inputs_trajectory.append(ut.input_dict)
        responses_trajectory.append(ut.expected_trajectory)

    final_trajectory_code = write_trajectory_pytest_code(inputs_trajectory, responses_trajectory)

    PYTEST = """
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

{final_trajectory_code}
"""
    await update_last_status(modified_config, state, "Pytest code generated", True)
    return {
        "agent_status_list": state["agent_status_list"],
        "utGeneration": utgenerated,
        "current_tab":"console",
        "pytest_code": PYTEST.format(final_trajectory_code=final_trajectory_code)
        }

async def generate_ut_llm_call(dry_runs: DryRuns, python_code: str, mock_tools_code: str, key_accesses: list[str]):

    pytest_llm = get_model(ModelName.GEMINI25FLASH).with_structured_output(UtGeneration)
    msg_list = [HumanMessage(content=PYTEST_WRITER_PROMPT
                      .format(
                          python_code=python_code,
                            use_cases=dry_runs.model_dump_json(indent=2),
                              mock_tools_code=mock_tools_code,
                                TRAJECTORY_STR=TRAJECTORY_STR))]    
    if len(key_accesses)>0:
        msg_list.append(HumanMessage(content=f"<KEY_ACCESSES>{str(key_accesses)}</KEY_ACCESSES>"))

    utgenerated: UtGeneration = await pytest_llm.ainvoke(msg_list)
    return utgenerated

