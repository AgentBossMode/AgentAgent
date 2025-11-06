import pytest
from dotenv import load_dotenv
load_dotenv()
import ast
from src_folder.tests.test_utils.stock_agent.stock_mock_tools  import stock_mock_tools
from src_folder.tests.test_utils.stock_agent.stock_main import stock_main
from src_folder.tests.test_utils.stock_agent.stock_dry_runs import stock_dry_runs

from src_folder.final_code.pydantic_models.UtGen import UtGeneration, TrajectoryUt
from src_folder.final_code.nodes.evaluation_pipeline_nodes.pytest_writer import pytest_writer
from src_folder.final_code.states.ReqAnalysis import DryRuns
from typing import List
from langgraph.types import Command

@pytest.mark.asyncio
async def test_pytest_writer():
    # Mock state with necessary fields
    updated_state: Command = await pytest_writer({
        "dry_runs":DryRuns.model_validate_json(stock_dry_runs),
        "python_code": stock_main,
        "mock_tools_code": stock_mock_tools
        },
        {"type": "test"})
    utgenerated: UtGeneration = updated_state.update["utGeneration"]

    ut_trajectory : List[TrajectoryUt] = utgenerated.trajectory_uts
    assert len(ut_trajectory) > 1, "No trajectory UTs generated"
    assert ast.parse(updated_state.update["pytest_code"])
