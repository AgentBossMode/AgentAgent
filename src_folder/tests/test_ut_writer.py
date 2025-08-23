import pytest
from dotenv import load_dotenv
load_dotenv()

from src_folder.tests.test_utils.stock_agent.stock_mock_tools  import stock_mock_tools
from src_folder.tests.test_utils.stock_agent.stock_main import stock_main
from src_folder.tests.test_utils.stock_agent.stock_dry_runs import stock_dry_runs

from src_folder.final_code.pydantic_models.UtGen import UtGeneration, TrajectoryUt
from src_folder.final_code.nodes.evaluation_pipeline_nodes.pytest_writer import generate_ut_llm_call
from src_folder.final_code.states.ReqAnalysis import DryRuns
from typing import List

@pytest.mark.asyncio
async def test_pytest_writer():
    # Mock state with necessary fields
    utgenerated: UtGeneration = await generate_ut_llm_call(DryRuns.model_validate_json(stock_dry_runs), stock_main, stock_mock_tools)
    ut_trajectory : List[TrajectoryUt] = utgenerated.trajectory_uts
    assert len(ut_trajectory) > 1, "No trajectory UTs generated"
    