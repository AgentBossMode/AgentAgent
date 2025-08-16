import pytest
from dotenv import load_dotenv
load_dotenv()
from src_folder.tests.test_utils.nutrition_agent_files.nutrition_agent_code import nutrition_agent_code
from src_folder.tests.test_utils.nutrition_agent_files.nutrition_req_analysis import nutrition_req_analysis
from src_folder.tests.test_utils.nutrition_agent_files.nutrition_mock_tools_code import nutrition_mock_tools_code
from src_folder.final_code.pydantic_models.UtGen import UtGeneration, TrajectoryUt, ResponseUt
from src_folder.final_code.nodes.evaluation_pipeline_nodes.pytest_writer import generate_ut_llm_call
from src_folder.final_code.states.ReqAnalysis import ReqAnalysis
from typing import List

@pytest.mark.asyncio
async def test_pytest_writer():
    # Mock state with necessary fields
    utgenerated: UtGeneration = await generate_ut_llm_call(ReqAnalysis.model_validate_json(nutrition_req_analysis), nutrition_agent_code, nutrition_mock_tools_code)
    ut_trajectory : List[TrajectoryUt] = utgenerated.trajectory_uts
    assert len(ut_trajectory) > 1, "No trajectory UTs generated"
    ut_final_response : List[ResponseUt] = utgenerated.final_response_uts
    assert len(ut_final_response) > 1, "No final response UTs generated"
    