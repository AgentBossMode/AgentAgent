import pytest
from tests.test_utils.trading.py_code import py_code
from tests.test_utils.trading.use_cases import use_cases
from src.final_code.pydantic_models.UtGen import UtGeneration, TrajectoryUt, ResponseUt
from src.final_code.nodes.evaluation_node import generate_ut_llm_call
from typing import List

def test_pytest_writer():
    # Mock state with necessary fields
    utgenerated: UtGeneration = generate_ut_llm_call(use_cases, py_code)
    ut_trajectory : List[TrajectoryUt] = utgenerated.trajectory_uts
    print(ut_trajectory)
    assert len(ut_trajectory) > 1, "No trajectory UTs generated"
    ut_final_response : List[ResponseUt] = utgenerated.final_response_uts
    print(ut_final_response)
    assert len(ut_final_response) > 1, "No final response UTs generated"
    