from pydantic import BaseModel, Field
from typing import List


class TrajectoryUt(BaseModel):
    input_dict: str = Field(description="input query to test against the workflow, this is a dict ")
    expected_trajectory: List[str] = Field(description="expected tool call trajectory for the avove input query Example: [\"start\", \"agent\", \"search\", \"agent\", \"end\"]")
    
class UtGeneration(BaseModel):
    trajectory_uts: List[TrajectoryUt] = Field(description="The list of TrajectoryUts")

def print_ut(utGeneration: UtGeneration):
    pytest_results_str = ""
    for i, ut in enumerate(utGeneration.trajectory_uts):
        pytest_results_str += f"Trajectory UT {i+1}:\n"
        pytest_results_str += f"  Input: {ut.input_dict}\n"
        # Join the list of strings for better readability in the logs.
        trajectory_str = ", ".join(map(str, ut.expected_trajectory))
        pytest_results_str += f"  Expected Trajectory: {trajectory_str}\n\n"
    return pytest_results_str