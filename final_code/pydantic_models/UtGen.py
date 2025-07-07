from pydantic import BaseModel, Field
from typing import List


class TrajectoryUt(BaseModel):
    input: str = Field(description="List of input queries to test against the workflow Example: [\"What is the capital of France?\", \"How are you?\"]")
    expected_trajectory: List[str] = Field(description="List of expected tool call trajectories for each query Example: [[\"start\", \"agent\", \"search\", \"agent\", \"end\"], [\"start\", \"agent\", \"end\"]]")

class ResponseUt(BaseModel):
    input: str = Field(description="List of input queries to test. Example: [\"What is the capital of France?\", \"Who painted the Mona Lisa?\"]")
    expected_response: str = Field(description="List of expected responses corresponding to each query. Example: [\"The capital of France is Paris.\", \"The Mona Lisa was painted by Leonardo da Vinci.\"]")

class UtGeneration(BaseModel):
    trajectory_uts: List[TrajectoryUt] = Field(description="The inputs for the trajectory related pytest")
    final_response_uts: List[ResponseUt] = Field(description="The inputs for the final response related pytest")

