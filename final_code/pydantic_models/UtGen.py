from pydantic import BaseModel, Field
from typing import List


class TrajectoryUt(BaseModel):
    input: str = Field(description="input query to test against the workflow Example: \"What is the capital of France\"")
    expected_trajectory: List[str] = Field(description="expected tool call trajectory for the avove input query Example: [\"start\", \"agent\", \"search\", \"agent\", \"end\"]")

class ResponseUt(BaseModel):
    input: str = Field(description="Input query to test. Example: \"Who painted the Mona Lisa?\"")
    expected_response: str = Field(description="Expected response corresponding to the above input query. Example: \"The Mona Lisa was painted by Leonardo da Vinci.\"")

class UtGeneration(BaseModel):
    trajectory_uts: List[TrajectoryUt] = Field(description="The list of TrajectoryUts")
    final_response_uts: List[ResponseUt] = Field(description="The list of ResponseUts")

