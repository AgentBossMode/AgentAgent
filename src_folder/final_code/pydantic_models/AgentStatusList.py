from pydantic import BaseModel, Field
from typing import List

class AgentStatusStep(BaseModel):
    status: str = Field(description="doing or Done")
    inProgress: bool = Field(description="inProgress")
    success: bool = Field(description="success")

class AgentStatusList(BaseModel):
    agent_status_steps : List[AgentStatusStep]
