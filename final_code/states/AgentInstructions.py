from pydantic import BaseModel, Field
from typing import List


class AgentInstructions(BaseModel):
    """
    Pydantic model to structure the instructions for building the Agent.
    This model is used as a tool for the LLM to output structured information.
    """
    objective: str = Field(description="What is the primary objective of the agent")
    usecases: List[str] = Field(description="What are the various responsibilities of the agent which it needs to fulfill")
    examples: str = Field(description="What are some examples of the usage of the agent (input query and expected output from the agent) ?")

