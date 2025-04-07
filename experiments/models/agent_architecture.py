from typing import List
from pydantic import BaseModel, Field

class AgentComponent(BaseModel):
    """Represents a component of an agent."""
    name: str = Field(description="Name of the agent component")
    description: str = Field(description="Description of the agent component")


class AgentRelation(BaseModel):
    """Represents a relation between agent components."""
    source: str = Field(description="Source component name")
    target: str = Field(description="Target component name")
    relation_type: str = Field(description="Type of relation")


class AgentArchitecture(BaseModel):
    agentComponents: List[AgentComponent] = Field(description="List of agent components")
    agentRelations: List[AgentRelation] = Field(description="List of agent relations")

