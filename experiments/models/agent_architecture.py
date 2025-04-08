from typing import List, Literal
from pydantic import BaseModel, Field

class AgentComponent(BaseModel):
    """Represents a component of an agent."""
    typeOfAgent: Literal["agent_node", "sub_agent"] = Field(description="agent_node if this is a simple node execution, else sub_agent if it may need more components")
    name: str = Field(description="Name of the agent component")
    description: str = Field(description="Description of the agent component")


class AgentRelation(BaseModel):
    """Represents a relation between agent components."""
    source: str = Field(description="Source component name")
    target: str = Field(description="Target component name")
    typeOfRelation: Literal["conditional", "fixed"] = Field(description="Type of relation between components")


class AgentArchitecture(BaseModel):
    agentComponents: List[AgentComponent] = Field(description="List of agent components")
    agentRelations: List[AgentRelation] = Field(description="List of agent relations")

