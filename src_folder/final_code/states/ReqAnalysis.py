from pydantic import BaseModel, Field
from typing import List, Literal
class Purpose(BaseModel):
    purpose: str = Field(description="What the purpose of the agent, 2 to 3 words long")
    emoji: str = Field(description="An emoji to represent the purpose")
    description: str = Field(description="Explain purpoose in 1-2 concise sentences")
    confident: bool = Field(description="Set true if high confidence that the above is intended given the input")
    selected: bool = Field(default="false")


class TargettedUser(BaseModel):
    persona: str = Field(description="Describe the persona, 2-3 words")
    emoji: str = Field(description="An emoji to represent the persona")
    description: str = Field(description="Explain why they might need this agent, 1 sentence")
    confident: bool = Field(description="Set true if high confidence that the above is intended given the input")
    selected: bool = Field(default="false")


class Capabiity(BaseModel):
    capability: str = Field(description="Describe a capability of the agent, 2 to 3 words")
    emoji: str = Field(description="An emoji to represent the capability")
    description: str = Field(description="Explain what it will do, 1-2 concise sentences")
    does_need_trigger: str = Field(description="is this capability which will need an external trigger to initiate the work")
    confident: bool = Field(description="Set true if high confidence that the above is intended given the input")
    selected: bool = Field(default="false")


class KnowledgeAndDataRequirements(BaseModel):
    knowledge_source: str = Field(description="Describe the knowledge source, 2-3 words")
    emoji: str = Field(description="An emoji to represent the knowledge source")
    description: str = Field(description="Why this knowledge source would be needed in 1-2 concise sentences")
    confident: bool = Field(description="Set true if high confidence that the above is intended given the input")
    selected: bool = Field(default="false")


class Tool(BaseModel):
    tool: str = Field(description="Describe the tool, 2-3 words")
    emoji: str = Field(description="An emoji to represent the tool")
    description: str = Field(description="Why this tool would be needed in 1-2 concise sentences, some famous examples of tools, for example for repository related work - Github, for Crm - hubspot, apollo etc")
    specific_tasks: List[str] = Field(description="The individualistic tasks, Create, Read, Update, Delete etc.., no fuzzy logic or no algorithm")
    confident: bool = Field(description="Set true if high confidence that the above is intended given the input")
    tooling_type: Literal["invokable", "pollable", "invokeORpoll"] = Field(description="This tool can be invoked by the agent, or could emit events that could trigger the agent execution with the provided data from the tool")
    selected: bool = Field(default="false")


class AgentAction(BaseModel):
    determinism: Literal["deterministic", "llm"] = Field(description="Description of the action, if this action can be done via plain non-ML python code, say deterministic, that does not mean string parsing on the input or any other by product")
    knowledge: str = Field(description="which knowledge base is used and why is it needed")
    toolings: str = Field(description="What toolings are used to to be used by this step")
    action: str = Field(description="Ex format: Using the 'knowledge', agent learns about 'abc' and then uses the x tool to do 'xyz'")

class DryRun(BaseModel):
    input_type: Literal["trigger", "on-demand"] = Field(description="on-demand if user is chatting with the agent, trigger if lets say webhook based/reminder/event based execution")
    input: str = Field(description="The input information for the dry run")
    agent_actions: List[AgentAction] = Field(description ="A list of actions performed by the agent in the dry run, required field")
    output: str = Field(description="The end result, what is achieved by the agent")
    selected: bool = Field(default="false")


class ReqAnalysis(BaseModel):
    purposes:  List[Purpose]= Field(description="Provide 5 suggestions")
    capabilities: List[Capabiity] = Field(description="Provide 5 suggestions")
    knowledge_sources: List[KnowledgeAndDataRequirements] = Field(description="Provide 5 suggestions")
    targetted_users: List[TargettedUser] = Field(description="Provide 5 suggestions")
    toolings: List[Tool] = Field(description="Provide 5 suggestions")
    additional_information: str = Field(description="Any additional information in user input not categorizable in above categories")
    dry_runs: List[DryRun] = Field(description="Provide 5 suggestions, Different dry runs, to help visualize how the agent should work")
    user_selections:dict = Field(default=None)