from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import  Literal, TypedDict
from .node_to_code_base import NodeBuilderState
from pydantic import Field, BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage


def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    options = ["FINISH"] + members


    class Router(BaseModel):
        """Worker to route to next. If no workers needed, route to FINISH."""
        next: Literal[*options] = Field(description="worker to act next") # type: ignore
        reason: str = Field(description="justification for selecting the particular next worker and expectations from it.")


    def supervisor_node(state: NodeBuilderState) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router."""
        plan = state["plan"]
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}.

You are a supervisor tasked with managing a conversation between the
        following workers: {members}. Given the step,
        respond with the worker to act next. The worker will perform a
        task and respond with their results and status. When finished,
        respond with FINISH.
        
        As a supervisor you need to identify which worker to call to execute the step along with justification.
"""

        messages =  [("user", task_formatted)]+[HumanMessage(content=state["node_info"])]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response.next
        if goto == "FINISH":
            return Command(goto="replan", update={"next": "replan",
                                          "past_steps": [(task, response.reason)],
                                          "messages": [AIMessage(content= response.reason)]})
               
        return Command(goto=goto, update={"next": goto,
                                          "task": task,
                                          "messages": [AIMessage(content= response.reason)]})

    return supervisor_node
