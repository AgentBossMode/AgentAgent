from final_code.states.AgentBuilderState import AgentBuilderState
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from langchain_core.messages import AIMessage
from langgraph.types import interrupt, Command
from langgraph.errors import GraphInterrupt
from typing import Literal
import traceback
from langchain_core.messages import AIMessage


def tool_interrupt(state: AgentBuilderState, config: RunnableConfig) -> Command[Literal["generate_additional_info_questions", "__end__"]]:
    try:
        confirmation = interrupt({
            "type": "toolset_integration",
            "tool_list" : state["tool_set"]})
        return Command(
            goto="generate_additional_info_questions",
            update={"current_tab": "code"}
        )
    except Exception as e:
        if isinstance(e, GraphInterrupt):
            raise
        return Command(
            goto="__end__",
            update={
                "exception_caught": f"{e}\n{traceback.format_exc()}",
                "messages": [AIMessage(content="An error occurred during tool confirmation. Please check logs for details.")]
            }
        )