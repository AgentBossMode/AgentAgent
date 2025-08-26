from langgraph.types import Command
from final_code.utils.copilotkit_interrupt_temp import copilotkit_interrupt
from typing import Literal
from final_code.ast_visitors_lib.validation_script import run_detailed_validation
from final_code.states.AgentBuilderState import AgentBuilderState
from langchain_core.runnables import RunnableConfig
from final_code.utils.get_filtered_file import get_filtered_file
from final_code.llms.model_factory import get_model
from langchain_core.messages import HumanMessage, SystemMessage


async def check_for_key_errors_node(state: AgentBuilderState, config: RunnableConfig) -> Command[Literal["key_access_interrupt", "mock_tools_writer"]]:
    python_file = get_filtered_file(state["python_code"])
    errors = run_detailed_validation(python_file)
    if len(errors) > 0:
        llm = get_model()
        from final_code.prompt_lib.debugging_guide.runtime_failures_debugging_guide import key_error_accessing
        response = await llm.ainvoke([SystemMessage(content=f"Follow the debugging guide:\n {key_error_accessing}"),
                               HumanMessage(content=python_file),
                               HumanMessage(content=str(errors))])
        return Command(goto="key_access_interrupt", update={"messages": [response]})
    return Command(goto="mock_tools_writer")

def key_access_interrupt(state: AgentBuilderState) -> Command[Literal["mock_tools_writer"]]:
    answer, response = copilotkit_interrupt("Please answer the above questions!")
    return Command(goto="mock_tools_writer")
        