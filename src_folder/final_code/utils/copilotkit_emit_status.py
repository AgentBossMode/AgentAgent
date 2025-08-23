from langchain_core.runnables import RunnableConfig
from src_folder.final_code.states.AgentBuilderState import AgentBuilderState
from src_folder.final_code.pydantic_models.AgentStatusList import AgentStatusList, AgentStatusStep
from src_folder.final_code.utils.check_is_test import check_is_test
from copilotkit.langgraph import copilotkit_emit_state

async def append_in_progress_to_list(config: RunnableConfig, state: AgentBuilderState, current_status: str, emit_state: bool = True):
    if "agent_status_list" not in state:
        state["agent_status_list"] = AgentStatusList(agent_status_steps=[])

    current_agent_status_list: AgentStatusList = state["agent_status_list"]
    new_agent_status_step : AgentStatusStep = AgentStatusStep(status=current_status, inProgress=True, success=False)
    current_agent_status_list.agent_status_steps.append(new_agent_status_step)
    state["agent_status_list"] = current_agent_status_list
    if emit_state and not check_is_test(config):
        await copilotkit_emit_state(config=config, state=state)

async def append_failure_to_list(config: RunnableConfig, state: AgentBuilderState, current_status: str, emit_state: bool = True):
    if "agent_status_list" not in state:
        state["agent_status_list"] = AgentStatusList(agent_status_steps=[])
    current_agent_status_list: AgentStatusList = state["agent_status_list"]
    new_agent_status_step : AgentStatusStep = AgentStatusStep(status=current_status, inProgress=False, success=False)
    current_agent_status_list.agent_status_steps.append(new_agent_status_step)
    state["agent_status_list"] = current_agent_status_list
    if emit_state and not check_is_test(config):
        await copilotkit_emit_state(config=config, state=state)

async def append_success_to_list(config: RunnableConfig, state: AgentBuilderState, current_status: str, emit_state: bool = True):
    if "agent_status_list" not in state:
        state["agent_status_list"] = AgentStatusList(agent_status_steps=[])
    current_agent_status_list: AgentStatusList = state["agent_status_list"]
    new_agent_status_step : AgentStatusStep = AgentStatusStep(status=current_status, inProgress=False, success=True)
    current_agent_status_list.agent_status_steps.append(new_agent_status_step)
    state["agent_status_list"] = current_agent_status_list
    if emit_state and not check_is_test(config):
        await copilotkit_emit_state(config=config, state=state)

async def update_last_status(config: RunnableConfig, state: AgentBuilderState, updated_status: str, success: bool, emit_state: bool = False):
    current_agent_status_list: AgentStatusList = state["agent_status_list"]
    current_agent_status_list.agent_status_steps[-1].status = updated_status
    current_agent_status_list.agent_status_steps[-1].inProgress = False
    current_agent_status_list.agent_status_steps[-1].success = success
    state["agent_status_list"] = current_agent_status_list
    if emit_state:
        await copilotkit_emit_state(config=config, state=state)

