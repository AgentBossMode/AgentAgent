from final_code.utils.create_e2b_exe_cmd import create_e2b_execution_command
import os 
from e2b_code_interpreter import AsyncSandbox
from langgraph.types import Command
from typing import Literal
from copilotkit.langgraph import copilotkit_emit_state, copilotkit_customize_config
from langchain_core.runnables import RunnableConfig
from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.pydantic_models.UtGen import UtGeneration, print_ut
from final_code.utils.get_filtered_file import get_filtered_file
from final_code.utils.copilotkit_emit_status import append_in_progress_to_list, update_last_status
from final_code.utils.check_is_test import check_is_test
import json



async def pytest_runner(state: AgentBuilderState, config: RunnableConfig) -> Command[Literal["evaluation_start"]]:
    is_test = check_is_test(config)

    pytest_out = []
    modified_config = copilotkit_customize_config(config, emit_messages=False)
    if "attempts" not in state:
        state["attempts"] = 10
    
    if "attempt_num" not in state:
        state["attempt_num"] = 1
    
    if "console_logs" not in state:
        state["console_logs"] = []
    
    async def pytest_results_handler(x: str):
            pytest_out.append(x)
            state["current_tab"] =  "console"
            state["console_logs"] = state["console_logs"] + [x]
            if is_test:
                pass
            else:
                await copilotkit_emit_state(state=state, config=modified_config)
    
    sandbox = await AsyncSandbox.create(envs= {
        "OPENAI_API_KEY" : os.environ["OPENAI_API_KEY"],
        "LANGSMITH_API_KEY": os.environ["LANGSMITH_API_KEY_INCEPTION"], 
        "LANGCHAIN_TRACING_V2": os.environ["LANGCHAIN_TRACING_V2_INCEPTION"],
        "LANGCHAIN_PROJECT": "inception_prompt"})

    await sandbox.files.write("./app.py", get_filtered_file(state["python_code"]))
    await sandbox.files.write("./tools_code.py", get_filtered_file(state["mock_tools_code"]))
    await sandbox.files.write("./test_app.py", get_filtered_file(state["pytest_code"]))
    await sandbox.commands.run(create_e2b_execution_command())
    await sandbox.commands.run("pip install pytest agentevals pytest-xdist pytest-json-report")

    try:  
        state["current_tab"] =  "console"
        state["console_logs_incoming"]= True
        await append_in_progress_to_list(modified_config, state, f"Running pytests... (attempt# {str(state["attempt_num"])})")


        utGeneration: UtGeneration = state["utGeneration"]        
        pytest_results_str = print_ut(utGeneration)

        state["console_logs"] = [pytest_results_str]
        if not is_test:
            await copilotkit_emit_state(state=state, config=modified_config)

        commandResult = await sandbox.commands.run("pytest -n 2 -vv \
                                                    --tb=long \
                                                    --capture=tee-sys \
                                                    --json-report \
                                                    --json-report-file=report.json ./test_app.py",
                                              background=False, 
                                              on_stderr= pytest_results_handler,
                                              on_stdout= pytest_results_handler,
                                              timeout=300)
        json_report = await sandbox.files.read("./report.json")
        await update_last_status(modified_config, state, f"Pytests run completed (attempt# {str(state["attempt_num"])})", True)
    except Exception as e:
        print(f"error: {e}")
        json_report = await sandbox.files.read("./report.json")
        await update_last_status(modified_config, state, f"Pytests run completed (attempt# {str(state["attempt_num"])})", True)
    finally:
        state["current_tab"] =  "console"
        state["console_logs_incoming"]= False
        
    final_report: dict = json.loads(json_report)
    
    return Command(
        update={
            "pytest_report": final_report,
            "current_tab": "console",
            "console_logs": [pytest_results_str] + pytest_out,
            "attempts": state["attempts"]-1,
            "agent_status_list": state["agent_status_list"],
            "attempt_num": state["attempt_num"]
            },
              goto= "evaluation_start")
