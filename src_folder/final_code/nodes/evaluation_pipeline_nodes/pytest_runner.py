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
import traceback
from langchain_core.messages import AIMessage
from langgraph.types import Command
from typing import Literal



async def pytest_runner(state: AgentBuilderState, config: RunnableConfig) -> Command[Literal["evaluation_start", "__end__"]]:
    try:
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
        
        state["current_tab"] =  "console"
        state["console_logs_incoming"]= True
        await append_in_progress_to_list(modified_config, state, f"Running pytests... (attempt# {str(state["attempt_num"])})")

        sandbox = await AsyncSandbox.create(envs= {
            "OPENAI_API_KEY" : os.environ["OPENAI_API_KEY"],
            "LANGSMITH_API_KEY": os.environ["LANGSMITH_API_KEY_INCEPTION"], 
            "LANGCHAIN_TRACING_V2": os.environ["LANGCHAIN_TRACING_V2_INCEPTION"],
            "LANGCHAIN_PROJECT": "inception_prompt"})
        await sandbox.commands.run("pip install pytest agentevals pytest-xdist pytest-json-report")

        await sandbox.files.write("./app.py", get_filtered_file(state["python_code"]))
        await sandbox.files.write("./tools_code.py", get_filtered_file(state["mock_tools_code"]))
        await sandbox.files.write("./test_app.py", get_filtered_file(state["pytest_code"]))
        await sandbox.commands.run(create_e2b_execution_command())

        try:
            utGeneration: UtGeneration = state["utGeneration"]        
            pytest_results_str = print_ut(utGeneration)

            state["console_logs"] = [pytest_results_str]
            if not is_test:
                await copilotkit_emit_state(state=state, config=modified_config)

            await sandbox.commands.run("pytest -n 2 -vv \
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
        
        await sandbox.kill()
        return Command(
            update={
                "pytest_report": final_report,
                "current_tab": "console",
                "console_logs": [pytest_results_str] + pytest_out,
                "attempts": state["attempts"]-1,
                "agent_status_list": state["agent_status_list"],
                "attempt_num": state["attempt_num"],
                # Save checkpoint for recovery
                "last_working_pytest_report": final_report,
                "last_working_python_code": state["python_code"],
                "last_working_pytest_code": state["pytest_code"],
                "last_working_mock_tools_code": state["mock_tools_code"]
                },
                  goto= "evaluation_start")
    except Exception as e:
        # Subsequent attempts - try to rollback to last working checkpoint
        has_checkpoint = (state.get("last_working_python_code") and 
                        state.get("last_working_pytest_code") and 
                        state.get("last_working_mock_tools_code") and
                        state.get("last_working_pytest_report"))        
        if has_checkpoint:
            # Rollback to last working checkpoint
            return Command(
                goto="evaluation_start",
                update={
                    "attempt_num": state["attempt_num"],
                    "attempts": state["attempts"]-1,
                    "exception_caught": f"{e}\n{traceback.format_exc()}",
                    "pytest_report": state["last_working_pytest_report"],
                    "python_code": state["last_working_python_code"],
                    "pytest_code": state["last_working_pytest_code"],
                    "mock_tools_code": state["last_working_mock_tools_code"],
                    "messages": [AIMessage(content=f"Some error occured, Rolling back to last working checkpoint and retrying...")]
                }
            )
        else:
            return Command(
                goto="__end__",
                update={
                    "exception_caught": f"{e}\n{traceback.format_exc()}",
                    "messages": [AIMessage(content="An error occurred during running pytest tests. Please try again.")]
                }
            )
