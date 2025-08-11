from final_code.utils.create_e2b_exe_cmd import create_e2b_execution_command
import os 
from e2b_code_interpreter import AsyncSandbox
from pydantic import BaseModel, Field
from langgraph.types import Command
from typing import Literal
from final_code.llms.model_factory import get_model
from langchain_core.messages import HumanMessage
from langgraph.graph import END
from final_code.nodes.code_generation_node import generate_code_gen_prompt
from copilotkit.langgraph import copilotkit_emit_state, copilotkit_customize_config
from langchain_core.runnables import RunnableConfig
from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.pydantic_models.UtGen import UtGeneration
def get_file_info_prompt(state: AgentBuilderState):
    FILE_INFO = """
<python_code.py code>
{python_code}
</python_code.py code>

<mock_tools_code.py code>
{mock_tools_code}
</mock_tools_code.py code>

<test_app.py code>
{pytest_code}
</test_app.py code>

<RELATION_OF_FILES>
test_app.py imports app from python_code.py 
python_code.py imports mock_tools_code.py
</RELATION_OF_FILES>

<pytest_results>
{pytest_results}
</pytest_results>
"""
    return FILE_INFO.format(python_code= state["python_code"], mock_tools_code=state["mock_tools_code"], pytest_code=state["pytest_code"], pytest_results=state["pytest_results"])

def get_context_info_prompt(state: AgentBuilderState):
    CONTEXT_WINDOW = """
<issue_type>
{issue_type}
</issue_type>

<file_that_needs_fixes>
{file_that_needs_fixes}
</file_that_needs_fixes>

<fix_needed>
{fix_needed}
</fix_needed>
"""
    return CONTEXT_WINDOW.format(issue_type=state["issue_type"], file_that_needs_fixes=state["file_that_needs_fixes"], fix_needed=state["fix_needed"])

async def pytest_runner(state: AgentBuilderState, config: RunnableConfig) -> Command[Literal["evaluation_supervisor"]]:
    pytest_out = []
    modified_config = copilotkit_customize_config(config, emit_messages=False)
    if "attempts" not in state:
        state["attempts"] = 5
    if state["attempts"] > 0:
        state["attempts"] -= 1
    else:
        return Command(goto=END, update={"current_status":{"inProcess":False ,"status": "Max attempts reached, please try again."} })

    async def pytest_results_handler(x: str):
            pytest_out.append(x)
            state["current_tab"] =  "console"
            state["console_logs"] = state["console_logs"] + [x]
            await copilotkit_emit_state(state=state, config=modified_config)
    
    sandbox = await AsyncSandbox.create(envs= {"OPENAI_API_KEY" : os.environ["OPENAI_API_KEY"], "LANGSMITH_API_KEY": os.environ["LANGSMITH_API_KEY_INCEPTION"], "LANGCHAIN_TRACING_V2": os.environ["LANGCHAIN_TRACING_V2_INCEPTION"], "LANGCHAIN_PROJECT": "inception_prompt"})


    await sandbox.files.write("./app.py", state["python_code"])
    await sandbox.files.write("./tools_code.py", state["mock_tools_code"])
    await sandbox.files.write("./test_app.py", state["pytest_code"])
    cmd = create_e2b_execution_command()
    await sandbox.commands.run(cmd)
    await sandbox.commands.run("pip install pytest pytest-xdist")
    try:
        
        state["current_status"] = {"inProcess":True ,"status": "Running pytests..."}
        state["current_tab"] =  "console"
        state["console_logs_incoming"]= True
        utGeneration: UtGeneration = state["utGeneration"]        
        pytest_results_str = ""
        for i, ut in enumerate(utGeneration.final_response_uts):
            pytest_results_str += f"Final Response UT {i+1}:\n"
            pytest_results_str += f"  Input: {ut.input}\n"
            pytest_results_str += f"  Expected Response: {ut.expected_response}\n\n"

        for i, ut in enumerate(utGeneration.trajectory_uts):
            pytest_results_str += f"Trajectory UT {i+1}:\n"
            pytest_results_str += f"  Input: {ut.input}\n"
            # Join the list of strings for better readability in the logs.
            trajectory_str = ", ".join(map(str, ut.expected_trajectory))
            pytest_results_str += f"  Expected Trajectory: {trajectory_str}\n\n"
        
        state["console_logs"] = [pytest_results_str]
        await copilotkit_emit_state(state=state, config=modified_config)
        commandResult = await sandbox.commands.run("pytest -n 2 -rfEP ./test_app.py",
                                              background=False, 
                                              on_stderr= pytest_results_handler,
                                              on_stdout= pytest_results_handler,
                                              timeout=300)
        state["current_status"] = {"inProcess":False ,"status": "Pytests run completed"}
        state["current_tab"] =  "console"
        state["console_logs_incoming"]= False

        await copilotkit_emit_state(state=state, config=modified_config)
    except Exception as e:
        print(e)

    return Command(update={"current_tab": "console", "console_logs": pytest_results_str + pytest_out, "pytest_results": "\n".join(pytest_out)}, goto= "evaluation_supervisor")


async def syntax_and_runtime_issues_node(state: AgentBuilderState, config: RunnableConfig) -> Command[Literal["pytest_runner"]]:
    llm = get_model()
    SYNTAX_AND_RUNTIME_ISSUE_FIX_PROMPT = """
You are an expert in making code fixes for langgraph agents.

<INSTRUCTIONS>
1. Understand the 'issue_type' and 'fix_needed' for the 'file_that_need_fixes'
2. Read the 'python_code.py', 'mock_tools_code.py', 'test_app.py' for reference.
3. Before applying fixes, make sure whatever fixes you make comply with the 'langgraph_context' section which acts as a guide for code generation.
4. Generate the compilable final code.
5. The final code should be executable WITHOUT ANY MARKDOWN blocks.
</INSTRUCTIONS>

{context_info}

{file_info}

<langgraph_context>
{langgraph_context}
</langgraph_context>
"""
    modified_config = copilotkit_customize_config(config, emit_messages=False)
    state["current_tab"] =  "console"
    state["current_status"] = {"inProcess":True ,"status": "Evaluating syntax and runtime failures..."}
    await copilotkit_emit_state(state=state, config=modified_config)
    
    final_code = await llm.ainvoke([HumanMessage(content=SYNTAX_AND_RUNTIME_ISSUE_FIX_PROMPT.format(
        context_info=get_context_info_prompt(state),
        file_info=get_file_info_prompt(state),
        langgraph_context=generate_code_gen_prompt()
    ))], config=modified_config)

    state["current_tab"] =  "console"
    state["current_status"] = {"inProcess":False ,"status": "Pytest errors evaluated, making code changes."}
    await copilotkit_emit_state(state=state, config=modified_config)
    if state["file_that_needs_fixes"] == "python_code":
        return Command(update={"python_code": final_code.content}, goto="pytest_runner")
    elif state["file_that_needs_fixes"] == "mock_tools_code":
        return Command(update={"mock_tools_code": final_code.content}, goto="pytest_runner")
    elif state["file_that_needs_fixes"] == "pytest_code":
        return Command(update={"pytest_code": final_code.content}, goto="pytest_runner")
    

async def fix_assert_fail_issue_node(state: AgentBuilderState, config: RunnableConfig) -> Command[Literal["pytest_runner"]]:
    llm = get_model()
    ASSERTION_FAIL_FIXES_PROMPT = """
You are an expert engineer building and evaluating langgraph agent. Just understand why the assertions are failing, what fixes need to be made.
<INSTRUCTIONS>
1. Understand the 'issue_type' and 'fix_needed' for the 'file_that_need_fixes'
2. Read the 'python_code.py', 'mock_tools_code.py', 'test_app.py' for reference.
3. You are to see what fixes need to be made, why the assertion is failing, and see how you can make the changes. If the selected file is 'python_code.py' make sure the changes being made are being done by checking langgraph_context
4. Generate the compilable final code.
5. The final code should be executable WITHOUT ANY MARKDOWN blocks.
</INSTRUCTIONS>
{context_info}

{file_info}

<langgraph_context>
{langgraph_context}
</langgraph_context>
"""
    modified_config = copilotkit_customize_config(config, emit_messages=False)
    state["current_status"] = {"inProcess":True ,"status": "Evaluating assertion failures..."}
    await copilotkit_emit_state(state=state, config=modified_config)
    
    final_code = await llm.ainvoke([HumanMessage(content=ASSERTION_FAIL_FIXES_PROMPT.format(context_info=get_context_info_prompt(state),
        file_info=get_file_info_prompt(state),
        langgraph_context=generate_code_gen_prompt()))], config=modified_config)
    
    state["current_status"] = {"inProcess":False ,"status": "Assertion failures evaluated, making fixes"}
    await copilotkit_emit_state(state=state, config=modified_config)

    if state["file_that_needs_fixes"] == "python_code":
        return Command(update={"python_code": final_code.content}, goto="pytest_runner")
    elif state["file_that_needs_fixes"] == "mock_tools_code":
        return Command(update={"mock_tools_code": final_code.content}, goto="pytest_runner")
    elif state["file_that_needs_fixes"] == "pytest_code":
        return Command(update={"pytest_code": final_code.content}, goto="pytest_runner")


EVALUATION_SUPERVISOR_PROMPT ="""
1. You are provided with 3 files: app.py, mock_tools.py, and test_app.py. 
2. You are also provided with pytest_results which were obtained for the tests present in test_app.py
3. Your job is to first see if the pytest_results show any failures, if not then set no_failures field to true
4. If pytest_results does show failure, using app.py, mock_tools.py and test_app.py as your context, identify the type of issue, the file needing fix, and the fix needed

{file_info}
"""
async def evaluation_supervisor(state: AgentBuilderState) -> Command[Literal["syntax_and_runtime_issues_node", "fix_assert_fail_issue_node", "__end__"]]:    
    class EvaluationResult(BaseModel):
        no_failures: bool = Field(description="True if there were no failures in the pytest results, otherwise False.")
        issue_type: Literal["syntax_error", "runtime_error", "assertion_fail"] = Field(description="identify the type of issue")
        file_that_needs_fixes: Literal["python_code", "mock_tools_code", "pytest_code"] = Field(description="identify the file to fix")
        fix_needed: str = Field(description="detailed explanation of fixes needed, in a diff format")

    llm = get_model()
    llm_eval_decide = llm.with_structured_output(EvaluationResult)
    eval_result: EvaluationResult = await llm_eval_decide.ainvoke([HumanMessage(content=EVALUATION_SUPERVISOR_PROMPT.format(file_info=get_file_info_prompt(state)))])
    if eval_result.no_failures:
        return Command(goto=END)
    elif eval_result.issue_type == "syntax_error" or eval_result.issue_type == "runtime_error":
        return Command(update={"issue_type": eval_result.issue_type, "file_that_needs_fixes": eval_result.file_that_needs_fixes, "fix_needed": eval_result.fix_needed}, goto="syntax_and_runtime_issues_node")
    elif eval_result.issue_type == "assertion_fail":
        return Command(update={"issue_type": eval_result.issue_type, "file_that_needs_fixes": eval_result.file_that_needs_fixes, "fix_needed": eval_result.fix_needed}, goto="fix_assert_fail_issue_node")
