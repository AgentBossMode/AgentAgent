from final_code.utils.create_e2b_exe_cmd import create_e2b_execution_command
import os 
from e2b_code_interpreter import AsyncSandbox
from pydantic import BaseModel, Field
from langgraph.types import Command
from typing import Literal
from final_code.llms.model_factory import get_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from final_code.nodes.code_generation_node import generate_code_gen_prompt
from copilotkit.langgraph import copilotkit_emit_state, copilotkit_customize_config
from langchain_core.runnables import RunnableConfig
from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.pydantic_models.UtGen import UtGeneration
from final_code.pydantic_models.PytestReport import TestResult
from final_code.utils.get_filtered_file import get_filtered_file
from typing import List
import json
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
"""
    return FILE_INFO.format(python_code= state["python_code"], mock_tools_code=state["mock_tools_code"], pytest_code=state["pytest_code"])

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

async def pytest_runner(state: AgentBuilderState, config: RunnableConfig) -> Command[Literal["evaluation_start"]]:
    if "type" in config and config["type"] == "test":
        is_test =True
    elif "metadata" in config and "type" in config["metadata"] and config["metadata"]["type"] == "test":
        is_test = True
    else:
        is_test = False
    pytest_out = []
    modified_config = copilotkit_customize_config(config, emit_messages=False)
    if "attempts" not in state:
        state["attempts"] = 10
    
    async def pytest_results_handler(x: str):
            pytest_out.append(x)
            state["current_tab"] =  "console"
            state["console_logs"] = state["console_logs"] + [x]
            if is_test:
                pass
            else:
                await copilotkit_emit_state(state=state, config=modified_config)
    
    sandbox = await AsyncSandbox.create(envs= {"OPENAI_API_KEY" : os.environ["OPENAI_API_KEY"], "LANGSMITH_API_KEY": os.environ["LANGSMITH_API_KEY_INCEPTION"], "LANGCHAIN_TRACING_V2": os.environ["LANGCHAIN_TRACING_V2_INCEPTION"], "LANGCHAIN_PROJECT": "inception_prompt"})

    await sandbox.files.write("./app.py", get_filtered_file(state["python_code"]))
    await sandbox.files.write("./tools_code.py", get_filtered_file(state["mock_tools_code"]))
    await sandbox.files.write("./test_app.py", get_filtered_file(state["pytest_code"]))
    cmd = create_e2b_execution_command()
    await sandbox.commands.run(cmd)
    await sandbox.commands.run("pip install pytest pytest-xdist pytest-json-report")
    try:  
        state["current_status"] = {"inProcess":True ,"status": "Running pytests..."}
        state["current_tab"] =  "console"
        state["console_logs_incoming"]= True

        if is_test:
            pytest_results_str = ""
            pass
        else:
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
        state["current_status"] = {"inProcess":False ,"status": "Pytests run completed"}
        state["current_tab"] =  "console"
        state["console_logs_incoming"]= False

        if is_test:
            pass
        else:
            await copilotkit_emit_state(state=state, config=modified_config)
    except Exception as e:
        json_report = await sandbox.files.read("./report.json")
        state["current_status"] = {"inProcess":False ,"status": "Pytests run completed"}
        state["current_tab"] =  "console"
        state["console_logs_incoming"]= False
    final_report: dict = json.loads(json_report)
    return Command(update={"pytest_report": final_report,"current_tab": "console", "console_logs": [pytest_results_str] + pytest_out, "pytest_results": "\n".join(pytest_out), "attempts": state["attempts"]-1}, goto= "evaluation_start")


async def evaluation_start(state: AgentBuilderState) -> Command[Literal["__end__", "evaluation_supervisor"]]:    
  
    pytest_report: dict = state["pytest_report"]
    if pytest_report.get("summary", {}).get("failed", 1) == 0:
        return Command(goto=END)
    elif state["attempts"] == 0:
        return Command(goto=END, update={"current_status":{"inProcess":False ,"status": "Max attempts reached, please try again."} })

    failed_tests: List[TestResult] = [TestResult.model_validate(test) for test in pytest_report["tests"] if test["outcome"] == "failed"]
    assertion_failures : List[TestResult] = []
    syntax_failures : List[TestResult] = []
    for failed_test in failed_tests:
        if failed_test.call.crash.message.__contains__("AssertionError"):
            assertion_failures.append(failed_test)
        else:
            syntax_failures.append(failed_test)
    return Command(goto="evaluation_supervisor", update={"syntax_issues": syntax_failures, "assertion_failures": assertion_failures})

async def evaluation_supervisor(state: AgentBuilderState, config: RunnableConfig) -> Command[Literal["pytest_runner", "__end__"]]:
    if "type" in config and config["type"] == "test":
        is_test =True
    elif "metadata" in config and "type" in config["metadata"] and config["metadata"]["type"] == "test":
        is_test = True
    else:
        is_test = False
    syntax_failures: List[TestResult] = state["syntax_issues"]
    assertion_failures: List[TestResult]  = state["assertion_failures"]

    failure_string = ""

    if len(syntax_failures) > 0:
        failure_string += "<Syntax_Issues>\n"
        i = 0
        for failure in syntax_failures:
            failure_string += f"<FAILURE_{i}>"
            failure_string += f"Test: {failure.nodeid}\n"
            failure_string += f"Traceback: {failure.call.traceback}\n\n"
            failure_string += f"Error: {failure.call.crash}\n\n"
            failure_string += f"</FAILURE_{i}>\n\n"
            i += 1
        failure_string+= "</Syntax_Issues>\n"
    if len(assertion_failures) > 0:
        failure_string += "<Assertion_Failures>"
        i=0
        for failure in assertion_failures:
            failure_string += f"<FAILURE_{i}>" +"\n"
            failure_string += f"Test: {failure.nodeid}\n"
            failure_string += f"Traceback: {failure.call.traceback}\n\n"
            failure_string += f"Error: {failure.call.crash}\n\n"
            failure_string += f"</FAILURE_{i}>\n\n"
            i += 1
        failure_string += "</Assertion_Failures>"
    
    EVALUATION_PROMPT =  """
You are an expert engineer in debugging issues related to langgraph agents.
User will provide you with a list of failures, which might contain, Syntax Issues, Assertion Failures, or both.
You are also provided with the python_code.py, mock_tools_code.py, and test_app.py files.

<failure_string>
{failure_string}
</failure_string>

<file_info>
{file_info}
</file_info>

<langgraph_context>
{langgraph_context}
</langgraph_context>
<INSTRUCTIONS>
1. Understand the failures provided in the failure_string and the files provided.
2. Now generate a list of fixes needed to be made, the file that needs fixes, and the type of issue.
3. Whenever suggesting a fix in a file, ensure that the issue is being fixes across the file. i.e. if a bug is identified in line 10 and same bug is present in line 20, then fix both the lines.
4. For suggesting changes to the python_code.py, always make sure it conforms to the patterns described in 'langgraph_context' section.
</INSTRUCTIONS>
    """
    class EvaluationResult(BaseModel):
        issue_type: Literal["syntax_error", "runtime_error", "assertion_fail"] = Field(description="identify the type of issue")
        file_that_needs_fixes: Literal["python_code", "mock_tools_code", "pytest_code"] = Field(description="identify the file to fix")
        fix_needed: str = Field(description="fixes needed, in a diff format")
        explanation: str = Field(description="Explanation of the issue and the fix needed, in a concise manner")

    class EvaluationResults(BaseModel):
        evaluation_results: List[EvaluationResult] = Field(description="List of evaluation results containing issue_type, file_that_needs_fixes, fix_needed, and explanation")
    llm = get_model()
    llm_evals  = llm.with_structured_output(EvaluationResults)
    evaluationResults: EvaluationResults = await llm_evals.ainvoke([HumanMessage(content=EVALUATION_PROMPT.format(langgraph_context=generate_code_gen_prompt(), failure_string=failure_string, file_info=get_file_info_prompt(state)))])

    # combine the fixes needed in python_code in a single string
    fixes_needed_python_code = ""
    i = 0
    for result in evaluationResults.evaluation_results:
        if result.file_that_needs_fixes == "python_code":
            fixes_needed_python_code += f"fix number: {i}"
            fixes_needed_python_code += f"fix_needed: {result.fix_needed}" + "\n"
            fixes_needed_python_code += f"Justification: {result.explanation}" + "\n\n"
            i += 1

    FIX_PROMPT = """
    You are an expert in making code fixes for langgraph agents.
    <file_info>
    {file_info}
    <file_info>

    You have to fix the following file:
    <file_to_fix>
    {file_to_fix}
    <file_to_fix>

    <INSTRUCTIONS>
   1. You are provided by user a list of fixes with justifications.
   2. You have to analyze the fixes mentioned, and apply them all to the 'file_to_fix'
   3. Generate the compilable final code.
   4. The final code should be executable WITHOUT ANY MARKDOWN blocks.
   5. No ```python at start or ``` at end. THIS IS IMPORTANT
    </INSTRUCTIONS>
"""

    python_code = state["python_code"]
    if fixes_needed_python_code != "":
        python_code_message = await llm.ainvoke([SystemMessage(content=FIX_PROMPT.format(file_to_fix="python_code.py", file_info=get_file_info_prompt(state))), HumanMessage(content=fixes_needed_python_code)], config=copilotkit_customize_config(state, emit_messages=False))
        python_code = python_code_message.content
    # combine the fixes needed in mock_tools_code in a single string
    fixes_needed_mock_tools_code = ""
    i = 0
    for result in evaluationResults.evaluation_results:
        if result.file_that_needs_fixes == "mock_tools_code":
            fixes_needed_mock_tools_code += f"fix number: {i}"
            fixes_needed_mock_tools_code += f"fix_needed: {result.fix_needed}"+ "\n"
            fixes_needed_mock_tools_code += f"Justification: {result.explanation}" + "\n\n"
            i+=1

    mock_tools_code = state["mock_tools_code"]
    if fixes_needed_mock_tools_code != "":
        mock_tools_code_message = await llm.ainvoke([SystemMessage(content=FIX_PROMPT.format(file_to_fix="mock_tools_code.py", file_info=get_file_info_prompt(state))) , HumanMessage(content=fixes_needed_mock_tools_code)], config=copilotkit_customize_config(state, emit_messages=False))
        mock_tools_code = mock_tools_code_message.content

    # combine the fixes needed in pytest_code in a single string
    fixes_needed_pytest_code = ""
    for result in evaluationResults.evaluation_results:
        if result.file_that_needs_fixes == "pytest_code":
            fixes_needed_pytest_code += f"fix number: {i}"
            fixes_needed_pytest_code += f"fix_needed: {result.fix_needed}" + "\n"
            fixes_needed_pytest_code += f"Justification: {result.explanation}"+ "\n\n"
    
    pytest_code = state["pytest_code"]
    if fixes_needed_pytest_code != "":
        pytest_code_message = await llm.ainvoke([SystemMessage(content=FIX_PROMPT.format(file_to_fix="test_app.py", file_info=get_file_info_prompt(state))), HumanMessage(content=fixes_needed_pytest_code)], config=copilotkit_customize_config(state, emit_messages=False))
        pytest_code = pytest_code_message.content
    if is_test:
        return Command(goto="__end__", update={ "python_code": python_code, "mock_tools_code": mock_tools_code, "pytest_code": pytest_code })
    return Command(goto="pytest_runner", update={ "python_code": python_code, "mock_tools_code": mock_tools_code, "pytest_code": pytest_code })