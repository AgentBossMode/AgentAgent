from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from final_code.prompt_lib.debugging_guide import debugging_tsg
from final_code.nodes.code_generation_node import generate_code_gen_prompt
from final_code.llms.model_factory import get_model
from final_code.states.AgentBuilderState import AgentBuilderState
from typing import List, Literal
from final_code.pydantic_models.PytestReport import TestResult
from final_code.utils.copilotkit_emit_status import append_success_to_list, append_failure_to_list, append_in_progress_to_list, update_last_status
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langgraph.graph import END
from copilotkit.langgraph import copilotkit_customize_config
from final_code.utils.check_is_test import check_is_test
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

async def build_failure_string(state: AgentBuilderState) -> str:
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
    llm = get_model()
    if len(assertion_failures) == 0 and len(syntax_failures) == 0:
        output = await llm.ainvoke([SystemMessage(content="Summarize the issue given by the user in a concise format"), HumanMessage(content=json.dumps(state["pytest_report"]))])
        failure_string = output.content
    return failure_string

EVALUATION_PROMPT =  """
You are an expert engineer in debugging issues related to langgraph agents.
User will provide you with a list of failures, which might contain, Syntax Issues, Assertion Failures, or both.
You are also provided with the python_code.py, mock_tools_code.py, and test_app.py files.
Follow the 'INSTRUCTIONS' section for each failure in 'failure_string'

<failure_string>
{failure_string}
</failure_string>

<INSTRUCTIONS>
1. Check the 'DEBUGGING_GUIDE' section below if you find the resolution of an issue there
1. Understand the failures provided in the failure_string and the files provided.
2. Now generate a list of fixes needed to be made, the file that needs fixes, and the type of issue.
3. Whenever suggesting a fix in a file, ensure that the issue is being fixes across the file. i.e. if a bug is identified in line 10 and same bug is present in line 20, then fix both the lines.
4. For suggesting changes to the python_code.py, always make sure it conforms to the patterns described in 'langgraph_context' section.
</INSTRUCTIONS>


<DEBUGGING_GUIDE>
{debugging_guide}
</DEBUGGING_GUIDE>

<file_info>
{file_info}
</file_info>

<langgraph_context>
{langgraph_context}
</langgraph_context>
    """

class EvaluationResult(BaseModel):
    issue_type: Literal["syntax_error", "runtime_error", "assertion_fail"] = Field(description="identify the type of issue")
    file_that_needs_fixes: Literal["python_code", "mock_tools_code", "pytest_code"] = Field(description="identify the file to fix")
    fix_needed: str = Field(description="fixes needed, in a diff format")
    explanation: str = Field(description="Explanation of the issue and the fix needed, in a concise manner")

class EvaluationResults(BaseModel):
    evaluation_results: List[EvaluationResult] = Field(description="List of evaluation results containing issue_type, file_that_needs_fixes, fix_needed, and explanation")
    

async def generate_fixed_code(state: AgentBuilderState, file_to_access_in_state: str, config: RunnableConfig, file_name: str, evaluationResults: EvaluationResults):
    def generate_fix_string(file_name: str, evaluationResults: EvaluationResults):
        fixes_string = ""
        i = 0
        for result in evaluationResults.evaluation_results:
            if result.file_that_needs_fixes == file_name:
                fixes_string += f"fix number: {i}"
                fixes_string += f"fix_needed: {result.fix_needed}" + "\n"
                fixes_string += f"Justification: {result.explanation}" + "\n\n"
                i += 1
        return fixes_string
    
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

    fixes_needed = generate_fix_string(file_to_access_in_state, evaluationResults)

    llm = get_model()
    generated_code = state[file_to_access_in_state]
    if fixes_needed != "":
        generated_code_message = await llm.ainvoke([
            SystemMessage(content=FIX_PROMPT.format(file_to_fix=file_name,
                                                    file_info=get_file_info_prompt(state))),
                                                    HumanMessage(content=fixes_needed)],
                                                    config=config)
        generated_code = generated_code_message.content
    return generated_code

async def evaluation_start(state: AgentBuilderState, config: RunnableConfig) -> Command[Literal["__end__", "evaluation_supervisor"]]:    
    pytest_report: dict = state["pytest_report"]
    if pytest_report.get("summary", {}).get("passed", 0) > 0 and pytest_report.get("summary", {}).get("failed", 0) == 0:
        await append_success_to_list(config, state, "All tests passed, your agent is ready!", False)
        return Command(goto=END, update={ "agent_status_list": state["agent_status_list"] })
    elif state["attempts"] == 0:
        await append_failure_to_list(config, state, "Max attempts reached, please try again.", False)
        return Command(goto=END, update={ "agent_status_list": state["agent_status_list"] })

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
    is_test = check_is_test(config)
    customized_config = copilotkit_customize_config(config, emit_messages=False)
    attempt_num = state["attempt_num"]
    await append_in_progress_to_list(config, state, f"Analyzing failures (attempt# {str(attempt_num)})")
    evaluationResults: EvaluationResults = await get_model().with_structured_output(EvaluationResults).ainvoke(
        [HumanMessage(content=EVALUATION_PROMPT.format(
            langgraph_context=generate_code_gen_prompt(),
            failure_string= await build_failure_string(state),
            file_info=get_file_info_prompt(state),
            debugging_guide=debugging_tsg))])
    
    python_code =  await generate_fixed_code(state, "python_code", customized_config, "python_code.py", evaluationResults)
    mock_tools_code = await generate_fixed_code(state, "mock_tools_code", customized_config, "mock_tools_code.py", evaluationResults)
    pytest_code =  await generate_fixed_code(state, "pytest_code", customized_config, "test_app.py", evaluationResults)

    if is_test:
        return Command(goto="__end__", update={ "python_code": python_code, "mock_tools_code": mock_tools_code, "pytest_code": pytest_code })
    await update_last_status(config, state, f"Analysis completed (attempt# {str(attempt_num)})", True)
    return Command(goto="pytest_runner", update={
        "attempt_num": state["attempt_num"]+1,
        "agent_status_list": state["agent_status_list"],
        "pytest_code": pytest_code,
        "python_code": python_code,
        "mock_tools_code": mock_tools_code,
        "pytest_code": pytest_code })