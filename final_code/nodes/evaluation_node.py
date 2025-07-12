# generate use cases
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from final_code.llms.model_factory import get_model, ModelName
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from final_code.nodes.tools.pytest_writing_tools import write_final_response_pytest_code, write_trajectory_pytest_code, TRAJECTORY_STR, FINAL_RESPONSE_STR
from final_code.nodes.tools.composio_info_tools import get_raw_tool_schema
from final_code.nodes.code_reflection_node import code_reflection_node_updated
from final_code.states.DryRunState import UseCaseAnalysis
from final_code.pydantic_models.UtGen import UtGeneration
import os
import tempfile
import subprocess
from e2b_code_interpreter import Sandbox
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.types import Command
from typing import Literal, List, Optional
from langgraph.graph import END
from copilotkit import CopilotKitState
load_dotenv()


class CodeEvalState(CopilotKitState):
    reflection_code: str
    python_code: str
    json_dict: str
    use_cases: List[UseCaseAnalysis]
    mocked_code: str
    pytest_code: str
    packages: list[str]
    pytest_out: str


def mock_test_writer(state: CodeEvalState):
    MOCK_TEST_WRITER = """
You are given langgraph code below:
<CODE>
{code}
</CODE>

<INSTRUCTIONS>
1. You will first write mock code stubs for composio tools and any method with the @tool decorator in <CODE> section.
    a. In case of a composio tool --> Follow 'COMPOSIOMOCKINSTRUCTIONS' below:
    <COMPOSIOMOCKINSTRUCTION>
        Let's say you see the following composio tool being initialized
        tool_name = composio_toolset.get_tools(actions=[\"TOOL_NAME_ABC\"])
    
        Instruction:
        1. call 'get_raw_tool_schema' tool, this will fetch information about the TOOL_NAME_ABC
        2. Now using this schema write a python function as follows:
            def tool_name(required input parameters as per the schema output from step 1)
                \"\"\"Docstring including what the tool does, as per the get_raw_tool_schema output \"\"\"
                logic that mocks the tasks of the tool and returns output as per the schema output from step 1 ...
    </COMPOSIOMOCKINSTRUCTION>
    b. In case of any method with @tool decorator --> Follow 'METHODMOCKINSTRUCTIONS' below:
    <METHODMOCKINSTRUCTIONS>
        Read the method docstring, analyze the code, and generate the code again but with mock implementation.
    </METHODMOCKINSTRUCTIONS>

2. Remove any reference of ComposioToolset, related imports etc.
</INSTRUCTIONS>

<OUTPUT>
You are supposed to generate a compilable python file with the mock code.
    <OUTPUT_FORMAT>
        - ONLY THE FINAL PYTHON CODE, NO MARKDOWNS, no use of ``` blocks
        - Code should be compilable python code without errors, no formatting errors
        - No SyntaxError
    </OUTPUT_FORMAT>
</OUTPUT>
"""
    app = create_react_agent(model= get_model(ModelName.GEMINI25FLASH),
    tools=[get_raw_tool_schema],
    name="mock_test_writer")

    
    final_response = app.invoke(
        {"messages": [HumanMessage(content=MOCK_TEST_WRITER.format(code=state["python_code"]))]})
    return {"mocked_code": final_response["messages"][-1].content}


def pytest_writer(state: CodeEvalState):
    
    PYTEST_WRITER_PROMPT = """
You are a python code writing expert, your job is to write a pytest given the langgraph code and use cases.
<CODE>
{code}
</CODE>
You are given the use cases for a workflow graph along with dry runs.
<USE_CASES>
{use_cases}
</USE_CASES>
2. With the mock code generated in step 1, you will now write pytest code,use the 'USE_CASES' to generate test cases for the code in 'CODE' section.The tests should cover the following:
    a. Final response: refer to <FINALRESPONSE> section
    b. Trajectory: refer to <TRAJECTORY> section

<FINALRESPONSE>
{FINAL_RESPONSE_STR}
</FINALRESPONSE>

<TRAJECTORY>
{TRAJECTORY_STR}
</TRAJECTORY>
"""
    use_case_list : List[UseCaseAnalysis] = state["use_cases"]
    use_cases = "\n".join(use_case.model_dump_json(indent=2) for use_case in use_case_list)
    pytest_llm = get_model(ModelName.GEMINI25FLASH).with_structured_output(UtGeneration)
    utgenerated: UtGeneration = pytest_llm.invoke([HumanMessage(content=PYTEST_WRITER_PROMPT.format(code=state["python_code"], use_cases=use_cases, FINAL_RESPONSE_STR=FINAL_RESPONSE_STR, TRAJECTORY_STR=TRAJECTORY_STR))])
    
    inputs = []
    responses = []
    for ut in utgenerated.final_response_uts:
        inputs.append(ut.input)
        responses.append(ut.expected_response)

    final_response_code = write_final_response_pytest_code(inputs, responses)

    inputs_trajectory = []
    responses_trajectory = []
    for ut in utgenerated.trajectory_uts:
        inputs_trajectory.append(ut.input)
        responses_trajectory.append(ut.expected_trajectory)

    final_trajectory_code = write_trajectory_pytest_code(inputs_trajectory, responses_trajectory)

    PYTEST = """
import pytest
from app import app
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

{final_response_code}

{final_trajectory_code}
"""
    return {"pytest_code": PYTEST.format(final_response_code=final_response_code, final_trajectory_code=final_trajectory_code)}


from openevals.code.e2b.sandbox.files import (
    PYTHON_EVALUATOR_SEPARATOR
)


def reflection_node(state: CodeEvalState):
    result = code_reflection_node_updated.invoke({"code_to_reflect": state["mocked_code"]})
    return {"mocked_code": result["reflection_code"]}


EXTRACT_IMPORT_NAMES="""
import ast
import sys

BUILTIN_MODULES = set(sys.stdlib_module_names)

def extract_import_names(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        source_code = f.read()
    
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    python_imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                import_path = name.name
                if not import_path.startswith((".", "/")):
                    base_package = import_path.split(".")[0]
                    if base_package not in python_imports and base_package not in BUILTIN_MODULES:
                        python_imports.append(base_package)
        elif isinstance(node, ast.ImportFrom):
            if node.module and not node.module.startswith((".", "/")):
                base_package = node.module.split(".")[0]
                if base_package not in python_imports and base_package not in BUILTIN_MODULES:
                    python_imports.append(base_package)

    return python_imports

file_path = "./app.py"
imports = extract_import_names(file_path)
print("\\n".join(imports))
"""


def _create_e2b_execution_command(
    *,
    execution_command: str = "python",
) -> str:
    return (" && ").join(
        [
            f"echo '{EXTRACT_IMPORT_NAMES}' > extract_import_names.py",
            "export PIP_DISABLE_PIP_VERSION_CHECK=1",
            "python3 extract_import_names.py > openevals_requirements.txt",
            'pip install -r openevals_requirements.txt',
        ]
    )


def pytest_runner(state: CodeEvalState):
    pytest_out = []
    sandbox = Sandbox(envs= {"OPENAI_API_KEY" : os.environ["OPENAI_API_KEY"], "LANGSMITH_API_KEY": os.environ["LANGSMITH_API_KEY"], "LANGCHAIN_TRACING_V2": os.environ["LANGCHAIN_TRACING_V2"], "LANGCHAIN_PROJECT": "inception_prompt"})

    sandbox.files.write("./app.py", state["mocked_code"])
    sandbox.files.write("./test_app.py", state["pytest_code"])
    cmd = _create_e2b_execution_command()
    sandbox.commands.run(cmd)
    sandbox.commands.run("pip install pytest pytest-xdist")
    try:
        commandResult = sandbox.commands.run("pytest -n 2 -rfEP ./test_app.py",
                                              background=False, 
                                              on_stderr=lambda x: pytest_out.append(x),
                                              on_stdout=lambda x: pytest_out.append(x),
                                              timeout=300)
    except Exception as e:
        print(e)

    return {"pytest_out": "\n".join(pytest_out)}
 

class EvaluationResult(BaseModel):
    no_failures: bool = Field(description="True if there were no failures in the pytest results, otherwise False.")
    file_to_fix: Literal["mocked_code", "pytest_code", "none"] = Field(description="identify the file to fix, if no fix needed then say none")
    fixed_code: Optional[str] = Field(default=None, description= "The code fix proposed for the 'file_to_fix' identified")
    explanation: Optional[str] = Field(default=None, description="Explanation of the changes made for 'file_to_fix', if any.")

def evaluate_test_results(state: CodeEvalState) -> Command[Literal["pytest_runner", "__end__"]]:
    mocked_code = state["mocked_code"]
    pytest_code = state["pytest_code"]
    pytest_out = state["pytest_out"]

    llm=get_model()
    EVALUATION_PROMPT = """
You are provided with the following three things:
1. The 'app' code that needs to be tested.
2. The 'pytest' code written to test the 'app' code.
3. The 'pytest_out' results when the 'pytest' was run


<INSTRUCTIONS>
1. Analyze the 'pytest_out' section, and the 'app' + 'pytest' code together. 
2. If there were no errors in the pytest results, just mark no_failures as true.
3. Else Figure what needs to be fixed, the 'app' code or the 'pytest' code.
4. If the 'app' code needs to be fixed, remember the app code is a langgraph agent, use factual information you have about langgraph and make fixes accordingly.
    a. If a value error is happening or a variable is not populated, remember that is an issue with the code, you will need to update the app code to remove that dependency.
5. Else if the 'pytest' code needs to be fixed, remember that the only fix you are allowed to make is the inputs in pytest parameterize. Or it could be any missing imports that are to be added.
</INSTRUCTIONS>
"""

    MESSAGE_PROMPT= """
User input:
<APP>
{mocked_code}
</APP>

<PYTEST>
{pytest_code}
</PYTEST>

<PYTEST_OUT>
{pytest_out}
</PYTEST_OUT>
"""

    llm_with_struct_output = llm.with_structured_output(EvaluationResult)
    eval_result: EvaluationResult =  llm_with_struct_output.invoke([SystemMessage(content=EVALUATION_PROMPT), HumanMessage(content=MESSAGE_PROMPT.format(mocked_code=mocked_code, pytest_code=pytest_code, pytest_out=pytest_out))])

    if eval_result.no_failures:
        return Command(goto=END)
    else:
        if eval_result.file_to_fix == "mocked_code":
            return Command(
                goto="pytest_runner",
                update={"mocked_code": eval_result.fixed_code})
        elif eval_result.file_to_fix == "pytest_code":
            return Command(
                goto="pytest_runner",
                update={"pytest_code": eval_result.fixed_code})


workflow = StateGraph(CodeEvalState)
workflow.add_node("mock_test_writer", mock_test_writer)
workflow.add_node("pytest_writer", pytest_writer)
workflow.add_node("reflection", reflection_node)
workflow.add_node("pytest_runner", pytest_runner)
workflow.add_node("evaluate_test_results", evaluate_test_results)

workflow.add_edge(START, "mock_test_writer")
workflow.add_edge("mock_test_writer", "pytest_writer")
workflow.add_edge("pytest_writer", "reflection")
workflow.add_edge("reflection", "pytest_runner")
workflow.add_edge("pytest_runner", "evaluate_test_results")
eval_pipeline_graph = workflow.compile()
