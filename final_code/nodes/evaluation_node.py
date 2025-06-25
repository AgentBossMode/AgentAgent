# generate use cases
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from final_code.llms.model_factory import get_model, ModelName
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from final_code.nodes.tools.pytest_writing_tools import write_final_response_pytest_code, write_trajectory_pytest_code
from final_code.nodes.tools.composio_info_tools import get_raw_tool_schema
from final_code.nodes.code_reflection_node import code_reflection_node_updated
from final_code.states.DryRunState import UseCaseAnalysis
import os
import tempfile
import subprocess
from e2b_code_interpreter import Sandbox
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.types import Command
from typing import Literal, List
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


def test_writer(state: CodeEvalState):
    TEST_WRITER_PROMPT = """
You are a python code writing expert, your job is to write a pytest given the langgraph code and use cases.
You are given langgraph code below:
<CODE>
{code}
</CODE>
You are given the use cases for a workflow graph along with dry runs.
<USE_CASES>
{use_cases}
</USE_CASES>

<INSTRUCTIONS>
1. You will first write mock code stubs for composio tools and any method with the @tool decorator in <CODE> section.
    a. In case of a composio tool --> Follow <COMPOSIOMOCKINSTRUCTIONS> below:
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
    b. In case of any method with @tool decorator --> Follow <METHODMOCKINSTRUCTIONS> below:
    <METHODMOCKINSTRUCTIONS>
     1. Read the method docstring, analyze the code, and generate the code again but with mock implementation.
    </METHODMOCKINSTRUCTIONS>
2. With the mock code generated in step 1, you will now write pytest code,use the <USE_CASES> for generating test case inputs.The tests should cover the following:
    a. Final response: use 'write_final_response_pytest_code' tool. the input should be what the user asked and the output would be something that an assistant would respond in a natural language.
    b. Trajectory: use the 'write_trajectory_pytest_code' tool.
    c. Both final response and trajectory type tests take a list of inputs and expected outputs.

3. Remove any reference of ComposioToolset, related imports etc.
</INSTRUCTIONS>

<OUTPUT>
You are supposed to generate a compilable python file.
1. The generated MOCK CODE should be present at the top of the final output
2. The generated pytest code in step 2 of INSTRUCTION should be appended at bottom
3. All imports from both code pieces are supposed to be at the top of the output.
    <OUTPUT_FORMAT>
        - ONLY THE FINAL PYTHON CODE, NO MARKDOWNS, no use of ``` blocks
        - Code should be compilable python code without errors, no formatting errors
        - No SyntaxError
    </OUTPUT_FORMAT>
</OUTPUT>
"""
    app = create_react_agent(model= get_model(ModelName.GEMINI25FLASH),
    tools=[write_final_response_pytest_code, write_trajectory_pytest_code, get_raw_tool_schema],
    name="pytest_writer")

    use_case_list : List[UseCaseAnalysis] = state["use_cases"]
    use_cases = "\n".join(use_case.model_dump_json(indent=2) for use_case in use_case_list)
    final_response = app.invoke(
        {"messages": [HumanMessage(content=TEST_WRITER_PROMPT.format(code=state["python_code"], use_cases=use_cases))]})
    return {"pytest_code": final_response["messages"][-1].content}

from openevals.code.e2b.sandbox.files import (
    PYTHON_EVALUATOR_SEPARATOR
)


def reflection_node(state: CodeEvalState):
    result = code_reflection_node_updated.invoke({"code_to_reflect": state["pytest_code"]})
    return {"pytest_code": result["reflection_code"]}


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

file_path = "./test_runner.py"
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
    sandbox = Sandbox(envs= {"OPENAI_API_KEY" : os.environ["OPENAI_API_KEY"]})

    sandbox.files.write("./test_runner.py", state["pytest_code"])
    cmd = _create_e2b_execution_command()
    sandbox.commands.run(cmd)
    sandbox.commands.run("pip install pytest-xdist")
    try:
        commandResult = sandbox.commands.run("pytest -n auto -rfEP ./test_runner.py",
                                              background=False, 
                                              on_stderr=lambda x: pytest_out.append(x),
                                              on_stdout=lambda x: pytest_out.append(x),
                                              timeout=300)
    except Exception as e:
        print(e)

    return {"pytest_out": "\n".join(pytest_out)}
 
def reflection_node(state: CodeEvalState):
    result = code_reflection_node_updated.invoke({"code_to_reflect": state["pytest_code"]})
    return {"pytest_code": result["reflection_code"]}

class PytestEvaluation(BaseModel):
    is_correct: bool = Field(description="True if there were no failures or errors in the pytest run, otherwise False.")
    pytest_code: str = Field(description= "If there were any failures or errors in the pytest run, this field will contain the corrected code with the fixes applied to the original code.")
    explanation: str = Field(description="Explanation of the changes made to the code, if any.")

def evaluate_test_results(state: CodeEvalState) -> Command[Literal["pytest_runner", "__end__"]]:
    python_code = state["pytest_code"]
    pytest_results = state["pytest_out"]

    llm=get_model()
    EVALUATION_PROMPT = """
You will be provided with a code that would contain the main code along with the pytests

You will also be provided with the output of the pytest run.

Your job is to evaluate the pytest results and see if there were any errors or failures in the pytest run.
Based on the failures, you are supposed to perform corrections in the code and return the corrected code.
"""

    llm_with_struct_output = llm.with_structured_output(PytestEvaluation)
    eval_result: PytestEvaluation =  llm_with_struct_output.invoke([SystemMessage(content=EVALUATION_PROMPT), HumanMessage(content=python_code), HumanMessage(content=pytest_results)])

    if eval_result.is_correct:
        return Command(goto=END)
    else:
        return Command(
            goto="pytest_runner",
            update={"pytest_code": eval_result.pytest_code})


workflow = StateGraph(CodeEvalState)
workflow.add_node("test_writer", test_writer)
workflow.add_node("reflection", reflection_node)
workflow.add_node("pytest_runner", pytest_runner)
workflow.add_node("evaluate_test_results", evaluate_test_results)

workflow.add_edge(START, "test_writer")
workflow.add_edge("test_writer", "reflection")
workflow.add_edge("reflection", "pytest_runner")
workflow.add_edge("pytest_runner", "evaluate_test_results")

eval_pipeline_graph = workflow.compile()
