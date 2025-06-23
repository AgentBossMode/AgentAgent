# generate use cases
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from final_code.llms.model_factory import get_model, ModelName
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from final_code.nodes.tools.pytest_writing_tools import write_final_response_pytest_code, write_trajectory_pytest_code
from final_code.nodes.code_reflection_node import code_reflection_node_updated
import os
import tempfile
import subprocess
from e2b_code_interpreter import Sandbox
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.types import Command
from typing import Literal
from langgraph.graph import END
from copilotkit import CopilotKitState
load_dotenv()


class CodeEvalState(CopilotKitState):
    reflection_code: str
    python_code: str
    json_dict: str
    use_cases: str
    mocked_code: str
    pytest_code: str
    packages: list[str]
    pytest_out: str


def use_case_generator(state: CodeEvalState):
    SYS_PROMPT= ChatPromptTemplate.from_template("""
You are given the json of a workflow graph below.
{json_dict}
You are supposed to write use cases for the graph.
You will also do dry run of the graph with the use cases.
The use cases should be in the format of a list of dictionaries.
Each dictionary should have the following
keys:
- name: The name of the use case
- description: The description of the use case
- dry_run: The dry run of the use case
""")
    llm = get_model(ModelName.GEMINI25FLASH)
    use_cases = llm.invoke([HumanMessage(content=SYS_PROMPT.format(json_dict=state["json_dict"]))])
    return {"use_cases": use_cases.content}

def mock_code_generator(state: CodeEvalState):
    TOOL_MOCK_PROMPT = """
    You are given a code, look for any methods with 'tool' decorator and modify them to have mock implementation.

    Return the modified code in the same format as the input code.

- DONT WRITE ANYTHING ELSE IN THE OUTPUT, ONLY OUTPUT THE PYTHON CODE, NO MARKDOWNS, no use of ``` blocks
    """
    llm = get_model(ModelName.GEMINI25FLASH)
    mocked_code = llm.invoke([HumanMessage(content=TOOL_MOCK_PROMPT), HumanMessage(content=state["python_code"])])
    return {"mocked_code": mocked_code.content}

def test_writer(state: CodeEvalState):
    TEST_WRITER_PROMPT = """
    You are given langgraph code below:
    <CODE>
    {code}
</CODE>
You are given the use cases for a workflow graph along with dry runs.
<USE_CASES>
{use_cases}
</USE_CASES>
                                                   
You are supposed to write test cases for the graph in the <CODE> section, use the <USE_CASES> for generating test case inputs.
                                              
The tests should cover the following:
1. Final response: use 'write_final_response_pytest_code' tool. the input should be what the user asked and the output would be something that an assistant would respond in a natural language.
2. Trajectory: use the 'write_trajectory_pytest_code' tool.

Both final response and trajectory type tests take a list of inputs and expected outputs.

Output guidelines:
- The given <CODE> should be present at the top of the final output
- DONT WRITE ANYTHING ELSE IN THE OUTPUT, ONLY OUTPUT THE PYTHON CODE, NO MARKDOWNS, no use of ``` blocks
- Code should be compilable python code without errors, no formatting errors. Donot have any SyntaxError
"""
    app = create_react_agent(
    model= get_model(ModelName.GEMINI25FLASH),
    tools=[write_final_response_pytest_code, write_trajectory_pytest_code],
    name="pytest_writer")

    final_response = app.invoke(
        {"messages": [HumanMessage(content=TEST_WRITER_PROMPT.format(code=state["mocked_code"], use_cases=state["use_cases"]))]})
    return {"pytest_code": final_response["messages"][-1].content}

def pytest_runner(state: CodeEvalState):
    pytest_out = []
    sandbox = Sandbox(envs= {"OPENAI_API_KEY" : os.environ["OPENAI_API_KEY"]})
    # sandbox.commands.run("pip install langchain langchain-core langgraph-supervisor langchain-openai langgraph langsmith pytest typing-extensions")
    with tempfile.TemporaryDirectory() as tmpdir:
        code_path = os.path.join(tmpdir, "main.py")
        with open(code_path, "w") as f:
            f.write(state["pytest_code"])

    # Step 2: Run pipreqs to generate requirements.txt
    try:
        subprocess.run(["pipreqs", tmpdir, "--force", "--encoding=utf-8"], check=True)
    except Exception as e:
        result = code_reflection_node_updated.invoke({"code_to_reflect":f"{state["pytest_code"]} ERROR: {e}"})
        state["pytest_code"] = result["reflection_code"]
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = os.path.join(tmpdir, "main.py")
            with open(code_path, "w") as f:
                f.write(state["pytest_code"])
        subprocess.run(["pipreqs", tmpdir, "--force", "--encoding=utf-8"], check=True)
    # Step 3: Read requirements from the generated requirements.txt
        req_path = os.path.join(tmpdir, "requirements.txt")
        with open(req_path, "r") as f:
            requirements = [line.strip() for line in f if line.strip()]

        if requirements:
            joined_packages = " ".join(req.split("==")[0] for req in requirements)
            print(f"Installing: {joined_packages}")
            sandbox.commands.run(f"pip install {joined_packages}")
    sandbox.files.write("/home/user/test_code_to_run.py", state["pytest_code"])
    try:
        commandResult = sandbox.commands.run("pytest -vv /home/user/test_code_to_run.py", background=False, 
                                                     on_stderr=lambda x: print(x),
                                                     on_stdout=lambda x: pytest_out.append(x))
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

    prompt = ChatPromptTemplate.from_messages([SystemMessage(content=EVALUATION_PROMPT), HumanMessage(content=python_code), HumanMessage(content=pytest_results)])
    llm_with_struct_output = llm.with_structured_output(PytestEvaluation)
    eval_result: PytestEvaluation =  llm_with_struct_output.invoke(prompt.invoke())

    if eval_result.is_correct:
        return Command(goto=END)
    else:
        return Command(
            goto="pytest_runner",
            update={"pytest_code": eval_result.pytest_code})


workflow = StateGraph(CodeEvalState)
workflow.add_node("use_case_generator", use_case_generator)
workflow.add_node("mock_code_generator", mock_code_generator)
workflow.add_node("test_writer", test_writer)
workflow.add_node("reflection", reflection_node)
workflow.add_node("pytest_runner", pytest_runner)
workflow.add_node("evaluate_test_results", evaluate_test_results)

workflow.add_edge(START, "use_case_generator")
workflow.add_edge("use_case_generator", "mock_code_generator")
workflow.add_edge("mock_code_generator", "test_writer")

workflow.add_edge("test_writer", "reflection")
workflow.add_edge("reflection", "pytest_runner")
workflow.add_edge("pytest_runner", "evaluate_test_results")

eval_pipeline_graph = workflow.compile()
