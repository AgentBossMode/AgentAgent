from langchain_core.messages import HumanMessage, AIMessage
from pydantic import Field 
from langgraph.graph import StateGraph, START, END
from final_code.nodes.code_reflection_node import code_reflection_node_updated
from final_code.llms.model_factory import get_model
from final_code.utils.create_e2b_exe_cmd import create_e2b_execution_command
from copilotkit import CopilotKitState
from e2b_code_interpreter import Sandbox

# --- LLM Initialization ---
# Initialize the Language Model (LLM) to be used throughout the application
llm = get_model()

class PythonCodeState(CopilotKitState): 
    """
    State for the graph that collects and compiles multiple tool codes.
    """
    python_code: str = Field(description="The Python code generated for the agent")
    mocked_code: str = Field(description= "If there were any failures or errors in the pytest run, this field will contain the corrected code with the fixes applied to the original code.")
    imports: str = Field(description="Required imports to run the generated python code")


compile_prompt = """
You are python code writing expert. You are given 2 snippets of code, your job is to combine them. 
You will be given **two Python code snippets** written using the LangGraph framework:

#### Snippet A (Old Code)
- Contains **complete and functional tool definitions**.
- **Complete but older version of the code** for nodes, edges, and graph creation.

<SNIPPET A>
{python_code}
<\SNIPPET A>

#### Snippet B (New Code)
- Uses the **updated LangGraph implementation and structure** for nodes, edges, and graph creation.
- Contains **mocked or placeholder tool definitions**.

<SNIPPET B>
{refactored_code}
<\SNIPPET B>

### Your Task:

1. **Extract** the working tool definitions from **Snippet A**.
2. **Replace** the mocked or dummy tool definitions in **Snippet B** with the real implementations from Snippet A.
3. **Ensure** the resulting code:
   - Uses the new LangGraph structure from Snippet B.
   - Includes the fully working tools from Snippet A for mocked or placeholder tool definitions.
   - Is **valid Python** and can **compile and run without errors**.
4. **Clean up** any unused code or duplicate definitions.

---

### Output:
Return a **single complete Python file** with:
- The **updated LangGraph syntax** (from Snippet B).
- The **fully working tool definitions** (from Snippet A).
- **No redundant or broken code**.

Make sure the final output is ready to run as-is.
"""


def import_runner(state: PythonCodeState):
    sandbox = Sandbox()
    cmd = create_e2b_execution_command(install_req=False)
    sandbox.commands.run(cmd)
    file_content = sandbox.files.read('./requirements.txt')

    return{
        "messages": [AIMessage(content="extracted imports")],
        "imports": file_content
    }


def reflection_node(state: PythonCodeState):
    result = code_reflection_node_updated.invoke({"code_to_reflect": state["python_code"]})
    return {"python_code": result["reflection_code"]}

def combine_node(state: PythonCodeState):
    python_code = state["python_code"]
    refactored_code = state["mocked_code"]
    compiled_code_output = llm.invoke([HumanMessage(content=compile_prompt.format(
        python_code=python_code,
        refactored_code=refactored_code
    ))])

    return {
        "messages": [AIMessage(content="extracted env variables!")],
        "python_code": compiled_code_output
    } 

workflow = StateGraph(PythonCodeState)
workflow.add_node("reflection", reflection_node)
workflow.add_node("combine_node", combine_node)
workflow.add_node("import_runner", import_runner)

workflow.add_edge(START, "combine_node")
workflow.add_edge("combine_node", "reflection")
workflow.add_edge("reflection", "import_runner")
workflow.add_edge("import_runner", END)
combine_code_pipeline_graph = workflow.compile()