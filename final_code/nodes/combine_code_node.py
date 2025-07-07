import json 
from typing import List 
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field 
from langgraph.graph import MessagesState,StateGraph, START, END
from final_code.utils.fetch_docs import fetch_documents
from final_code.llms.model_factory import get_model
from langgraph.types import interrupt
from final_code.states.NodesAndEdgesSchemas import JSONSchema
from copilotkit import CopilotKitState

# --- LLM Initialization ---
# Initialize the Language Model (LLM) to be used throughout the application
llm = get_model()

class pythonCodeState(CopilotKitState): # Renamed from 'toolcollector' for convention
    """
    State for the graph that collects and compiles multiple tool codes.
    """
    python_code: str = Field(description="The Python code generated for the agent")
    pytest_code: str = Field(description= "If there were any failures or errors in the pytest run, this field will contain the corrected code with the fixes applied to the original code.")

compile_prompt = """
You are python code writing expert. You are given 2 snippets of code, your job is to combine them. 
You will be given **two Python code snippets** written using the LangGraph framework:

#### Snippet A (Old Code)
- Contains **complete and functional tool definitions**.
- Uses **older LangGraph syntax** for nodes, edges, and graph creation.

<SNIPPET A>
{python_code}
<\SNIPPET A>

#### Snippet B (New Code)
- Uses the **updated LangGraph API and structure** for nodes, edges, and graph creation.
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

def env_var_node(state: pythonCodeState):
    python_code = state["python_code"]
    refactored_code = state["pytest_code"]
    compiled_code_output = llm.invoke([HumanMessage(content=compile_prompt.format(
        python_code=python_code,
        refactored_code=refactored_code
    ))])

    return {
        "messages": [AIMessage(content="extracted env variables!")],
        "python_code": compiled_code_output
    } 
