from e2b_code_interpreter import Sandbox
from  src.final_code.utils.custom_multifile_e2b_evaluator import custom_multi_file_e2b_evaluator
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.types import Command
from typing import Literal
from final_code.llms.model_factory import get_model, ModelName
from pydantic import Field
import os

REFLECTION_SYSTEM_PROMPT = """
 You are an expert software engineer.
 You will be given a langgraph code.
 You wil also be provided a tools_code.py file which is present in the same directory as the langgraph code.
 You need to fix the langgraph code and make it runnable.
 Make sure it is correct, complete, and executable without modification.
 Make sure that any generated code is contained in a properly formatted markdown code block.
 Use ChatOpenAI gpt-4o-mini wherever llm is needed.
 
Explain your reasoning for fix along with fixed code in a markdown format. 

Some helpful information about imports for different objects that might be present in the code:

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from typing import Literal
from langgraph.prebuilt import create_react_agent


OUTPUT FORMAT (in markdown): 
1. What are the fixes identified as per the bug
2. the fixed python code in markdown format
 """

class CodeState(MessagesState):
    code_to_reflect: str = Field(description="the input code")
    mock_tools_code: str = Field(description="the mock tools code")
    reflection_code: str= Field(description="the reflection code")
    remaining_steps: int = 4

TOOLS_CODE_SNIPPET="""
<tools_code.py>
{tools_code}
</tools_code.py>
"""
def code_rectification_node(state: CodeState) -> Command[Literal["run_reflection", "__end__"]]:
        if state["remaining_steps"] == 0:
            py_code = state["code_to_reflect"]
            return Command(goto="__end__", update={"reflection_code":py_code, "message": [AIMessage(content="Code reflection taking more than 4 steps")]})
        llm = get_model(ModelName.GEMINI25FLASH)
        result = llm.invoke([SystemMessage(content=REFLECTION_SYSTEM_PROMPT), HumanMessage(content=state["code_to_reflect"]), HumanMessage(content=TOOLS_CODE_SNIPPET.format(tools_code=state["mock_tools_code"]))])
        return Command(goto="run_reflection", update= {
            "code_to_reflect": result.content
        }  )
        
def run_reflection(state: CodeState) -> Command[Literal["__end__", "code_rectification_node"]]:
        if "remaining_steps" not in state:
            state["remaining_steps"] = 4
        sandbox = Sandbox("OpenEvalsPython", timeout=60,envs={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"], "COMPOSIO_API_KEY" : os.environ["COMPOSIO_API_KEY"]})
        evaluator = custom_multi_file_e2b_evaluator(
            sandbox=sandbox,
            code_extraction_strategy="markdown_code_blocks",
        )
        py_code = state["code_to_reflect"]
        #sandbox.files.write("openevals/__init__.py", "")
        sandbox.files.write("openevals/tools_code.py", state["mock_tools_code"])
        result = evaluator(outputs=py_code)
        
        try:
           py_code= sandbox.files.read("openevals/outputs.py")
        except:
           py_code = state["code_to_reflect"]

        if result["score"]:
            return Command(goto="__end__", update={"reflection_code":py_code})
        else:
            return  Command(goto="code_rectification_node", update= {
                "remaining_steps": state["remaining_steps"]-1,
                "code_to_reflect":f"I ran the code and found some problems: {result['metadata']} {result['comment']}\n\n"
                        f"PYTHON CODE that led to above failures: \n\n{py_code}\n\n"
                        "Try to fix it. Make sure to regenerate the entire code snippet. "
                
            })

workflow = StateGraph(CodeState)
workflow.add_node("code_rectification_node", code_rectification_node)
workflow.add_node("run_reflection", run_reflection)

workflow.add_edge(START, "run_reflection")
code_reflection_node_updated = workflow.compile()