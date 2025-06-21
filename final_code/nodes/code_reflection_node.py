from e2b_code_interpreter import Sandbox
from openevals.code.e2b.execution import create_e2b_execution_evaluator
from langgraph_reflection import create_reflection_graph
from langgraph.graph import StateGraph, MessagesState, START, END
from final_code.states.AgentBuilderState import AgentBuilderState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.types import Command
from typing import Literal
from final_code.llms.model_factory import get_model, ModelName
from pydantic import Field, BaseModel
from copilotkit import CopilotKitState
import os

REFLECTION_SYSTEM_PROMPT = """
 You are an expert software engineer.
 You will be given a langgraph code. You need to fix it and make it runnable.
 Make sure it is correct, complete, and executable without modification.
 Make sure that any generated code is contained in a properly formatted markdown code block.
 Use ChatOpenAI gpt-4o-mini wherever llm is needed.

 Explain your reasoning for fix along with fixed code in a markdown format.
 """

class CodeState(MessagesState):
    code_to_reflect: str = Field(description="the input code")
    reflection_code: str= Field(description="the reflection code")
    remaining_steps: int = 4

def code_rectification_node(state: CodeState) -> Command[Literal["run_reflection", "__end__"]]:
        if state["remaining_steps"] == 0:
            return Command(goto="__end__")
        llm = get_model(ModelName.GEMINI25FLASH)
        result = llm.invoke([SystemMessage(content=REFLECTION_SYSTEM_PROMPT)]+ [HumanMessage(content=state["code_to_reflect"])])
        return Command(goto="run_reflection", update= {
            "code_to_reflect": result.content
        }  )
        
def run_reflection(state: CodeState) -> Command[Literal["__end__", "code_rectification_node"]]:
        if "remaining_steps" not in state:
            state["remaining_steps"] = 4
        sandbox = Sandbox("OpenEvalsPython", timeout=60,envs={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"], "COMPOSIO_API_KEY" : os.environ["COMPOSIO_API_KEY"]})
        evaluator = create_e2b_execution_evaluator(
            sandbox=sandbox,
            code_extraction_strategy="markdown_code_blocks",
        )
        py_code = state["code_to_reflect"]
        result = evaluator(outputs=py_code)
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