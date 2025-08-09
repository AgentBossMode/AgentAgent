from e2b_code_interpreter import Sandbox, AsyncSandbox
from  final_code.utils.custom_multifile_e2b_evaluator import custom_multi_file_e2b_evaluator
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from final_code.states.BaseCopilotRenderingState import BaseCopilotRenderingState
from langgraph.types import Command
from typing import Literal
from final_code.llms.model_factory import get_model, ModelName
from pydantic import Field, BaseModel
import os
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_customize_config, copilotkit_emit_state
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


TOOLS_CODE_SNIPPET="""
<tools_code.py>
{tools_code}
</tools_code.py>
"""
async def code_rectification_node(state: BaseCopilotRenderingState, config: RunnableConfig) -> Command[Literal["run_reflection", "__end__"]]:
        modified_config = copilotkit_customize_config(config, emit_messages=False)
        llm = get_model(ModelName.GEMINI25FLASH)
        state["current_status"] = {"inProcess":True ,"status": "Analyzing errors and making fixes."} 
        await copilotkit_emit_state(config=modified_config, state=state)
        result = await llm.ainvoke([SystemMessage(content=REFLECTION_SYSTEM_PROMPT), HumanMessage(content=state["python_code"]), HumanMessage(content=TOOLS_CODE_SNIPPET.format(tools_code=state["mock_tools_code"]))], config=modified_config)
        state["current_status"] = {"inProcess":False ,"status": "Fixes made, now will attempt to run the code."}
        await copilotkit_emit_state(config=modified_config, state=state)
        return Command(goto="run_reflection", update= {
            "python_code": result.content
        }  )
        
async def run_reflection(state: BaseCopilotRenderingState, config: RunnableConfig) -> Command[Literal["__end__", "code_rectification_node"]]:
        sandbox = await AsyncSandbox.create("OpenEvalsPython", timeout=60,envs={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"], "COMPOSIO_API_KEY" : os.environ["COMPOSIO_API_KEY"]})
        evaluator = custom_multi_file_e2b_evaluator(
            sandbox=sandbox,
            code_extraction_strategy="markdown_code_blocks",
        )
        py_code = state["python_code"]
        #sandbox.files.write("openevals/__init__.py", "")
        await sandbox.files.write("openevals/tools_code.py", state["mock_tools_code"])
        result = evaluator(outputs=py_code)
        
        try:
           py_code= await sandbox.files.read("openevals/outputs.py")
           print ("got the pycode from sandbox")
        except:
           py_code = state["python_code"]

        if result["score"]:
            if py_code.startswith("```python"):
               py_code = py_code[len("```python"):]
            if py_code.endswith("```"):
               py_code = py_code[:-len("```")]
            return Command(goto="__end__", update={"python_code":py_code})
        else:
            final_log_str = result['metadata'] + result['comment']
            modified_config = copilotkit_customize_config(config, emit_messages=False)
            state["console_logs"] = state["console_logs"] + [final_log_str]
            await copilotkit_emit_state(config=modified_config, state=state)
            return  Command(goto="code_rectification_node", update= {
                "remaining_steps": state["remaining_steps"]-1,
                "python_code":f"I ran the code and found some problems: {result['metadata']} {result['comment']}\n\n"
                        f"PYTHON CODE that led to above failures: \n\n{py_code}\n\n"
                        "Try to fix it. Make sure to regenerate the entire code snippet. "
            })

workflow = StateGraph(BaseCopilotRenderingState)
workflow.add_node("code_rectification_node", code_rectification_node)
workflow.add_node("run_reflection", run_reflection)

workflow.add_edge(START, "run_reflection")
code_reflection_node_updated = workflow.compile()