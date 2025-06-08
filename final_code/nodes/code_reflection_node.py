from langgraph.prebuilt import create_react_agent
from e2b_code_interpreter import Sandbox
from openevals.code.e2b.execution import create_e2b_execution_evaluator
from langchain_core.runnables import RunnableConfig
from langgraph_reflection import create_reflection_graph
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from final_code.states.AgentBuilderState import AgentBuilderState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from final_code.llms.model_factory import get_model, ModelName
from langchain_core.prompts import ChatPromptTemplate



REFLECTION_SYSTEM_PROMPT = """
 You are an expert software engineer.
 You will be given a langgraph code. You need to fix it and make it runnable.
 Make sure it is correct, complete, and executable without modification.
 Make sure that any generated code is contained in a properly formatted markdown code block.
 """

def create_base_agent():
    def check_code(state: MessagesState):
        llm = get_model(ModelName.GEMINI25FLASH)
        result = llm.invoke([SystemMessage(content=REFLECTION_SYSTEM_PROMPT)]+ state["messages"])
        return {
            "messages": [result]
        }  
    workflow = StateGraph(MessagesState)
    workflow.add_node("check_code", check_code)
    workflow.add_edge(START, "check_code")
    workflow.add_edge("check_code", END)
    app = workflow.compile()
    return app

def create_judge_graph(sandbox: Sandbox):
    def run_reflection(state: MessagesState):
        evaluator = create_e2b_execution_evaluator(
            sandbox=sandbox,
            code_extraction_strategy="markdown_code_blocks",
        )
        py_code = state["messages"][-1].content
        result = evaluator(outputs=py_code)

        code_extraction_failed = result["metadata"] and result["metadata"].get(
            "code_extraction_failed"
        )

        if not result["score"] and not code_extraction_failed:
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": f"I ran the code and found some problems with the code you generated: {result['comment']}\n\n"
                        "Try to fix it. Make sure to regenerate the entire code snippet. "
                        "If you are not sure what is wrong, search for more information by pulling more information "
                        "from the LangGraph docs.",
                    }
                ]
            }

    return (
        StateGraph(MessagesState)
        .add_node("run_reflection", run_reflection)
        .add_edge("__start__", "run_reflection")
        .compile()
    ).with_config(run_name="Judge Agent")

_GLOBAL_SANDBOX = None

def get_or_create_sandbox():
    global _GLOBAL_SANDBOX
    if _GLOBAL_SANDBOX is None:
        _GLOBAL_SANDBOX = Sandbox("OpenEvalsPython")
    return _GLOBAL_SANDBOX

def code_reflection_node_updated(state: AgentBuilderState):
    """
    LangGraph node to run code reflection and fixing using E2B sandbox - updated for AgentBuilderState
    """
    python_code = state['python_code']
    sandbox = get_or_create_sandbox()

    judge = create_judge_graph(sandbox)
    graph = create_base_agent()
    reflection_agent = create_reflection_graph(graph, judge, MessagesState).compile().with_config(run_name="Mini Chat LangChain")

    # Run the reflection agent with the generated code
    result = reflection_agent.invoke({
        "messages": [HumanMessage(content=f"{python_code}\n")]
    })

    # Extract the final message content as the fixed code
    final_message = result["messages"][-1]
    fixed_code = final_message.content if hasattr(final_message, 'content') else str(final_message)

    return {
        "messages": [AIMessage(content="Code has been tested and fixed using E2B sandbox reflection.")],
        "python_code": fixed_code
    }
