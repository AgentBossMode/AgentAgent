import requests
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from bs4 import BeautifulSoup
from e2b_code_interpreter import Sandbox
from langgraph.graph import StateGraph, MessagesState
from openevals.code.e2b.pyright import create_e2b_pyright_evaluator



SYSTEM_PROMPT = """
You are an expert software engineer.

You will be given a langgraph code. You need to fix it and make it runnable.
If you are not aware about some piece of code or a prebuilt function,use the get_langgraph_docs_index tool to get an index of the LangGraph docs first,
then follow up with the get_request tool. Be persistent - if your first page does
not result in confident information, keep digging!

Make sure it is correct, complete, and executable without modification.
Make sure that any generated code is contained in a properly formatted markdown code block.

You can use the following URLs with your "get_langgraph_docs_content" tool to help answer questions:

{langgraph_llms_txt}
"""


@tool
def get_langgraph_docs_content(url: str) -> str:
    """Sends a get request to a webpage and returns plain text
    extracted via BeautifulSoup."""
    res = requests.get(url).text
    soup = BeautifulSoup(res, features="html.parser")
    return soup.get_text()


def create_base_agent(model: BaseChatModel):
    langgraph_llms_txt = requests.get(
        "https://langchain-ai.github.io/langgraph/llms.txt"
    ).text
    return create_react_agent(
        model=model,
        tools=[get_langgraph_docs_content],
        prompt=SYSTEM_PROMPT.format(langgraph_llms_txt=langgraph_llms_txt),
    ).with_config(run_name="Base Agent")


def create_judge_graph(sandbox: Sandbox):
    def run_reflection(state: dict) -> dict | None:
        evaluator = create_e2b_pyright_evaluator(
            sandbox=sandbox,
            code_extraction_strategy="markdown_code_blocks",
        )

        result = evaluator(outputs=state)

        code_extraction_failed = result["metadata"] and result["metadata"].get(
            "code_extraction_failed"
        )

        if not result["score"] and not code_extraction_failed:
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": f"I ran pyright and found some problems with the code you generated: {result['comment']}\n\n"
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

from typing import Optional
from e2b_code_interpreter import Sandbox
from langgraph.graph import MessagesState
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langgraph_reflection import create_reflection_graph


_GLOBAL_SANDBOX = None


def get_or_create_sandbox():
    global _GLOBAL_SANDBOX
    if _GLOBAL_SANDBOX is None:
        _GLOBAL_SANDBOX = Sandbox("OpenEvalsPython")
    return _GLOBAL_SANDBOX


def create_reflection_agent(
    config: RunnableConfig,
):
    configurable = config.get("configurable", {})
    sandbox = configurable.get("sandbox", None)
    model = configurable.get("model", None)
    if sandbox is None:
        sandbox = get_or_create_sandbox()
    if model is None:
        model = init_chat_model(
            model="gpt-4.1-mini",
            max_tokens=4096,
        )
    judge = create_judge_graph(sandbox)
    return (
        create_reflection_graph(create_base_agent(model), judge, MessagesState)
        .compile()
        .with_config(run_name="Mini Chat LangChain")
    )
    
final_agent = create_reflection_agent({})
