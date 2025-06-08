from e2b_code_interpreter import Sandbox
from openevals.code.e2b.execution import create_e2b_execution_evaluator
from langgraph_reflection import create_reflection_graph
from langgraph.graph import StateGraph, MessagesState, START, END
from final_code.states.AgentBuilderState import AgentBuilderState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from final_code.llms.model_factory import get_model, ModelName
from pydantic import Field
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
    graph_schema: str = Field(description="the graph schema obtained from e2b.")
    main_logs: str = Field(description="logs appended from main.")

def create_base_agent():
    def check_code(state: CodeState):
        llm = get_model(ModelName.GEMINI25FLASH)
        result = llm.invoke([SystemMessage(content=REFLECTION_SYSTEM_PROMPT)]+ [state["messages"][-1]])
        return {
            "messages": [result]
        }  
    workflow = StateGraph(CodeState)
    workflow.add_node("check_code", check_code)
    workflow.add_edge(START, "check_code")
    workflow.add_edge("check_code", END)
    app = workflow.compile()
    return app

def create_judge_graph(sandbox: Sandbox):
    def run_reflection(state: CodeState):
        evaluator = create_e2b_execution_evaluator(
            sandbox=sandbox,
            code_extraction_strategy="markdown_code_blocks",
        )
        py_code = state["messages"][-1].content
        result = evaluator(outputs=py_code)
        try:
            graph_schema = sandbox.files.read("/home/user/graph.json")
        except:
            graph_schema = "File not found"

        try:
            main_logs = sandbox.files.read("/home/user/llm_stream.txt")
        except:
            main_logs = "File not found"
        

        code_extraction_failed = result["metadata"] and result["metadata"].get(
            "code_extraction_failed"
        )

        if not result["score"] and not code_extraction_failed:
            return {
                "graph_schema": graph_schema,
                "main_logs": main_logs,
                "messages": [
                    {
                        "role": "user",
                        "content": f"I ran the code and found some problems: {result['comment']}\n\n"
                        f"PYTHON CODE that led to above failures: \n\n{py_code}\n\n"
                        "Try to fix it. Make sure to regenerate the entire code snippet. ",
                    }
                ]
            }

    return (
        StateGraph(CodeState)
        .add_node("run_reflection", run_reflection)
        .add_edge("__start__", "run_reflection")
        .compile()
    ).with_config(run_name="Judge Agent")

def code_reflection_node_updated(state: AgentBuilderState):
    """
    LangGraph node to run code reflection and fixing using E2B sandbox - updated for AgentBuilderState
    """
    python_code = state['python_code']
    sandbox = Sandbox("OpenEvalsPython", envs={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]})

    judge = create_judge_graph(sandbox)
    graph = create_base_agent()
    reflection_agent = create_reflection_graph(graph, judge, CodeState).compile().with_config(run_name="Mini Chat LangChain")

    # Run the reflection agent with the generated code
    result = reflection_agent.invoke({
        "messages": [HumanMessage(content=f"{python_code}\n")]
    })
    main_logs = result["main_logs"]
    graph_schema = result["graph_schema"]
    # Extract the final message content as the fixed code
    final_message = result["messages"][-1]
    fixed_code = final_message.content if hasattr(final_message, 'content') else str(final_message)

    return {
        "graph_schema": graph_schema,
        "main_logs": main_logs,
        "messages": [AIMessage(content="Code has been tested and fixed using E2B sandbox reflection.")],
        "python_code": fixed_code
    }
