from langsmith import testing as t
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

@tool
def increment_by1_tool(x: str) -> str:
    """This is called when user query pertains to ading by 1"""
    try:
        x = int(x)
        return str(x + 1)
    except ValueError:
        return "Invalid input, please provide a number."

SYSTEM_PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful assistant tasked with answering user queries.

You are given a list of messages. Your job is to analyze the full message history, focusing especially on the last message in the list.

Instructions:

Determine if a tool needs to be called to answer the user's question.
Only call a tool if it is required to answer the question.
Do not call any tool more than once per question.
If a tool has already been called and provided a response, use that tool response to answer the question.
If no tool is needed, answer the user directly.
Available tools:

increment_by1_tool: Adds 1 to a number. Input and output are strings.
Be efficient and avoid redundant tool usage."""
)

def node_a(state: MessagesState):
    llm = ChatOpenAI(model= "gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools([increment_by1_tool])
    response = llm_with_tools.invoke(
    [SystemMessage(content=SYSTEM_PROMPT.format())] + state["messages"])
    return {
        "messages": [response]
    }

workflow = StateGraph(MessagesState)
workflow.add_node("node_a", node_a)
workflow.add_node("tools", ToolNode([increment_by1_tool]))

workflow.add_edge(START, "node_a")
workflow.add_conditional_edges("node_a", tools_condition, ["tools", END])
workflow.add_edge("tools", "node_a")
graph = workflow.compile()

###################### tests/test_my_app.py ######################
import pytest
from langsmith import testing as t

# adding a LLM judge on the final response
@pytest.mark.langsmith  # <-- Mark as a LangSmith test case
def test_langgraph_response() -> None:
    user_query = "How are you"
    t.log_inputs({"user_query": user_query})  # <-- Log example inputs, optional

    expected = "I'm just a computer program, so I don't have feelings, but I'm here and ready to help you! How can I assist you today?"
    t.log_reference_outputs({"sql": expected})  # <-- Log example reference outputs, optional

    response = graph.invoke({"messages": [HumanMessage(content=user_query)]})
    t.log_outputs({"response": response})  # <-- Log run outputs, optional
    
    # t.log_feedback(key="valid_sql", score=is_valid_sql(sql))  # <-- Log feedback, optional

    assert response["messages"][-1].content == expected  # <-- Test pass/fail status automatically logged to LangSmith under 'pass' feedback key

def run_graph(inputs: dict) -> dict:
        """Run graph and track the trajectory it takes along with the final response."""
        trajectory = []
        # Set subgraph=True to stream events from subgraphs of the main graph: https://langchain-ai.github.io/langgraph/how-tos/streaming-subgraphs/
        # Set stream_mode="debug" to stream all possible events: https://langchain-ai.github.io/langgraph/concepts/streaming
        for namespace, chunk in graph.stream({"messages": [
                {
                    "role": "user",
                    "content": inputs['user_query'],
                }
            ]}, subgraphs=True, stream_mode="debug"):
            # Event type for entering a node
            if chunk['type'] == 'task':
                # Record the node name
                trajectory.append(chunk['payload']['name'])
                # Given how we defined our dataset, we also need to track when specific tools are
                # called by our question answering ReACT agent. These tool calls can be found
                # when the ToolsNode (named "tools") is invoked by looking at the AIMessage.tool_calls
                # of the latest input message.
                if chunk['payload']['name'] == 'tools' and chunk['type'] == 'task':
                    for tc in chunk['payload']['input']['messages'][-1].tool_calls:
                        trajectory.append(tc['name'])

        return {"trajectory": trajectory}


@pytest.mark.langsmith  # <-- Mark as a LangSmith test case
# parametrize the test with different inputs and outputs
@pytest.mark.parametrize(
    "user_query, expected",
    [
        ("What is 1 added to 3", ["node_a", "tools", "increment_by1_tool", "node_a"]),
        ("How are you", ["node_a"]),
        ("What is your name?",["node_a"])
    ],
)
def test_graph_tract(user_query: str, expected: list) -> None:
    t.log_inputs({"user_query": user_query})
    trajectory = run_graph({"user_query": user_query})
    assert trajectory['trajectory'] == expected

@pytest.mark.langsmith
def test_searches_for_correct_ticker() -> None:
  """Test that the model looks up the correct ticker on simple query."""
  # Log the test example
  query = "What is 1 added to 3"
  t.log_inputs({"query": query})
  expected = "3"
  t.log_reference_outputs({"ticker": expected})

  # Call the agent's model node directly instead of running the full ReACT loop.
  result = graph.nodes["node_a"].invoke(
      {"messages": [{"role": "user", "content": query}]}
  )
  tool_calls = result["messages"][0].tool_calls
  if tool_calls[0]["name"] == increment_by1_tool.name:
      actual = tool_calls[0]["args"]["x"]
  else:
      actual = None
  t.log_outputs({"ticker": actual})

  # Check that the right ticker was queried
  assert actual == expected