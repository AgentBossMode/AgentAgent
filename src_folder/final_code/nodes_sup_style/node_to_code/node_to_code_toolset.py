from langchain_core.messages import HumanMessage
from langgraph.types import Command
from typing import  Literal
from langchain_core.messages import SystemMessage, HumanMessage
from final_code.llms.model_factory import get_model
from final_code.nodes_sup_style.node_to_code.node_to_code_base import NodeBuilderState


llm = get_model()

toolset_retrieval_prompt = """
    User will provide information about a node, Your task is to see if meeting the requirements of the node needs llm(aka model)'s tool-binding capability.
    
    Information about tool calling capabilities of LLMs, how to create tools and how to bind them with a LLM is provided below:

<NEED_FOR_TOOL_BINDING>
a. if the node's function is to answer a user_query which needs factual information to answer
b. personal assistant which may need to respond with which tool should be used if any to meet the requirements 
c. Whenever the node involves a LLM(model) needing sensors or actuators, for information retrieval, and allowing llm(model) to 'DO' stuff
d. One of the best use cases of this technique is when you know that there would be 'n' different tools which could be used in different scenarios to handle a query in a particular domain.
</NEED_FOR_TOOL_BINDING>

<TOOL_BINDING_EXAMPLE>
Example 1: 
``` python
from typing import Literal
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

# Define the tools needed by the LLM

@tool
def get_weather(location: str):
    \"\"\"Call to get the current weather.\"\"\"
    if location.lower() in ["sf", "san francisco"]:
        return "It's 60 degrees and foggy."
    else:
        return "It's 90 degrees and sunny."


@tool
def get_coolest_cities():
    \"\"\"Get a list of coolest cities\"\"\"
    return "nyc, sf"

tools = [get_weather, get_coolest_cities]

# Bind the model(llm) with tools
model_with_tools = ChatAnthropic(
    model="claude-3-haiku-20240307", temperature=0
).bind_tools(tools)

# Generate a tool node.
tool_node = ToolNode(tools)

# conditional edge
def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

# node implementation
def call_model(state: MessagesState):
    \"\"\"This node answers questions related to weather using a varied weather related toolset\"\"\"
    messages = state["messages"]
    response = model_with_tools.invoke([SystemMessage(content= "Please analyze the following weather-related query and provide a detailed response with relevant information: You have tools like get_weather and get_coolest_cities to help you answer the queries" )] + messages)
    return {"messages": [response]}


# Stategraph compilation
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")
app = workflow.compile()
```
</TOOL_BINDING_EXAMPLE>

<Output> 
Identify if tool_binding is relevant based on information 'NEED_FOR_TOOL_BINDING'
if yes: identify what kind of tools may be needed to be used by a LLM, and then write code for the llm binding functionality, use 'TOOL_BINDING_EXAMPLE'
if no: just respond no tool_binding needed.

Only write python code as output
</Output>
    """

def toolset_generation(state: NodeBuilderState) -> Command[Literal["ai_node_gen_supervisor"]]:
    """Generate the code for the node."""
    result = llm.invoke([SystemMessage(content=toolset_retrieval_prompt)] + state["messages"])
          
    return Command(
        update={
            "past_steps": [(state["task"], result.content)],
            "messages": [
                HumanMessage(content=result.content, name="toolset_generator")
            ]
        },
        goto="replan",
    )