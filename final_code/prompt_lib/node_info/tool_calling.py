tool_calling = """
```python
node_name_tools = [list_of_tools]
def node_name(state: GraphState) -> GraphState:
    '''Node purpose: [Clear description]'''
    agent = create_react_agent(
        model=llm,
        tools=node_name_tools,
        prompt="The prompt for the agent to follow, also mention which tools to use, if any.")
    response = agent.invoke([HumanMessage(content="Perform action based on state")])
    return {{
        "messages": [response["messages"]]
    }}

## Implementation Patterns

### Pattern 1: Manual Tool Binding with ToolNode. To be used when not using create_react_agent in a node and we want tool calling capabilities
```python
# ❌ Incorrect - improper tool definition
def analysis_node(state):
    tool_call = some_tool(state["data"])  # Direct tool call without binding
    
# ✅ Correct - proper tool binding and invocation using toolNode
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

@tool
def some_tool(x: str) -> str:
    '''doc string for the tool'''
    ## this part will have the tool implementation

def analysis_node(state: MessagesState):
    model_with_tools = model.bind_tools([some_tool])
    result = model_with_tools.invoke(state["messages"])
    return {{"messages": [result]}}

workflow = StateGraph(MessagesState)
workflow.add_node("analysis_node", analysis_node)
workflow.add_node("tools", ToolNode([some_tool]))

workflow.add_edge(START, "analysis_node")
workflow.add_conditional_edges("analysis_node", tools_condition, ["tools", END])
workflow.add_edge("tools", "analysis_node")
graph = workflow.compile()
```

### Pattern 2: Using create_react_agent (Simplified Approach)
```python
# ✅ Correct - using create_react_agent for automatic tool handling
from langgraph.prebuilt import create_react_agent

# Create agent with tools - handles tool calling automatically
agent = create_react_agent(
    model=your_model,
    tools=[your_tool1, your_tool2],
    state_modifier="Your agent instructions here"
)

# Use the agent directly - no need for manual ToolNode
def agent_node(state):
    response = agent.invoke({{"messages": state["messages"]}})
    return {{"messages": response["messages"]}}
```

Use ToolNode implementation approach when:
- You want to execute a specific sequence of tools in a controlled manner. 
- You need more control over how tools are invoked and their results are handled. 

Use create_react_agent when: 
- You need a flexible agent that can reason about the best course of action. 
- The task requires the LLM to make decisions based on its reasoning and understanding of the situation
"""

tool_calling_checklist = """
- [ ] **Tool Registration**: Tools are properly defined and registered in the node
- [ ] **Schema Adherence**: Tool inputs/outputs match their defined schemas exactly
- [ ] **Tool Invocation**: Tools are called using correct LangGraph patterns (`model.bind_tools()`, `ToolNode`)
- [ ] **Structured Output**: Uses `model.with_structured_output()` when structured responses are needed
- [ ] **Error Handling**: Proper exception handling for tool execution failures

**Example Fix:**
```python
# ❌ Incorrect - improper tool definition
def analysis_node(state):
    tool_call = some_tool(state["data"])  # Direct tool call without binding
    
# ✅ Correct - proper tool binding and invocation
from langchain.tools import Tool
from langgraph.prebuilt import ToolNode

analysis_tool = Tool(
    name="analyze_data",
    description="Analyzes the provided data",
    func=analyze_function
)

def analysis_node(state):
    model_with_tools = model.bind_tools([analysis_tool])
    result = model_with_tools.invoke(state["messages"])
    return {{"messages": [result]}}
```
"""