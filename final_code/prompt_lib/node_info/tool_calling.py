tool_calling = """
Follow this example:
```python
node_name_tools = [list_of_tools]
def node_name(state: GraphState) -> GraphState:
    # define a state which inherits from MessagesState, also always contains remaining_step  and structured_response
    class CustomStateForReact(MessagesState):
        remaining_steps: int
        structured_response: any
    # define what kind of result you need from the agent.
    class CustomClass(BaseModel):
        attr1: type = Field(description="What is the field")
        attr2: type = Field(description="What is the field")

    agent = create_react_agent(
      model=llm,
      prompt="The prompt for the agent to follow, also mention which tools to use, if any.",
      tools=node_name_tools,
      state_schema=CustomStateForReact,
      response_format=CustomClass)

    result: CustomClass = agent.invoke(state["messages"]) #or whatever content you wish to put as per the state.
    # Logic that either updates the state variable with result.attr1/result.attr2
    # DO NOT do string parsing or regex parsing
```

Please ensure that the code produced for a tool node follows:
1. **Tool Registration**: Tools are properly defined and registered in the node
2. **Schema Adherence**: Tool inputs/outputs match their defined schemas exactly
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

tool_node_pattern = """
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
"""