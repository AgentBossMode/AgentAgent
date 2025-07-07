from langchain_core.messages import AIMessage, HumanMessage
from final_code.llms.model_factory import get_model
from final_code.states.AgentBuilderState import AgentBuilderState
from pydantic import BaseModel, Field
from copilotkit.langgraph import copilotkit_customize_config
from langchain_core.runnables import RunnableConfig
llm = get_model()


ANALYSIS_COMPILE_PROMPT = """
You are an expert LangGraph code refactoring AI. Your task is to analyze the provided Python code for a LangGraph implementation and automatically correct it using comprehensive checklists to ensure adherence to best practices.

## Analysis Checklists

### Node-Level Checklist

For each node in the graph, verify and correct:

#### 1. State Management
- [ ] **State Input Consistency**: Node correctly accesses state properties according to the state schema
- [ ] **State Output Updates**: Node properly updates state with correct data types and structure
- [ ] **Type Safety**: All state reads/writes maintain consistent data types
- [ ] **Required Fields**: Node handles missing or optional state fields appropriately

**Example Fix:**
```python
# ❌ Incorrect - accessing undefined state property
def research_node(state):
    query = state["search_query"]  # Property doesn't exist in state schema
    
# ✅ Correct - accessing properly defined state property
def research_node(state):
    query = state["query"]  # Matches state schema definition
```

#### 2. Tool Definition and Usage
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

#### 3. Function Structure
- [ ] **Return Format**: Node returns dictionary with proper state updates
- [ ] **Function Signature**: Accepts state parameter correctly
- [ ] **Async Handling**: Proper async/await usage if applicable
- [ ] **Documentation**: Clear docstrings explaining node purpose

**Example Fix:**
```python
# ❌ Incorrect - returning wrong format
def process_node(state):
    result = process_data(state["input"])
    return result  # Should return dict for state update

# ✅ Correct - proper state update return
def process_node(state):
    "Process input data and update state."
    result = process_data(state["input"])
    return {{"output": result, "status": "processed"}}
```

### Edge-Level Checklist

For each edge in the graph, verify and correct:

#### 1. Conditional Edges
- [ ] **Condition Function**: Properly defined condition functions that return valid next node names
- [ ] **Edge Mapping**: Correct mapping between condition outcomes and target nodes
- [ ] **Default Paths**: Appropriate default/fallback paths defined
- [ ] **State Access**: Condition functions correctly access required state properties

**Example Fix:**
```python
# ❌ Incorrect - condition function returns invalid node name
def should_continue(state):
    if state["count"] > 5:
        return "invalid_node"  # Node doesn't exist in graph
    return "continue"

# ✅ Correct - returns valid node names
def should_continue(state):
    if state["count"] > 5:
        return "end_node"  # Valid node in graph
    return "process_node"  # Valid node in graph

# Proper edge definition
graph.add_conditional_edges(
    "start_node",
    should_continue,
    {{
        "end_node": "end_node",
        "process_node": "process_node"
    }}
)
```

#### 2. Standard Edges
- [ ] **Valid Connections**: Source and target nodes exist in the graph
- [ ] **Flow Logic**: Edge connections support the intended workflow
- [ ] **No Orphaned Nodes**: All nodes are reachable through edge connections

**Example Fix:**
```python
# ❌ Incorrect - connecting to non-existent node
graph.add_edge("process_node", "nonexistent_node")

# ✅ Correct - connecting to valid nodes
graph.add_edge("process_node", "output_node")
```

### Graph-Level Checklist

For the overall graph structure, verify and correct:

#### 1. Graph Configuration
- [ ] **State Schema**: Well-defined state schema with appropriate types
- [ ] **Entry Point**: Proper graph entry point definition
- [ ] **Exit Conditions**: Clear termination conditions and END nodes
- [ ] **Compilation**: Graph compiles without errors

**Example Fix:**
```python
# ❌ Incorrect - poorly defined state schema
class GraphState(MessageState):
    data: any  # Vague type definition

# ✅ Correct - well-defined state schema
from typing import List, Optional
rom langgraph.graph import MessagesState

class GraphState(MessageState):
    messages: List[dict]
    query: str
    results: Optional[List[dict]] = None
    status: str = "initialized"
```

#### 2. Workflow Logic
- [ ] **Node Dependencies**: Proper sequencing of nodes based on data dependencies
- [ ] **Cycle Prevention**: No infinite loops in graph execution
- [ ] **Error Recovery**: Graceful handling of node failures
- [ ] **Memory Management**: Efficient state updates without unnecessary data retention

**Example Fix:**
```python
# ❌ Incorrect - potential infinite loop
graph.add_conditional_edges(
    "node_a",
    lambda state: "node_a",  # Always returns to self
    {{"node_a": "node_a"}}
)

# ✅ Correct - proper termination condition
graph.add_conditional_edges(
    "node_a",
    lambda state: "node_b" if state["processed"] else "node_a",
    {{
        "node_a": "node_a",
        "node_b": "node_b"
    }}
)
```

#### 3. Best Practices
- [ ] **Modularity**: Nodes are focused and handle single responsibilities
- [ ] **Reusability**: Common functionality is properly abstracted
- [ ] **Testing**: Code structure supports unit testing
- [ ] **Performance**: Efficient state management and tool usage

## Analysis Process

1. **Parse the provided code** and identify all nodes, edges, and state definitions
2. **Apply node-level checklist** to each node, noting violations and corrections needed
3. **Apply edge-level checklist** to all graph connections
4. **Apply graph-level checklist** to overall structure
5. **Generate corrected code** with all identified issues resolved
6. **Add explanatory comments** for significant changes made

## Input Format
```python
<input_code>
{compiled_code}
</input_code>
```

## Output Format

Provide the corrected Python code with:
- All checklist violations fixed
- Explanatory comments for major changes
- Proper imports and dependencies
- Clean, readable structure following LangGraph best practices

### Example Analysis and Correction

**Original Code Issues:**
- Node accessing undefined state property
- Tool not properly bound to model
- Conditional edge returning invalid node name
- Missing error handling

**Corrected Code:**
```python
# Fixed: Proper imports and state definition
from langchain_core.messages import HumanMessage
from langgraph import StateGraph
from langgraph.prebuilt import ToolNode
rom langgraph.graph import MessagesState
from typing import List, Optional

class GraphState(MessageState):
    messages: List[dict]
    query: str
    results: Optional[List[dict]] = None

# Fixed: Proper tool binding and state access
def research_node(state: GraphState):
    "Research node with proper tool usage."
    try:
        # Fixed: Accessing correct state property
        query = state.query  # Was: state["search_query"]
        
        # Fixed: Proper tool binding
        model_with_tools = model.bind_tools([search_tool])
        result = model_with_tools.invoke([HumanMessage(content=query)])
        
        # Fixed: Proper state update
        return {{'results': [result.content]}}
    except Exception as e:
        # Fixed: Added error handling
        return {{'results': [], "error": str(e)}}

# Fixed: Proper condition function
def should_continue(state: GraphState):
    "Determine next node based on results."
    # Fixed: Returns valid node names only
    return "end" if state.results else "research_node"

# Fixed: Proper graph construction
workflow = StateGraph(GraphState)
workflow.add_node("research_node", research_node)
workflow.add_conditional_edges(
    "research_node",
    should_continue,
    {{
        "end": END,  # Fixed: Valid termination
        "research_node": "research_node"
    }}
)
workflow.set_entry_point("research_node")
graph = workflow.compile()
```

Now analyze the provided code and apply all necessary corrections based on these checklists.

Output:
Provide only the updated and corrected LangGraph Python code in a single block. Do not include explanations outside of the code's comments.
"""



class DFSAnalysis(BaseModel):
    correct_code: str = Field(description= "If there are any issues in the code, this field will contain the corrected code with the fixes applied to the original code.")
    explanation: str = Field(description="Explanation of the changes made to the code, if any.")


def dfs_analysis_node(state: AgentBuilderState, config: RunnableConfig): # Renamed for clarity
     """
     LangGraph node to analyse the code
     """
     modifiedConfig = copilotkit_customize_config(
        config,
        emit_messages=False, # if you want to disable message streaming 
        emit_tool_calls=False # if you want to disable tool call streaming 
    )
     main_agent_code = state['python_code']
     llm_dfs = llm.with_structured_output(DFSAnalysis)

     # Use LLM to merge the main agent code with the generated tool definitions
     response: DFSAnalysis = llm_dfs.invoke([HumanMessage(content=ANALYSIS_COMPILE_PROMPT.format(
         compiled_code=main_agent_code,
     ))], config =modifiedConfig)

     # The response from this LLM call is expected to be the final, complete Python code
     return {
         "messages": [AIMessage(content=response.explanation)], # Storing the LLM's final code as a message for now
         "python_code": response.correct_code # Update compiled_code with the final merged code
     }

