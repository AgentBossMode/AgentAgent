graph_state = """
Always use MessagesState as base and extend as needed, refer to the input_schema attached to a node definition.
```python
class GraphState(MessagesState):
    \"\"\" The GraphState represents the state of the LangGraph workflow.
    Below is the definition of MessagesState, the AnyMessage refers to AIMessage, HumanMessage, or SystemMessage etc.
    the add_messages is a reducer, which means that when doing return {{\"messages\": [AIMessage(content=\"...\")]}}, it will append the new message to the messages variable and not override it..
    class MessagesState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
    \"\"\" 
    # Add domain-specific fields based on your analysis
    # Add other fields as required by your architecture
"""


node_state_management= """
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
"""

graph_state_checklist = """
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
    query: str
    results: Optional[List[dict]] = None
    status: str = "initialized"
```
"""