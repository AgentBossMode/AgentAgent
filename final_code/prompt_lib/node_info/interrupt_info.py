interrupt_info ="""
**When to use Human-in-the-Loop (HIL) with 'interrupt':** Node requires human approval, creative input, or oversight before proceeding. This is crucial for approving critical steps (e.g., API calls), correcting mistakes, updating state with additional information, or reviewing sensitive tool calls.

***Example Implementation:***
```python
from langgraph.types import interrupt
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState
import operator

# Define your graph's state
class AgentState(MessageState):
    some_text: str
    decision: str # To store human's decision

# A basic example node demonstrating interrupt
def human_node(state: AgentState):
    '''
    This node interrupts the graph and surfaces 'text_to_revise' to the human.
    Upon resumption, 'value' will contain the input provided by the human.
    '''
    value = interrupt(
        # Any JSON serializable value to surface to the human.
        # This is what the human sees and can act upon.
        {{
            "prompt_for_human": "Please review and revise the following text:",
            "text_to_revise": state["some_text"]
        }}
    )
    # 'value' here is the data sent by the human to resume the graph.
    # It could be a string, a dictionary, etc., depending on what you expect.
    print(f"--- Human Node: Resumed with human input: {value} ---")
    return {{
        "some_text": value, # Update the state with the human's revised text
        "messages": [AIMessage(content="Human intervention occurred. Text revised.")],
    }}
```

## Interrupt Design Patterns

### 1. Approve/Reject Pattern
Use when decisions need human approval before proceeding.

```python
from typing import Literal
from langgraph.types import interrupt, Command

def human_approval_node(state: State) -> Command[Literal["approved_path", "rejected_path"]]:
    decision = interrupt({
        "question": "Do you approve this action?",
        "action_details": state["proposed_action"],
        "risk_level": "high"
    })
    
    if decision == "approve":
        return Command(goto="approved_path", update={"status": "approved"})
    else:
        return Command(goto="rejected_path", update={"status": "rejected"})
```

### 2. Edit/Review Pattern
Use when content needs human review and editing.

```python
def human_review_node(state: State):
    result = interrupt({
        "task": "Review and edit the generated content",
        "generated_content": state["draft_content"],
        "guidelines": "Check for accuracy, tone, and completeness"
    })
    
    return {
        "final_content": result["edited_content"],
        "review_notes": result.get("notes", ""),
        "messages": [AIMessage(content="Content reviewed and updated")]
    }
```

### 3. Input Collection Pattern
Use when additional information is needed from human.

```python
def collect_input_node(state: State):
    user_input = interrupt({
        "prompt": "What additional context should I consider?",
        "current_context": state["context"],
        "input_type": "text"
    })
    
    return {
        "context": state["context"] + f"\nUser input: {user_input}",
        "messages": [AIMessage(content="Additional context collected")]
    }
```

### Placement and Handling Returned Values

**Placement:** For optimal behavior, `interrupt()` is best placed at the beginning of a node or within a dedicated human intervention node. This is because the entire node where `interrupt()` was called will be re-executed upon resumption.

**Handling the `value` returned by `interrupt()`:**

* When `interrupt()` is called, it **pauses** execution. It doesn't immediately return a value.
* When the graph is **resumed** by the human (using `app.invoke(Command(resume=human_input_value), ...)`, `human_input_value` is the data provided by the human.
* Upon resumption, LangGraph **re-executes the node from the beginning where `interrupt()` was called**. This time, the `interrupt()` function **returns** the `human_input_value` that was passed in the `Command(resume=...)`.
* Therefore, the `value` variable in the `human_node` example above will contain whatever the human provided to resume the graph. You should design your human interaction and the expected `human_input_value` format (e.g., a simple string, a JSON object, etc.) to match what your node expects to process.
"""