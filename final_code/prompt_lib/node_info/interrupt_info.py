interrupt_info ="""
**When to use:** Node requires human approval, creative input, or oversight.

*** Example Implementation:***
```python
from langgraph.types import interrupt

def human_node(state: State):
    value = interrupt(
        # Any JSON serializable value to surface to the human.
        # For example, a question or a piece of text or a set of keys in the state
       {{
          "text_to_revise": state["some_text"]
       }}
    )
    # Update the state with the human's input or route the graph based on the input.
    return {{
        "some_text": value
    }}
```
"""