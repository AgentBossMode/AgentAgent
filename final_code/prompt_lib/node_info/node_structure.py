node_structure = """
For each node, include reasoning comments:
```python
def node_name(state: GraphState) -> GraphState:
    \"\"\"
    Node purpose: [Clear description]
    Implementation reasoning: [Why this pattern was chosen]
    \"\"\"
    # Implementation here
    return {{"field": "value",
        "messages": "value_message"
    }}
```
**Important:** Every node's return dictionary **must** include a \"messages\" key, even if it just contains a AiMessage for status.
"""

function_structure= """
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
    return {{"output": result, "status": "processed","messages": "value_message"
    }}
```
"""