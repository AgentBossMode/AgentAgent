edge_checklist = """
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
"""

edge_fix_example ="""
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
"""

edge_info= """
<Edges>
Edges define how the logic is routed and how the graph decides to stop. This is a big part of how your agents work and how different nodes communicate with each other. There are a few key types of edges:

Normal Edges: Go directly from one node to the next.
Conditional Edges: Call a function to determine which node(s) to go to next.
A node can have MULTIPLE outgoing edges. If a node has multiple out-going edges, all of those destination nodes will be executed in parallel as a part of the next superstep.

<NonConditionalEdges>
If you always want to go from node A to node B, you can use the add_edge method directly.

``` python
graph.add_edge("node_a", "node_b")
```
#### The following must be ensured without fail when defining a Non-Conditional Edges
- [ ] **Valid Connections**: Source and target nodes exist in the graph
- [ ] **Flow Logic**: Edge connections support the intended workflow
- [ ] **No Orphaned Nodes**: All nodes are reachable through edge connections

</NonConditionalEdges>

<ConditionalEdges>
If you want to optionally route to 1 or more edges (or optionally terminate), you can use the add_conditional_edges method. This method accepts the name of a node and a "routing function" to call after that node is executed:

``` python
graph.add_conditional_edges("node_a", routing_function)
```

Similar to nodes, the routing_function accepts the current state of the graph and returns a value.

By default, the return value routing_function is used as the name of the node (or list of nodes) to send the state to next. All those nodes will be run in parallel as a part of the next superstep.

You can optionally provide a dictionary that maps the routing_function's output to the name of the next node.

``` python 
graph.add_conditional_edges("node_a", routing_function, {{True: "node_b", False: "node_c"}})
```
#### The following must be ensured without fail when defining a Conditional Edges
- [ ] **Condition Function**: Properly defined condition functions that return valid next node names
- [ ] **Edge Mapping**: Correct mapping between condition outcomes and target nodes
- [ ] **Default Paths**: Appropriate default/fallback paths defined
- [ ] **State Access**: Condition functions correctly access required state properties

</ConditionalEdges>
</Edges>

<Command>
It can be useful to combine control flow (edges) and state updates (nodes). 
For example, you might want to BOTH perform state updates AND decide which node to go to next in the SAME node. 
LangGraph provides a way to do so by returning a Command object from node functions:

``` python
def my_node(state: State) -> Command[Literal["my_other_node"]]:
    return Command(
        # state update
        update={{"foo": "bar"}},
        # control flow
        goto="my_other_node"
    )
``` 

With Command you can also achieve dynamic control flow behavior (identical to conditional edges):

``` python
def my_node(state: State) -> Command[Literal["my_other_node"]]:
    if state["foo"] == "bar":
        return Command(update={{"foo": "baz"}}, goto="my_other_node")
```
                                               
Important:
When returning Command in your node functions, you must add return type annotations with the list of node names the node is routing to, e.g. Command[Literal["my_other_node"]]. This is necessary for the graph rendering and tells LangGraph that my_node can navigate to my_other_node.

Navigating to a node in a parent graph:
If you are using subgraphs, you might want to navigate from a node within a subgraph to a different subgraph (i.e. a different node in the parent graph). To do so, you can specify graph=Command.PARENT in Command:

``` python
def my_node(state: State) -> Command[Literal["other_subgraph"]]:
    return Command(
        update={{"foo": "bar"}},
        goto="other_subgraph",  # where `other_subgraph` is a node in the parent graph
        graph=Command.PARENT
    )
```
                                          
Note:
Setting graph to Command.PARENT will navigate to the closest parent graph.
State updates with Command.PARENT
When you send updates from a subgraph node to a parent graph node for a key that's shared by both parent and subgraph state schemas, you must define a reducer for the key you're updating in the parent graph state. See this example.

This is particularly useful when implementing multi-agent handoffs.
</Command>

<CommandOrConditionalEdge>
Use Command 
1. when you need to both update the graph state and route to a different node. 
2. For example, when implementing multi-agent handoffs where it's important to route to a different agent and pass some information to that agent.
Use conditional edges to route between nodes conditionally without updating the state.
</CommandOrConditionalEdge>

Output: python code with appropriate inline comments
Follow the below algorithm to generate output: 
if: non-conditional edge, then: refer to implementation in 'NonConditionalEdge' for implementation
else if: either the return type of the function is Command or according to 'CommandOrConditionalEdge' we should use Command functionality, then: refer to 'Command' section for implementation
else if : according to 'CommandOrConditionalEdge' conditional_edge should be used, then: refer to 'ConditionalEdges' section for implementation.

Ensure that all the generated nodes follow the checklist below:
- [ ] **Node Dependencies**: Proper sequencing of nodes based on data dependencies
- [ ] **Error Recovery**: Graceful handling of node failures
- [ ] **Memory Management**: Efficient state updates without unnecessary data retention
"""