from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.retriever import create_retriever_tool
from final_code.llms.model_factory import get_model
from final_code.utils.dependency_setup import get_embeddings, get_vector_store

embeddings = get_embeddings()
vector_store = get_vector_store(embeddings, "langstuffindex")
retriever = vector_store.as_retriever()
langgraph_glossary_retriever = create_retriever_tool(retriever=retriever, name="langgraph_glossary_retriever", description= "Contains all information about langgraph and langchain, use to clarify doubts or retrieve similar examples")

llm = get_model()


prompt = ChatPromptTemplate.from_messages([
("system", """You are a ReAct (Reasoning and Act) agent, expert at writing python code for implementing langgraph routing.
You are tasked with writing python code snippet that implements edges/conditional-edges/Command given the description of the edge, the code for the graph node,the src and potential targets.

Information about how to implement edges/conditional-edges is below:
<Information>

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
Use Command when you need to both update the graph state and route to a different node. For example, when implementing multi-agent handoffs where it's important to route to a different agent and pass some information to that agent.
Use conditional edges to route between nodes conditionally without updating the state.
</CommandOrConditionalEdge>

Apart from langgraph edge/conditional_edge/command creation information in the 'Edges' section, call the 'langgraph_glossary_retriever' tool to find for similar examples to the description of the input
Output: python code with appropriate inline comments
Follow the below algorithm to generate output: 
if: non-conditional edge, then: refer to implementation in 'NonConditionalEdge' for implementation
else if: either the return type of the function is Command or according to 'CommandOrConditionalEdge' we should use Command functionality, then: refer to 'Command' section for implementation
else if : according to 'CommandOrConditionalEdge' conditional_edge should be used, then: refer to 'ConditionalEdges' section for implementation.
"""),
("placeholder", "{messages}")])

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
edge_builder_agent = create_react_agent(model = llm, prompt= prompt, tools = [langgraph_glossary_retriever], checkpointer=checkpointer)