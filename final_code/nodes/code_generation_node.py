from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.llms.model_factory import get_model
from final_code.utils.MockJsonSchema import json_schema_str
from copilotkit.langgraph import copilotkit_customize_config
from langchain_core.runnables import RunnableConfig
import uuid
from langchain_core.tools import tool


llm = get_model()

CODE_GEN_PROMPT = PromptTemplate.from_template("""
You are an expert Python programmer specializing in AI agent development via the Langgraph and Langchain SDK. Your primary task is to generate compilable, logical, and complete Python code for a LangGraph state graph based on user 'JSON' section below. You must prioritize LLM-based implementations for relevant tasks.

<JSON>                                              
{json_schema}
</JSON>
<TOOL_BINDING_INSTRUCTIONS>
1. From the toolset field in the json schema identify the tool schema which might look like the following:
Example 1 (composio tool)
{{
          "name": "CRM_Tool",
          "description": "Tool to interact with CRM systems to update lead information, log activities, and retrieve lead statuses.",
          "is_composio_tool": true,
          "composio_toolkit_name": "HubSpot",
          "composio_tool_name": "DO_ABC_ACTIVITY",
          "py_code": null,
          "node_ids": [
            "node_a",
            "node_c"
          ]
}}

Example 2 (non-comosio tool)
{{
          "name": "search_customer_database",
          "description": "Tool to search customer database",
          "is_composio_tool": false,
          "composio_toolkit_name": "None",
          "composio_tool_name": "None",
          "py_code": "The python code to implement this tool ....",
          "node_ids": [
            "node_a",
            "node_b"
          ]
}}
                                               
2. IF is_composio_tool is true (Example 1), THEN: 
```python
from composio import Composio
from composio_langchain import LangchainProvider
composio = Composio(provider=LangchainProvider())
tools = composio.tools.get(user_id=os.environ(\"USER_ID\"), tools=[\"composio_tool_name\"])  # Replace with actual tool name (in this example DO_ABC_ACTIVITY)

```                                               
3. ELSE IF the tools corresponding to a node are non-composio tools  (Example 2) use the py_code field in the json schema:
``` python
                                               
from langchain_core.tools import tool

#py_code goes here. For example:
@tool
def search_customer_database(customer_id: str) -> str:
    '''Search for customer information by ID.'''
    # Direct implementation - no nested LLM calls
    return f"Customer {{customer_id}} data retrieved"
```
4. Follow the above for all the different tools in the Tool List.
</TOOL_BINDING_INSTRUCTIONS>

<NODE_IMPLEMENTATION_INSTRUCTIONS>

<COMMONINSTRUCTION>
1. Always use MessagesState as base and extend as needed, refer to the input_schema attached to a node definition.
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

2. For each node, include reasoning comments:
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
**Important:** Every node's return dictionary **must** include a \"messages\" key, even if it just contains a system message for status.
</COMMONINSTRUCTION>
                                            
## Pattern 1: Tool-calling react agent
1. if the Id of the node is linked to any of the tool in the tools list, you will follow the below format:
```python
node_name_tools = [list_of_tools]
from langchain_openai import ChatOpenAI
def node_name(state: GraphState) -> GraphState:
    '''Node purpose: [Clear description]'''
    agent = create_react_agent(
        model=ChatOpenAI(model="gpt-4o", temperature=0.7),
        tools=node_name_tools,
        prompt="The prompt for the agent to follow, also mention which tools to use, if any.")
    response = agent.invoke([HumanMessage(content="Perform action based on state")])
    return {{
        "messages": [response["messages"]]
    }}
            
```
## Pattern 2: LLM with Structured Output
**When to use:** Node needs to make decisions, classify inputs, or extract structured data.

**Example Implementation:**
<Example1>
```python
from pydantic import BaseModel, Field
from typing import Literal

class IntentClassification(BaseModel):
    '''Structured output for intent classification.'''
    intent: Literal["support", "sales", "billing"] = Field(description="Classified intent")
    confidence: float = Field(description="Confidence score 0-1")
    reasoning: str = Field(description="Brief explanation of classification")

def intent_classifier_node(state: GraphState) -> GraphState:
    # Reasoning: This node needs structured decision making for routing
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = llm.with_structured_output(IntentClassification)
    
    user_message = state["messages"][-1].content
    prompt = f"Classify this user message: {{user_message}}"
    
    result = structured_llm.invoke(prompt)
    return {{
        "messages": [("system", f"Intent classified as: {{result.intent}}")],
        "intent": result.intent,
        "confidence": result.confidence
    }}
```

</Example1>
<Example2>
Here is an example of how to use structured output. In this example, we want the LLM to generate and fill up the pydantic class Joke, based on user query. 
``` python
from typing import Optional
from pydantic import BaseModel, Field

# Pydantic class for structured output
class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )

class JokeBuilderState(MessagesState):
    joke: Joke = Field(description= "joke generated by the GenerateJoke node.")

def GenerateJoke(state: JokeBuilderState):
    structured_llm = llm.with_structured_output(Joke)
    joke: Joke = structured_llm.invoke("Tell me a joke about cats")
    return {{ "joke": joke }}
```
</Example2>

## Pattern 3: Human-in-the-Loop with Interrupt
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
        "some_text": value,
        "messages": [("system", "Human intervention occurred.")] # Example message
    }}
```

## Pattern 4: Multi-Step LLM Processing
**When to use:** Complex tasks requiring multiple LLM operations.

**Example Implementation:**
```python
def content_enhancement_node(state: GraphState) -> dict:
    # Reasoning: Multi-step enhancement requires sequential LLM processing
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    raw_content = state.get("raw_content", "")
    
    # Step 1: Structure the content
    structured_prompt = f"Structure this content logically: {{raw_content}}"
    structured = llm.invoke(structured_prompt).content
    
    # Step 2: Enhance with examples
    enhanced_prompt = f"Add relevant examples to: {{structured}}"
    enhanced = llm.invoke(enhanced_prompt).content
    
    return {{
        "messages": [("system", "Content enhanced with structure and examples")],
        "enhanced_content": enhanced,
        "processing_steps": ["structured", "enhanced"]
    }}
```
</NODE_IMPLEMENTATION_INSTRUCTIONS>

<EDGE_IMPLEMENTATION_INSTRUCTIONS>
                                               
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

</EDGE_IMPLEMENTATION_INSTRUCTIONS>

                                                                                       
<INSTRUCTIONS>
1. First create the tools, refer to <TOOLBINDINGINSTRUCTIONS> section.
2. Now start analyzing the nodes, refer to <NODE_IMPLEMENTATION_INSTRUCTIONS>
   - For the first LLM call or first node, ensure the LLM's input collects information contains `state["messages"]` to incorporate the conversation history.
3. Now create the edges, refer to the <EDGE_IMPLEMENTATION_INSTRUCTIONS> section.
4. Now to piece it all together follow <CODE_GENERATION_INSTRUCTIONS>
</INSTRUCTIONS>


<CODE_GENERATION_INSTRUCTIONS>

Generate a single, self-contained, and compilable Python script following this structure:

### 1. Imports and Setup
Below are some common langggraph imports, you need note necessarily add them, refer to them when writing the code.
```python
from typing import Dict, Any, List, Optional, Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver, InMemorySaver
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import re
import json
```



###2: Final Graph Compilation
```python
checkpointer = InMemorySaver()
app = workflow.compile(
    checkpointer=checkpointer
)
                        

DONOT ADD '__main__' block or any other boilerplate code, the code should be self-contained and compilable.
```
<QUALITY_CHECKLIST>
Before finalizing your code, verify:
- [ ] All imports are included and correct, no duplicate imports.
- [ ] GraphState properly extends MessagesState 
- [ ] LLM calls include proper error handling
- [ ] Tools are self-contained (no nested LLM calls)
- [ ] Structured output uses proper Pydantic models
- [ ] Conditional edges handle all possible routing outcomes
- [ ] Code is compilable and logically consistent
- [ ] Ensure that the code does not access graphstate like an object attribute, it needs be accessed like a dict
- [ ] Assume any API keys(e.g., OPENAI_API_KEY, GOOGLE_API_KEY) are part of the environment variables and all environment variables are to be defined using the os.environs notation
- [ ] **Every node's return dictionary includes a "messages" key.**
- [ ] **The first LLM call/node appropriately utilizes `state["messages"]` as part of its input.**
</QUALITY_CHECKLIST>

<KEY_EXTRACTION_INSTRUCTIONS>
After generating the complete Python script, add a section titled:

## Required Keys and Credentials

List all environment variables, API keys, and external dependencies needed as comment :
- Environment variables (e.g., OPENAI_API_KEY)
- Tool-specific credentials 
- External service configurations
- Database connection strings (if applicable)

If no external keys are needed, state: "No external API keys required for this implementation."
</KEY_EXTRACTION_INSTRUCTIONS>
</CODE_GENERATION_INSTRUCTIONS>



Please return only complete and compilable langgraph python code
""")

@tool
def write_code(code: str): # pylint: disable=invalid-name,unused-argument
    """Writes the code to a file.

      Args:
        code (str): The code to write.
      """
    

def code_node(state: AgentBuilderState, config: RunnableConfig):
    """
    LangGraph node to generate the final Python code for the agent.
    It uses the gathered agent_instructions and the CODE_GEN_PROMPT.
    """

    modifiedConfig = copilotkit_customize_config(
        config,
        emit_intermediate_state=[{
            "state_key": "python_code",
            "tool": "write_code",
            "tool_argument": "code",
        }],
    )

    code_llm_writer = llm.bind_tools([write_code])
    json_schema_final = state["json_schema"].model_dump_json(indent=2)
    #json_schema_final = json_schema_str
    response = code_llm_writer.invoke([SystemMessage(content="Call the 'write_code' tool."), HumanMessage(content=CODE_GEN_PROMPT.format(json_schema=json_schema_final))], config=modifiedConfig)
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_call = response.tool_calls[0]
        # Handle tool_call as a dictionary or an object
        if isinstance(tool_call, dict):
            tool_call_id = tool_call["id"]
            tool_call_args = tool_call["args"]
            tool_call_name= tool_call["name"]
        else:
            # Handle as an object (backward compatibility)
            tool_call_id = tool_call.id
            tool_call_args = tool_call.args
            tool_call_name= tool_call.name
    # Return the generated Python code and an AI message
    return {
        "python_code": tool_call_args["code"],
    }
