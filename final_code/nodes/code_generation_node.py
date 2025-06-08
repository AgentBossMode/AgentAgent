from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from final_code.states.AgentBuilderState import AgentBuilderState, AgentInstructions
from final_code.llms.model_factory import get_model

llm = get_model()

CODE_GEN_PROMPT = PromptTemplate.from_template("""
You are an expert Python programmer specializing in AI agent development via the Langgraph and Langchain SDK. Your primary task is to generate compilable, logical, and complete Python code for a LangGraph state graph based on user 'INPUT' section below. You must prioritize LLM-based implementations for relevant tasks.
<INPUT>
<JSON>                                               
{json_dict}
</JSON>
<Architecture>
{justification}
</Architecture>

<OBJECTIVE>
{objective}
</OBJECTIVE>
                                               
<USECASES>
{usecases}
</USECASES>
                                               
<EXAMPLES>
{examples}
</EXAMPLES>
</INPUT>
                                               
<CODE_GENERATION_INSTRUCTIONS>
Generate a single, self-contained, and compilable Python script that implements your chosen strategy.
Use the INPUT section for reference. 
1.  **Imports:** Include all necessary Python libraries (e.g., `typing`, `langgraph.graph`, `langgraph.checkpoint.memory`, LLM client libraries like `langchain_openai`, `langchain_google_genai`, `langchain_core.pydantic_v1`, `langchain_core.tools`, `re`).

2.  **State Definition (`GraphState`):**
    * Define a `GraphState` class using `MessagesState` (langgraph prebuilt class).

3.  **Node Implementation (Python Functions):**
    For each conceptual node in your chosen architecture (these may map directly to JSON you define):
    * Create a Python function. This function must accept the `GraphState` and return a dictionary representing the partial update to the state.
    * **Decision Logic for Implementation (Prioritize LLM, No Mock Data):**
        * If you think the description of the node indicates that a the node needs to be work on the basis of latest/factual information, refer to 'Tool definition and Usage'
        
        * **Default to LLM-Based Solutions:** Your default stance should be to implement an **LLM-based solution** if the node's `description` (from JSON or your architectural design) suggests tasks like:
            * Natural Language Understanding (NLU)
            * Complex classification or routing
            * Content summarization
            * Tool selection and usage
            * Planning or complex decision-making.
            * Any task where an LLM would provide more robust, flexible, or intelligent behavior than simple hardcoded logic.
        * **Algorithmic Logic (Use Sparingly):** Only use purely algorithmic Python code (like from the `code` attribute or written new) if the node's task is genuinely simple, deterministic (e.g., basic data formatting, fixed calculation), *and* an LLM would offer no significant benefit for that specific, narrow function.
        * **Functional LLM Calls:** When an LLM is used, instantiate a generic model (e.g., `llm = ChatOpenAI(model="gpt-3.5-turbo")` or `llm = ChatGoogleGenerativeAI(model="gemini-pro")`) and include a **functional, descriptive prompt** relevant to the node's task. Ensure the code for the LLM call is complete and not just a comment. Add a `TODO` comment for the user to specify API keys and potentially refine the model/prompt.
        * **No Mock Data:** Generated functions must be logical and aim for completeness. **Avoid using mock data or overly simplistic placeholder logic** where an LLM or a proper algorithmic implementation is expected.
        * **Structured Output & Tools:** If the task implies structured output from an LLM or the use of tools, define necessary Pydantic models and/or LangChain tools, and integrate them with the LLM call.
            * Define a Pydantic model (e.g., `from langchain_core.pydantic_v1 import BaseModel, Field`) representing the desired structured output.
            * If implementing an LLM call, configure it to use the Pydantic model for its output (e.g., with OpenAI's function calling/tool usage features, or by instructing the LLM to generate JSON conforming to the model).
        * **Tool Definition and Usage:** If a node's `description` (or your architectural design) implies the LLM within that node needs to interact with external systems, perform specific actions, or fetch data (e.g., "search customer database," "get weather update"):
                * Define these capabilities as discrete LangChain tools using the `@tool` decorator (e.g., `from langchain_core.tools import tool`).
                * **Crucially, each tool's internal Python function should be self-contained and directly perform its advertised action** (e.g., make a specific API call to an external service, run a local script, perform a calculation, retrieve data algorithmically). **Avoid embedding a *new, separate general-purpose LLM call within the tool's own implementation logic* unless the tool's explicit and documented purpose is to be a specialized, self-contained sub-agent (which is an advanced case).** The primary LLM within the graph node is responsible for *deciding to call* the tool and for interpreting its output.
                * Bind these well-defined tools to the LLM instance operating within that graph node. The node's LLM will then intelligently decide when to call a tool and with what inputs.
        * **Human in the Loop Nodes:** If you've designed a HITL step as a dedicated node, its function might primarily format data for human review and then process the subsequent human input (which would be added to the state, potentially by an external mechanism or a subsequent node). The graph might pause using an interruption mechanism tied to this node.
        * **State Coherence:** Ensure variable assignments and updates within node functions are coherent with the `GraphState` definition and how state is managed in LangGraph.

4.  **Graph Construction (`StatefulGraph`):**
    * Instantiate `StatefulGraph(GraphState)`.
    * Add each implemented node function to the graph using `graph.add_node("node_id", node_function)`.
    * Set the graph's entry point using `graph.add_edge(START, "entry_node_id")` where `"entry_node_id"` is the target of the edge originating from `"__START__"`.

5.  **Edge Implementation:**
    * Iterate through the `edges` list in the JSON.
    * **Regular Edges:** If `conditional` is `false`:
        * If `target` is `__END__`, use `graph.add_edge(source_node_id, END)`.
        * Otherwise, use `graph.add_edge(source_node_id, target_node_id)`.
    * **Conditional Edges:** If `conditional` is `true`:
        * The `source` node of these conditional edges is expected to produce some output in the `GraphState` (e.g., an `intent` field) that determines the next path.
        * Create a separate routing function (e.g., `def route_after_source_node(state: GraphState) -> str:`).
        * This routing function must inspect the relevant fields in the `state` and return the string ID of the next node to execute, based on the logic described in the `routing_conditions` for each conditional edge originating from that source.
        * Use `graph.add_conditional_edges(source_node_id, routing_function, {{ "target_id_1": "target_id_1", "target_id_2": "target_id_2", ... "__END__": END }})`. The keys in the dictionary are the possible return values from your routing function, and the values are the actual node IDs or `END`.

6.  **Compilation:**
    * Compile the graph: `app = graph.compile()`. The compiled graph must be assigned to a variable named `app`.
    * add the following code just after above step.
    with open("/home/user/graph.json" , "w" ) as write:
        json.dump(app.get_graph(xray=True).to_json(), write)
    * If you write 'main' code, instead of printing, append the logs to /home/user/llm_stream.txt 
    

</CODE_GENERATION_INSTRUCTIONS>

<KEY_EXTRACTION_INSTRUCTIONS>
After generating the complete Python script, add a separate section at the end of your response, clearly titled:
`## Required Keys and Credentials`
In this section, list all environment variables or API keys a user would need to set for the generated code to execute successfully (e.g., `OPENAI_API_KEY`, `GOOGLE_API_KEY`, tool-specific keys). If no external keys are needed, state that.
</KEY_EXTRACTION_INSTRUCTIONS>

You need to use <INPUT> section and based on the CODE_GENERATION_INSTRUCTIONS and KEY_EXTRACTION_INSTRUCTIONS

**Important Considerations (General):**
* The primary goal is **compilable, logical, and functionally plausible Python code** that intelligently interprets the JSON input.
* Focus on creating a system that leverages LLMs effectively for tasks suited to them.
* Ensure node functions correctly update and return relevant parts of the `GraphState`.
* Handle `__START__` and `__END__` correctly in edge definitions. `langgraph.graph.START` and `langgraph.graph.END` should be used.
""")

def code_node(state: AgentBuilderState):
    """
    LangGraph node to generate the final Python code for the agent.
    It uses the gathered agent_instructions and the CODE_GEN_PROMPT.
    """
    instructions: AgentInstructions = state["agent_instructions"]
 
    # Invoke LLM to generate code based on the detailed prompt and instructions
    code_output = llm.invoke([HumanMessage(content=CODE_GEN_PROMPT.format(
        json_dict=state["json_dict"],
        justification = state["justification"],
        objective=instructions.objective,
        usecases=instructions.usecases,
        examples=instructions.examples
    ))])
    # Return the generated Python code and an AI message
    return {
        "messages": [AIMessage(content="Generated final python code!")],
        "python_code": code_output.content,
    }
