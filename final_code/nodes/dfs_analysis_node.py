from langchain_core.messages import AIMessage, HumanMessage
from final_code.llms.model_factory import get_model
from final_code.states.AgentBuilderState import AgentBuilderState

llm = get_model()


ANALYSIS_COMPILE_PROMPT = """
 You are an expert LangGraph code refactoring AI. Your task is to analyze the provided Python code for a LangGraph implementation and automatically correct it to ensure adherence to best practices, specifically concerning tool definition within nodes and the consistency of state objects.
 Please perform the following analysis and apply corrections directly to the code:
 Tool Definition and Usage in Nodes:
 Correct Invocation: Identify and fix any instances where tools called within graph nodes are not defined or invoked correctly according to LangGraph's protocols.
 Schema Adherence: Ensure that the inputs provided to tools and the outputs received from them strictly adhere to their defined schemas. Modify the code to align with these schemas if discrepancies are found.
 Tool Registration (if applicable): If tools are not correctly registered or made available to the nodes that intend to use them, update the code to ensure proper registration.
 Error Handling: Implement or improve error handling for tool execution failures within the nodes, making the graph more robust.
 State Object Management and Consistency:
 State Definition Review (and potential refinement): While the primary goal is to ensure consistency with the existing definition, if the state definition itself is unclear or problematic for achieving consistency, you may suggest minor refactorings to the state definition (clearly noting these changes).
 Node-State Interaction: For each node:
 Input State: Correct any instances where the node incorrectly accesses or misinterprets information from the input state.
 Output State: Rectify how the node updates the state object to ensure it's consistent with its defined purpose, the overall graph flow, and the state definition. Ensure all modifications are explicit and correct.
 Type Consistency: Enforce that the data types of values being read from and written to the state are consistent across different nodes and with the state definition. Apply necessary type conversions or corrections.
 Immutability (where applicable): If parts of the state are intended to be immutable but are modified, adjust the node logic to respect this or ensure modifications are handled through proper state update mechanisms.
 State Transitions: Refine the logic of state changes between nodes if it leads to inconsistencies or deviates from the intended graph objective.
 Overall Code Health (related to tools and state):
 Clarity and Readability: Refactor code related to tool usage and state manipulation to improve its clarity and readability, potentially by adding comments or restructuring logic.
 Modularity: If tool definitions or state interactions can be better encapsulated within their respective nodes for improved modularity, make these changes.
 Input:
 You will be provided with the Python code for the LangGraph implementation.
 Output:
 Provide the updated and corrected LangGraph Python code.
 input code:
 <input_code>
 {compiled_code}
 </input_code>
 """

def dfs_analysis_node(state: AgentBuilderState): # Renamed for clarity
     """
     LangGraph node to analyse the code
     """
     main_agent_code = state['python_code']

     # Use LLM to merge the main agent code with the generated tool definitions
     response = llm.invoke([HumanMessage(content=ANALYSIS_COMPILE_PROMPT.format(
         compiled_code=main_agent_code,
     ))])

     # The response from this LLM call is expected to be the final, complete Python code
     return {
         "messages": [AIMessage(content=response.content)], # Storing the LLM's final code as a message for now
         "python_code": response.content # Update compiled_code with the final merged code
     }

