from langchain_core.messages import AIMessage, HumanMessage
from final_code.llms.model_factory import get_model
from final_code.states.AgentBuilderState import AgentBuilderState
from pydantic import BaseModel, Field
from copilotkit.langgraph import copilotkit_customize_config
from langchain_core.runnables import RunnableConfig
llm = get_model()

from final_code.prompt_lib.edge_info.edge_info import edge_checklist, edge_fix_example
from final_code.prompt_lib.node_info.graph_state import node_state_management, graph_state_checklist
from final_code.prompt_lib.node_info.node_structure import function_structure
from final_code.prompt_lib.node_info.tool_calling import tool_calling_checklist
from final_code.prompt_lib.faulty_info.faulty_info import incorrect_example_needing_fixes

ANALYSIS_COMPILE_PROMPT = """
You are an expert LangGraph code refactoring AI. Your task is to analyze the provided Python code for a LangGraph implementation and automatically correct it using comprehensive checklists to ensure adherence to best practices.

## Analysis Checklists

### Node-Level Checklist

For each node in the graph, verify and correct:

#### 1. State Management
{node_state_management}
```
#### 2. Function Structure
{function_structure}

#### 3. Tool Definition and Usage
{tool_calling_checklist}

### Edge-Level Checklist
{edge_checklist}

### Graph-Level Checklist

For the overall graph structure, verify and correct:

#### 1. Graph Configuration
{graph_state_checklist}
#### 2. Workflow Logic
{edge_fix_example}

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
{incorrect_example_needing_fixes}
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

     llm_dfs = llm.with_structured_output(DFSAnalysis)
     # Use LLM to merge the main agent code with the generated tool definitions
     response: DFSAnalysis = llm_dfs.invoke([HumanMessage(content=ANALYSIS_COMPILE_PROMPT.format(
         compiled_code=state['python_code'],
         edge_checklist=edge_checklist,
         node_state_management=node_state_management,
         function_structure=function_structure,
         tool_calling_checklist=tool_calling_checklist,
         graph_state_checklist=graph_state_checklist,
         edge_fix_example=edge_fix_example,
         incorrect_example_needing_fixes=incorrect_example_needing_fixes
     ))], config=modifiedConfig)

     # The response from this LLM call is expected to be the final, complete Python code
     return {
         "messages": [AIMessage(content=response.explanation)], # Storing the LLM's final code as a message for now
         "python_code": response.correct_code # Update compiled_code with the final merged code
     }

