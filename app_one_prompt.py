from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END # Core LangGraph components for building stateful graphs
from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.nodes.req_analysis_node import requirement_analysis_node
from final_code.nodes.evaluation_node import eval_pipeline_graph
from final_code.nodes.tool_generation_node import tool_compile_graph
from final_code.nodes.tool_generation_nodev2 import app2
from final_code.nodes.json_generation_node import json_node
from final_code.nodes.code_generation_node import code_node
from final_code.nodes.dfs_analysis_node import dfs_analysis_node
from final_code.nodes.code_reflection_node import code_reflection_node_updated

main_workflow = StateGraph(AgentBuilderState) # Define state type

# Add nodes to the main workflow
main_workflow.add_node("requirement_analysis_node", requirement_analysis_node)
main_workflow.add_node("json_node", json_node)
main_workflow.add_node("code_node", code_node)
main_workflow.add_node("tool_subgraph_processing", tool_compile_graph) # Renamed node
# app2
main_workflow.add_node("eval_pipeline", eval_pipeline_graph) # Add evaluation pipeline graph

main_workflow.add_node("dfs_analysis_node", dfs_analysis_node)
main_workflow.add_node("code_reflection_node", code_reflection_node_updated)

# Define edges for the main workflow
main_workflow.add_edge(START, "requirement_analysis_node")
main_workflow.add_edge("json_node", "code_node")
main_workflow.add_edge("code_node", "tool_subgraph_processing")
main_workflow.add_edge("tool_subgraph_processing", "dfs_analysis_node")
main_workflow.add_edge("dfs_analysis_node", "code_reflection_node")
main_workflow.add_edge("code_reflection_node", "eval_pipeline")         # End after tool processing
main_workflow.add_edge("eval_pipeline", END)

app = main_workflow.compile()
