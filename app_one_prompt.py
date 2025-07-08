from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END # Core LangGraph components for building stateful graphs
from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.nodes.req_analysis_node import requirement_analysis_node
from final_code.nodes.evaluation_node import eval_pipeline_graph
from final_code.nodes.tool_generation_nodev2 import tool_graph
from final_code.nodes.json_generation_node import json_node, dry_run_node
from final_code.nodes.code_generation_node import code_node
from final_code.nodes.dfs_analysis_node import dfs_analysis_node
from final_code.nodes.tool_interrupt import tool_interrupt, add_toolset
from final_code.nodes.code_reflection_node import code_reflection_node_updated
from langchain_core.messages import HumanMessage

main_workflow = StateGraph(AgentBuilderState) # Define state type

# Add nodes to the main workflow
main_workflow.add_node("requirement_analysis_node", requirement_analysis_node)
main_workflow.add_node("json_node", json_node)
main_workflow.add_node("dry_run_node", dry_run_node)  # Add dry_run_node
main_workflow.add_node("code_node", code_node)
main_workflow.add_node("tool_interrupt", tool_interrupt)
main_workflow.add_node("add_toolset", add_toolset)
main_workflow.add_node("tool_graph", tool_graph) # Renamed node
# app2
main_workflow.add_node("eval_pipeline", eval_pipeline_graph) # Add evaluation pipeline graph

main_workflow.add_node("dfs_analysis_node", dfs_analysis_node)

# EDGE SECTION

# PROD workflow
main_workflow.add_edge(START, "requirement_analysis_node")
main_workflow.add_edge("json_node", "dry_run_node")  # Connect json_node to dry_run_node
main_workflow.add_edge("dry_run_node", "tool_graph")  # Connect dry_run_node to tool_graph
main_workflow.add_edge("tool_graph", "tool_interrupt")
main_workflow.add_edge("tool_interrupt", "code_node")
main_workflow.add_edge("code_node", "dfs_analysis_node")
main_workflow.add_edge("dfs_analysis_node", "eval_pipeline")
main_workflow.add_edge("eval_pipeline", END)

# uncomment to test tool_set composio and comment the rest
# main_workflow.set_entry_point("add_toolset")
# main_workflow.add_edge(START, "add_toolset")
# main_workflow.add_edge("add_toolset", "tool_interrupt")
# main_workflow.add_edge("tool_interrupt", END)

# uncomment to test code_node and comment the rest
# main_workflow.set_entry_point("code_node")
# main_workflow.add_edge(START, "code_node")
# main_workflow.add_edge("code_node", END)

app = main_workflow.compile()
