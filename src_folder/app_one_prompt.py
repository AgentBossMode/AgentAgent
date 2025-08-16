from langgraph.graph import StateGraph, START, END # Core LangGraph components for building stateful graphs
from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.nodes.req_analysis_node import requirement_analysis_node,analyze_reqs
from final_code.nodes.tool_generation_nodev2 import tool_graph
from final_code.nodes.json_generation_node import json_node, dry_run_node
from final_code.nodes.code_generation_node import code_node
from final_code.nodes.dfs_analysis_node import dfs_analysis_node
from final_code.nodes.tool_interrupt import tool_interrupt, add_toolset
from final_code.nodes.extract_env_var_node import env_var_node
from final_code.nodes.combine_code_node import combine_code_pipeline_graph
from final_code.nodes.evaluation_pipeline_nodes.mock_test_writer import mock_test_writer
from final_code.nodes.evaluation_pipeline_nodes.pytest_writer import pytest_writer
from final_code.nodes.evaluation_pipeline_nodes.syntactic_code_reflection import reflection_node
from final_code.nodes.evaluation_pipeline_nodes.pytest_evaluation import pytest_runner, evaluation_supervisor, fix_assert_fail_issue_node, syntax_and_runtime_issues_node

main_workflow = StateGraph(AgentBuilderState) # Define state type

# Add nodes to the main workflow
main_workflow.add_node("analyze_reqs", analyze_reqs)
main_workflow.add_node("requirement_analysis_node", requirement_analysis_node)
main_workflow.add_node("json_node", json_node)
main_workflow.add_node("dry_run_node", dry_run_node)  # Add dry_run_node
main_workflow.add_node("code_node", code_node)
main_workflow.add_node("tool_interrupt", tool_interrupt)
main_workflow.add_node("add_toolset", add_toolset)
main_workflow.add_node("tool_graph", tool_graph) # Renamed node
main_workflow.add_node("mock_test_writer", mock_test_writer)
main_workflow.add_node("pytest_writer", pytest_writer)
main_workflow.add_node("reflection", reflection_node)
main_workflow.add_node("pytest_runner", pytest_runner)
main_workflow.add_node("syntax_and_runtime_issues_node", syntax_and_runtime_issues_node)
main_workflow.add_node("fix_assert_fail_issue_node", fix_assert_fail_issue_node)
main_workflow.add_node("evaluation_supervisor", evaluation_supervisor)


main_workflow.add_node("dfs_analysis_node", dfs_analysis_node)
main_workflow.add_node("combine_code_pipeline_graph", combine_code_pipeline_graph)

main_workflow.add_node("env_var_node", env_var_node)
# Define edges for the main workflow

# EDGE SECTION

# PROD workflow
main_workflow.add_edge(START, "analyze_reqs")
main_workflow.add_edge("json_node",  "tool_graph")  # Connect json_node to dry_run_node
main_workflow.add_edge("tool_graph", "tool_interrupt")
main_workflow.add_edge("tool_interrupt", "code_node")
main_workflow.add_edge("code_node", "mock_test_writer")
main_workflow.add_edge("mock_test_writer", "pytest_writer")
main_workflow.add_edge("pytest_writer", "reflection")
main_workflow.add_edge("reflection", "pytest_runner")
# main_workflow.add_edge("env_var_node", "eval_pipeline")
# main_workflow.add_edge("combine_code_pipeline_graph", END)

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
