from langgraph.graph import StateGraph, START # Core LangGraph components for building stateful graphs
from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.nodes.req_analysis_node import requirement_analysis_node,analyze_reqs, generate_dry_run, dry_run_interrupt 
from final_code.nodes.tool_generation_nodev2 import tool_graph
from final_code.nodes.json_generation_node import json_node
from final_code.nodes.code_generation_node import code_node, code_analyzer_node
from final_code.nodes.dfs_analysis_node import dfs_analysis_node
from final_code.nodes.tool_interrupt import tool_interrupt, add_toolset
from final_code.nodes.extract_env_var_node import env_var_node
from final_code.nodes.combine_code_node import combine_code_pipeline_graph
from final_code.nodes.evaluation_pipeline_nodes.mock_tools_writer import mock_tools_writer
from final_code.nodes.evaluation_pipeline_nodes.pytest_writer import pytest_writer
from final_code.nodes.evaluation_pipeline_nodes.syntactic_code_reflection import reflection_node
from final_code.nodes.evaluation_pipeline_nodes.pytest_runner import pytest_runner
from final_code.nodes.evaluation_pipeline_nodes.pytest_evaluation import evaluation_start, evaluation_supervisor


main_workflow = StateGraph(AgentBuilderState) # Define state type

# Add nodes to the main workflow
main_workflow.add_node("analyze_reqs", analyze_reqs)
main_workflow.add_node("requirement_analysis_node", requirement_analysis_node)
main_workflow.add_node("generate_dry_run", generate_dry_run)
main_workflow.add_node("dry_run_interrupt", dry_run_interrupt) # Renamed node
main_workflow.add_node("json_node", json_node)
main_workflow.add_node("code_node", code_node)
main_workflow.add_node("tool_interrupt", tool_interrupt)
main_workflow.add_node("add_toolset", add_toolset)
main_workflow.add_node("tool_graph", tool_graph) # Renamed node
main_workflow.add_node("mock_tools_writer", mock_tools_writer)
main_workflow.add_node("pytest_writer", pytest_writer)
main_workflow.add_node("reflection", reflection_node)
main_workflow.add_node("pytest_runner", pytest_runner)
main_workflow.add_node("evaluation_start", evaluation_start)
main_workflow.add_node("evaluation_supervisor", evaluation_supervisor)
main_workflow.add_node("code_analyzer_node", code_analyzer_node)


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
main_workflow.add_edge("code_node", "code_analyzer_node")
main_workflow.add_edge("code_analyzer_node", "mock_tools_writer")
main_workflow.add_edge("mock_tools_writer", "pytest_writer")
main_workflow.add_edge("pytest_writer", "reflection")
main_workflow.add_edge("reflection", "pytest_runner")

# from src_folder.tests.test_utils.github_issues_agent.github_code import github_code
# from src_folder.tests.test_utils.github_issues_agent.github_tests import github_tests
# from src_folder.tests.test_utils.github_issues_agent.github_mock_tools import github_mock_tools
# from src_folder.tests.test_utils.stock_agent.stock_main import stock_main
# from src_folder.tests.test_utils.stock_agent.stock_tests import stock_tests
# from src_folder.tests.test_utils.stock_agent.stock_mock_tools import stock_mock_tools
# from src_folder.tests.test_utils.stock_agent.stock_uts import stock_uts
# from src_folder.final_code.pydantic_models.UtGen import UtGeneration
# import json
# def setup_code(state: AgentBuilderState) -> AgentBuilderState:
#     return {"python_code": stock_main, "pytest_code": stock_tests, "mock_tools_code": stock_mock_tools, "console_logs": [],
#        "current_status": {},
#          "messages": [],
#          "dry_runs": [],
#          "req_analysis": {},
#          "utGeneration": UtGeneration.model_validate_json(stock_uts)}
# main_workflow.add_node("setup_code", setup_code)
# main_workflow.add_edge(START, "setup_code")
# main_workflow.add_edge("setup_code", "pytest_runner")

app = main_workflow.compile()
