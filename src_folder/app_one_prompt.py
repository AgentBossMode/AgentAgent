from langgraph.graph import StateGraph, START, END # Core LangGraph components for building stateful graphs
from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.nodes.req_analysis_node import requirement_analysis_node,analyze_reqs, generate_dry_run, dry_run_interrupt 
from final_code.nodes.tool_generation_nodev2 import get_composio_tools_node, process_non_composio_tools, generate_tools_code
from final_code.nodes.json_generation_node import json_node
from final_code.nodes.code_generation_node import code_node, code_analyzer_node
from final_code.nodes.tool_interrupt import tool_interrupt
from final_code.nodes.evaluation_pipeline_nodes.mock_tools_writer import mock_tools_writer
from final_code.nodes.evaluation_pipeline_nodes.pytest_writer import pytest_writer
from final_code.nodes.get_additional_info import generate_additional_info_questions, additional_info_interrupt
from final_code.nodes.evaluation_pipeline_nodes.pytest_runner import pytest_runner
from final_code.nodes.evaluation_pipeline_nodes.pytest_evaluation import evaluation_start, evaluation_supervisor
from final_code.nodes.deployment_readiness import deployment_readiness


main_workflow = StateGraph(AgentBuilderState) # Define state type

# Add nodes to the main workflow
main_workflow.add_node("analyze_reqs", analyze_reqs)
main_workflow.add_node("requirement_analysis_node", requirement_analysis_node)
main_workflow.add_node("generate_dry_run", generate_dry_run)
main_workflow.add_node("dry_run_interrupt", dry_run_interrupt)
main_workflow.add_node("json_node", json_node)
main_workflow.add_node("get_composio_tools", get_composio_tools_node)
main_workflow.add_node("process_non_composio_tools", process_non_composio_tools)
main_workflow.add_node("generate_tools_code", generate_tools_code)
main_workflow.add_node("tool_interrupt", tool_interrupt)
main_workflow.add_node("generate_additional_info_questions", generate_additional_info_questions)
main_workflow.add_node("additional_info_interrupt", additional_info_interrupt)
main_workflow.add_node("code_node", code_node)
main_workflow.add_node("code_analyzer_node", code_analyzer_node)
main_workflow.add_node("mock_tools_writer", mock_tools_writer)
main_workflow.add_node("pytest_writer", pytest_writer)
main_workflow.add_node("pytest_runner", pytest_runner)
main_workflow.add_node("evaluation_start", evaluation_start)
main_workflow.add_node("evaluation_supervisor", evaluation_supervisor)
main_workflow.add_node("deployment_readiness", deployment_readiness)


# EDGE SECTION

# PROD workflow
main_workflow.add_edge(START, "analyze_reqs")
main_workflow.add_edge("json_node",  "get_composio_tools")  # Connect json_node to dry_run_node
main_workflow.add_edge("get_composio_tools", "process_non_composio_tools")
main_workflow.add_edge("process_non_composio_tools", "generate_tools_code")
main_workflow.add_edge("generate_tools_code", "tool_interrupt")
main_workflow.add_edge("tool_interrupt", "generate_additional_info_questions")
main_workflow.add_edge("generate_additional_info_questions", "additional_info_interrupt")
main_workflow.add_edge("additional_info_interrupt", "code_node")
main_workflow.add_edge("code_node", "code_analyzer_node")
main_workflow.add_edge("code_analyzer_node", "mock_tools_writer")
main_workflow.add_edge("mock_tools_writer", "pytest_writer")
main_workflow.add_edge("pytest_writer", "pytest_runner")
main_workflow.add_edge("deployment_readiness", END)

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
