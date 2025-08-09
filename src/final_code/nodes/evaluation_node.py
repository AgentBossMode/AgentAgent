# generate use cases
from langgraph.graph import StateGraph, START, END
from src.final_code.states.CodeEvalState import CodeEvalState
from src.final_code.nodes.evaluation_pipeline_nodes.mock_test_writer import mock_test_writer
from src.final_code.nodes.evaluation_pipeline_nodes.pytest_writer import pytest_writer
from src.final_code.nodes.evaluation_pipeline_nodes.syntactic_code_reflection import reflection_node
from src.final_code.nodes.evaluation_pipeline_nodes.pytest_evaluation import pytest_runner, evaluation_supervisor, fix_assert_fail_issue_node, syntax_and_runtime_issues_node

from langgraph.graph import END


workflow = StateGraph(CodeEvalState)
workflow.add_node("mock_test_writer", mock_test_writer)
workflow.add_node("pytest_writer", pytest_writer)
workflow.add_node("reflection", reflection_node)
workflow.add_node("pytest_runner", pytest_runner)
workflow.add_node("syntax_and_runtime_issues_node", syntax_and_runtime_issues_node)
workflow.add_node("fix_assert_fail_issue_node", fix_assert_fail_issue_node)
workflow.add_node("evaluation_supervisor", evaluation_supervisor)

workflow.add_edge(START, "mock_test_writer")
workflow.add_edge("mock_test_writer", "pytest_writer")
workflow.add_edge("pytest_writer", "reflection")
workflow.add_edge("reflection", "pytest_runner")
eval_pipeline_graph = workflow.compile()
