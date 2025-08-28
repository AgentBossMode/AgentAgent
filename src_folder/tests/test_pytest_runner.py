# import pytest
# from dotenv import load_dotenv
# load_dotenv()
# from src_folder.final_code.nodes.evaluation_pipeline_nodes.pytest_evaluation import evaluation_start, evaluation_supervisor
# from src_folder.final_code.nodes.evaluation_pipeline_nodes.pytest_runner import pytest_runner
# from src_folder.tests.test_utils.github_issues_agent.github_code import github_code
# from src_folder.tests.test_utils.github_issues_agent.github_mock_tools import github_mock_tools
# from src_folder.tests.test_utils.github_issues_agent.github_tests import github_tests
# from langgraph.graph import StateGraph, START
# from src_folder.final_code.states.AgentBuilderState import AgentBuilderState
# from src_folder.final_code.pydantic_models.UtGen import UtGeneration

# @pytest.mark.asyncio
# async def test_pipeline():
#     workflow = StateGraph(AgentBuilderState)
#     workflow.add_node("pytest_runner", pytest_runner)
#     workflow.add_node("evaluation_supervisor", evaluation_supervisor)
#     workflow.add_node("evaluation_start", evaluation_start)
#     workflow.add_edge(START, "pytest_runner")
#     app = workflow.compile()
#     state = {
#         "python_code": github_code,
#         "pytest_code": github_tests,
#         "mock_tools_code": github_mock_tools,
#         "console_logs": [],
#         "current_status": {},
#         "messages": [],
#         "dry_runs": [],
#         "req_analysis": {},
#         "utGeneration":UtGeneration.model_validate({"trajectory_uts":[]})
#     }

#     config = {"type":"test"}
#     final_result = await app.ainvoke(input=state, config=config)
#     assert False
