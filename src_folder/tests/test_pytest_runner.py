# import pytest
# from dotenv import load_dotenv
# load_dotenv()
# from src_folder.final_code.nodes.evaluation_pipeline_nodes.pytest_evaluation import pytest_runner, evaluation_start, evaluation_supervisor
# from src_folder.tests.test_utils.nutrition_agent_files.pytest_runner_code_files.nutrition_main import nutrition_main
# from src_folder.tests.test_utils.nutrition_agent_files.pytest_runner_code_files.nutrition_mock_tools import nutrition_mock_tools
# from src_folder.tests.test_utils.nutrition_agent_files.pytest_runner_code_files.nutrition_tests import nutrition_tests
# from langgraph.graph import StateGraph, START
# from src_folder.final_code.states.AgentBuilderState import AgentBuilderState

# @pytest.mark.asyncio
# async def test_pipeline():
#     workflow = StateGraph(AgentBuilderState)
#     workflow.add_node("pytest_runner", pytest_runner)
#     workflow.add_node("evaluation_supervisor", evaluation_supervisor)
#     workflow.add_node("evaluation_start", evaluation_start)
#     workflow.add_edge(START, "pytest_runner")
#     app = workflow.compile()
#     state = {
#         "python_code": nutrition_main,
#         "pytest_code": nutrition_tests,
#         "mock_tools_code": nutrition_mock_tools,
#         "console_logs": [],
#         "current_status": {},
#         "messages": [],
#         "dry_runs": [],
#         "req_analysis": {},
#         "utGeneration":[]
#     }

#     config = {"type":"test"}
#     final_result = await app.ainvoke(input=state, config=config)
#     print(final_result)
#     assert False