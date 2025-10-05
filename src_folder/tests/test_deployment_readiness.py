import ast
from src_folder.final_code.nodes.deployment_readiness import deployment_readiness, StateInheritanceTransformer
from tests.test_utils.stock_agent.stock_main import stock_main
from langgraph.types import Command
def test_deployment_readiness():
    new_state: Command = deployment_readiness({"python_code": stock_main}, {})
    module = ast.parse(new_state.update["python_code"])
    transformer = StateInheritanceTransformer()
    new_tree = transformer.visit(module)
    assert transformer.copilotkit_import_exists == True
    assert transformer.transformation_made == False