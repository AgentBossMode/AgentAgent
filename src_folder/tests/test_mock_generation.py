import pytest 
import asyncio
import ast
import json
from dotenv import load_dotenv
load_dotenv()
from src_folder.final_code.nodes.evaluation_pipeline_nodes.mock_test_writer import mock_test_writer
from tests.test_utils.nutrition_agent_files.nutrition_tools_code import nutrition_tools_code
from tests.test_utils.nutrition_agent_files.nutrition_json_schema import json_schema_nutrition
from src_folder.final_code.states.NodesAndEdgesSchemas import JSONSchema
class PydanticDictVisitor(ast.NodeVisitor):
    """
    AST visitor to check for 'dict' type hints in Pydantic models.
    """
    def __init__(self):
        self.errors = []

    def visit_ClassDef(self, node):
        # Check if the class inherits from BaseModel
        is_basemodel = any(
            isinstance(base, ast.Name) and base.id == 'BaseModel' 
            for base in node.bases
        )
        
        if is_basemodel:
            for item in node.body:
                if isinstance(item, ast.AnnAssign):
                    # This is an annotated assignment inside a BaseModel.
                    # Walk the annotation to see if 'dict' is used as a type name.
                    if any(sub_node.id == 'dict' for sub_node in ast.walk(item.annotation) if isinstance(sub_node, ast.Name)):
                        field_name = item.target.id if isinstance(item.target, ast.Name) else 'unknown'
                        self.errors.append(f"Field '{field_name}' in Pydantic model '{node.name}' uses forbidden type 'dict'.")
        
        # Continue visiting nested classes
        self.generic_visit(node)

def test_mock_test_writer():
    state = {
        "tools_code": nutrition_tools_code,
        "json_schema": JSONSchema.model_validate_json(json_schema_nutrition),
        "current_status": {}
    }
    config = {"type":"test"}

    # Call the mock_test_writer function
    result = asyncio.run(mock_test_writer(state, config))
    # Check if the result is as expected
    assert isinstance(result, dict)
    assert "mock_tools_code" in result
    mock_tools_code = result["mock_tools_code"]

    print(mock_tools_code)

    # AST-based check to ensure no 'dict' types are in Pydantic models
    try:
        tree = ast.parse(mock_tools_code)
        visitor = PydanticDictVisitor()
        visitor.visit(tree)
        assert not visitor.errors, f"Found forbidden 'dict' types in Pydantic models:\n" + "\n".join(visitor.errors)
    except SyntaxError as e:
        pytest.fail(f"Generated mock_tools_code has a syntax error: {e}\nCode:\n{mock_tools_code}")