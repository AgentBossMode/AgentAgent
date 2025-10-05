from dotenv import load_dotenv
load_dotenv()
import pytest 
from final_code.nodes.evaluation_pipeline_nodes.mock_tools_writer import mock_tools_writer
from src_folder.tests.test_utils.stock_agent.stock_tools import stock_tools
from src_folder.tests.test_utils.stock_agent.stock_json import stock_json
from src_folder.final_code.ast_visitors_lib.PydanticDictVisitor import PydanticDictVisitor
from src_folder.final_code.states.NodesAndEdgesSchemas import JSONSchema
from src_folder.final_code.validators_lib.validate_struct_output import validate_struct_output
from src_folder.final_code.validators_lib.validate_ast_parse import validate_ast_parse

@pytest.mark.asyncio
async def test_mock_tools_writer():
    state = {
        "tools_code": stock_tools,
        "json_schema": JSONSchema.model_validate_json(stock_json),
        "current_status": {}
    }
    config = {"type":"test"}

    # Call the mock_test_writer function
    result = await mock_tools_writer(state, config)
    # Check if the result is as expected
    assert isinstance(result.update, dict)
    assert "mock_tools_code" in result.update
    mock_tools_code = result.update["mock_tools_code"]

    # AST-based check to ensure no 'dict' types are in Pydantic models
    try:
        tree = validate_ast_parse(mock_tools_code)
        visitor = PydanticDictVisitor()
        visitor.visit(tree)
        assert not visitor.errors, f"Found forbidden 'dict' types in Pydantic models:\n" + "\n".join(visitor.errors)
        validate_struct_output(tree)
    except SyntaxError as e:
        pytest.fail(f"Generated mock_tools_code has a syntax error: {e}\nCode:\n{mock_tools_code}")
    except Exception as e:
        pytest.fail(f"Generated code contains unexpected errors: {e}")