from dotenv import load_dotenv
load_dotenv()
import pytest
from src_folder.tests.test_utils.stock_agent.stock_json import stock_json
from final_code.states.NodesAndEdgesSchemas import JSONSchema

@pytest.mark.parametrize(
    "json_schema",
    [
        (stock_json)
    ],
)
def test_validate_json(json_schema : str):
    assert JSONSchema.model_validate_json(json_schema)