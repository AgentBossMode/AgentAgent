from dotenv import load_dotenv
load_dotenv()
import pytest
import ast
from src_folder.final_code.nodes.code_generation_node import generate_python_code, code_analyzer_node
from src_folder.tests.test_utils.stock_agent.stock_tools import stock_tools
from src_folder.tests.test_utils.stock_agent.stock_json import stock_json

from src_folder.final_code.states.NodesAndEdgesSchemas import JSONSchema
from src_folder.final_code.ast_visitors_lib.validation_script import run_detailed_validation
from src_folder.final_code.utils.get_filtered_file import get_filtered_file

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "json_schema,tools_code",
    [
        (stock_json, stock_tools)
    ],
)
async def test_code_generation_llm(json_schema : str, tools_code: str):
    generated_code = await generate_python_code({}, JSONSchema.model_validate_json(json_schema), tools_code)
    state = await code_analyzer_node({"python_code": generated_code}, {})
    validation_report = run_detailed_validation(get_filtered_file(state["python_code"]))
    for error in validation_report["errors"]:
        print(error)
    total_errors = validation_report["summary"]["total_errors"]
    total_warnings = validation_report["summary"]["total_warnings"]
    assert total_errors == 0 and total_warnings == 0
