from dotenv import load_dotenv
load_dotenv()
import pytest
import ast
from src_folder.final_code.nodes.code_generation_node import generate_python_code
from src_folder.tests.test_utils.nutrition_agent_files.nutrition_json_schema import json_schema_nutrition
from src_folder.tests.test_utils.nutrition_agent_files.nutrition_tools_code import nutrition_tools_code
from src_folder.final_code.states.NodesAndEdgesSchemas import JSONSchema
from src_folder.tests.validators_lib.validate_struct_output import validate_struct_output
from src_folder.tests.validators_lib.validate_ast_parse import validate_ast_parse
from src_folder.tests.validators_lib.validate_import_statements import validate_import_statements
from src_folder.final_code.ast_visitors_lib.PydanticDictVisitor import PydanticDictVisitor
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "json_schema,tools_code",
    [
        (json_schema_nutrition, nutrition_tools_code)
    ],
)
async def test_code_generation_llm(json_schema : str, tools_code: str):
    generated_code = await generate_python_code({}, JSONSchema.model_validate_json(json_schema), tools_code)

    # assume code is a python code, how to write uts 
    assert isinstance(generated_code, str), "Generated code should be a string."
    assert "checkpointer = InMemorySaver()" in generated_code, "Generated code should contain 'checkpointer = InMemorySaver()'."
    assert "app = workflow.compile(" in generated_code, "Generated code should contain 'app = workflow.compile('."
    
    # IF AND ONLY IF THE CODE HAS with_structured_output, check if there is pydantic import
    if "with_structured_output" in generated_code:
        assert "from pydantic import BaseModel, Field" in generated_code, "Generated code should contain 'from pydantic import BaseModel, Field'."
    # use the ast module to check if the code is valid python code
    try:
        # with open("generated_code.py", "w") as f:
        #    f.write(ast.dump(module, indent=2))
        module = validate_ast_parse(generated_code)
        validate_struct_output(module)
        validate_import_statements(module)
        visitor = PydanticDictVisitor()
        visitor.visit(module)
        assert not visitor.errors, f"Found forbidden 'dict' types in Pydantic models:\n" + "\n".join(visitor.errors)
    except SyntaxError as e:
        pytest.fail(f"Generated code contains syntax errors: {e}")
    except Exception as e:
        pytest.fail(f"Generated code contains unexpected errors: {e}")
