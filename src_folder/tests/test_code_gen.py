from dotenv import load_dotenv
load_dotenv()
import pytest
from src_folder.final_code.nodes.code_generation_node import generate_python_code
from src_folder.tests.test_utils.nutrition_agent_files.nutrition_json_schema import json_schema_nutrition
from src_folder.tests.test_utils.nutrition_agent_files.nutrition_tools_code import nutrition_tools_code
from src_folder.final_code.states.NodesAndEdgesSchemas import JSONSchema
import ast

class ClassVisitor(ast.NodeVisitor):
    def __init__(self, structured_output_args):
        self.classes = []
        self.structured_output_args = structured_output_args

    def visit_ClassDef(self, node):
        if node.name =="GraphState":
            assert node.bases[0].id == "MessagesState"
        elif node.name in self.structured_output_args:
            assert any(base.id == "BaseModel" for base in node.bases), f"class {node.name} should contain BaseModel"
        self.generic_visit(node)


class StructuredOutputVisitor(ast.NodeVisitor):
    def __init__(self):
        self.structured_output_args = []

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and \
           node.func.attr == 'with_structured_output' and \
           isinstance(node.func.value, ast.Name) and \
           node.func.value.id == 'llm':
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    self.structured_output_args.append(arg.id)
        self.generic_visit(node)

@pytest.mark.parametrize(
    "json_schema,tools_code",
    [
        (json_schema_nutrition, nutrition_tools_code)
    ],
)
def test_code_generation_llm(json_schema : str, tools_code: str):
    generated_code = generate_python_code({}, JSONSchema.model_validate_json(json_schema), tools_code)
    print(generated_code)  # Print the generated code for debugging

    # assume code is a python code, how to write uts 
    assert isinstance(generated_code, str), "Generated code should be a string."
    assert "assuming" not in generated_code, "Generated code should not contain 'assuming'."
    assert "placeholder" not in generated_code, "Generated code should not contain 'placeholder'."
    assert "checkpointer = InMemorySaver()" in generated_code, "Generated code should contain 'checkpointer = InMemorySaver()'."
    assert "app = workflow.compile(" in generated_code, "Generated code should contain 'app = workflow.compile('."
    
    # IF AND ONLY IF THE CODE HAS with_structured_output, check if there is pydantic import
    if "with_structured_output" in generated_code:
        assert "from pydantic import BaseModel, Field" in generated_code, "Generated code should contain 'from pydantic import BaseModel, Field'."
    # use the ast module to check if the code is valid python code
    try:
        # with open("generated_code.py", "w") as f:
        #    f.write(ast.dump(module, indent=2))
        # in generated code remove the first and last line if they are ````python` and ```
        if generated_code.startswith("```python"):
            generated_code = generated_code[10:]
        if generated_code.endswith("```"):
            generated_code = generated_code[:-3]
        module = ast.parse(generated_code)
        structured_output_visitor = StructuredOutputVisitor()
        structured_output_visitor.visit(module)
        visitor = ClassVisitor(structured_output_visitor.structured_output_args)
        visitor.visit(module)
        # searc
    except SyntaxError as e:
        pytest.fail(f"Generated code contains syntax errors: {e}")
    except Exception as e:
        pytest.fail(f"Generated code contains unexpected errors: {e}")
