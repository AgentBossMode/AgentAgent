import pytest
from final_code.nodes.code_generation_node import generate_python_code
from final_code.utils.MockJsonSchema import json_schema_str
from final_code.nodes.code_generation_node import PythonCode
import ast

class ClassVisitor(ast.NodeVisitor):
    def __init__(self):
        self.classes = []

    def visit_ClassDef(self, node):
        if node.name =="GraphState":
            assert node.bases[0].id == "MessagesState"
        self.generic_visit(node)

def test_code_generation_llm():
    generated_code: PythonCode = generate_python_code({}, json_schema_str)
    print(generated_code.code)  # Print the generated code for debugging

    # assume code is a python code, how to write uts 
    assert isinstance(generated_code.code, str), "Generated code should be a string."
    assert "assuming" not in generated_code.code, "Generated code should not contain 'assuming'."
    assert "placeholder" not in generated_code.code, "Generated code should not contain 'placeholder'."
    assert "checkpointer = InMemorySaver()" in generated_code.code, "Generated code should contain 'checkpointer = InMemorySaver()'."
    assert "app = workflow.compile(" in generated_code.code, "Generated code should contain 'app = workflow.compile('."
    
    # IF AND ONLY IF THE CODE HAS with_structured_output, check if there is pydantic import
    if "with_structured_output" in generated_code.code:
        assert "from pydantic import BaseModel, Field" in generated_code.code, "Generated code should contain 'from pydantic import BaseModel, Field'."
    # use the ast module to check if the code is valid python code
    try:
        with open("generated_code.py", "w") as f:
            module = ast.parse(generated_code.code)
            f.write(ast.dump(module, indent=2))
            visitor = ClassVisitor()
            visitor.visit(module)

        # searc
    except SyntaxError as e:
        pytest.fail(f"Generated code contains syntax errors: {e}")
    except Exception as e:
        pytest.fail(f"Generated code contains unexpected errors: {e}")
