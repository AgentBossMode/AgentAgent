import ast
from src_folder.tests.ast_visitors_lib.StructOutputVisitor import StructOutputVistor
from src_folder.tests.ast_visitors_lib.StructOutputClassValidator import StructOutputClassValidator
def validate_struct_output(generated_code: str):
    if generated_code.startswith("```python"):
        generated_code = generated_code[10:]
    if generated_code.endswith("```"):
        generated_code = generated_code[:-3]
    module = ast.parse(generated_code)
    structured_output_visitor = StructOutputVistor()
    structured_output_visitor.visit(module)
    visitor = StructOutputClassValidator(structured_output_visitor.structured_output_args)
    visitor.visit(module)
