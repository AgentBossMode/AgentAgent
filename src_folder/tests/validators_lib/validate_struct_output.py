import ast
from src_folder.final_code.ast_visitors_lib.StructOutputVisitor import StructOutputVistor
from src_folder.final_code.ast_visitors_lib.StructOutputClassValidator import StructOutputClassValidator

def validate_struct_output(code_module: ast.Module):
    structured_output_visitor = StructOutputVistor()
    structured_output_visitor.visit(code_module)
    visitor = StructOutputClassValidator(structured_output_visitor.structured_output_args)
    visitor.visit(code_module)
