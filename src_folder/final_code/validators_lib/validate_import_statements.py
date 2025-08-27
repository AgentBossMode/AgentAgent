import ast
from final_code.ast_visitors_lib.ImportVisitor import ImportVisitor
def validate_import_statements(code_module: ast.Module):
    """
    Validates that the generated code contains necessary import statements.
    """
    methods_to_check = [
            ("create_react_agent", "from langgraph.prebuilt import create_react_agent")
        ]
    visitor = ImportVisitor(methods_to_check)
    visitor.visit(code_module)
    results = visitor.get_results()
    assert len(results["missing_imports"]) == 0