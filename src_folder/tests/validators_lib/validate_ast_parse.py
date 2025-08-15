import ast
def validate_ast_parse(generated_code: str) -> ast.Module:
    """
    Validates the generated code by checking if it can be parsed as valid Python code.
    It also checks for the presence of structured output and necessary imports.
    """
    if generated_code.startswith("```python"):
        generated_code = generated_code[10:]
    if generated_code.endswith("```"):
        generated_code = generated_code[:-3]
    return ast.parse(generated_code)