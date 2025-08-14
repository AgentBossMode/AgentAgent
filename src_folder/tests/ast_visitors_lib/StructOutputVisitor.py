import ast

class StructOutputVistor(ast.NodeVisitor):
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

