
import ast

class StructOutputClassValidator(ast.NodeVisitor):
    def __init__(self, structured_output_args):
        self.classes = []
        self.structured_output_args = structured_output_args

    def visit_ClassDef(self, node):
        if node.name =="GraphState":
            assert node.bases[0].id == "MessagesState"
        elif node.name in self.structured_output_args:
            assert any(base.id == "BaseModel" for base in node.bases), f"class {node.name} should contain BaseModel"
        self.generic_visit(node)

