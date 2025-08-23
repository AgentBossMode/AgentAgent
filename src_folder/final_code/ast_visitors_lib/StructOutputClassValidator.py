
import ast

class StructOutputClassValidator(ast.NodeVisitor):
    def __init__(self, structured_output_args, dont_assert = False):
        self.classes = []
        self.errors = []
        self.structured_output_args = structured_output_args
        self.dont_assert = dont_assert

    def visit_ClassDef(self, node):
        if node.name =="GraphState":
            if not node.bases[0].id == "MessagesState":
                self.errors.append(f"class {node.name} should inherit from MessagesState")
            if not self.dont_assert:
                assert node.bases[0].id == "MessagesState"
        elif node.name in self.structured_output_args:
            if not any(base.id == "BaseModel" for base in node.bases):
                self.errors.append(f"class {node.name} should contain BaseModel")
            if not self.dont_assert:   
              assert any(base.id == "BaseModel" for base in node.bases), f"class {node.name} should contain BaseModel"
        self.generic_visit(node)

