import ast

class PydanticDictVisitor(ast.NodeVisitor):
    """
    AST visitor to check for 'dict' type hints in Pydantic models.
    """
    def __init__(self):
        self.errors = []

    def visit_ClassDef(self, node):
        # Check if the class inherits from BaseModel
        is_basemodel = any(
            isinstance(base, ast.Name) and base.id == 'BaseModel' 
            for base in node.bases
        )
        
        if is_basemodel:
            for item in node.body:
                if isinstance(item, ast.AnnAssign):
                    # This is an annotated assignment inside a BaseModel.
                    # Walk the annotation to see if 'dict' is used as a type name.
                    if any(sub_node.id == 'dict' for sub_node in ast.walk(item.annotation) if isinstance(sub_node, ast.Name)):
                        field_name = item.target.id if isinstance(item.target, ast.Name) else 'unknown'
                        self.errors.append(f"Field '{field_name}' in Pydantic model '{node.name}' uses forbidden type 'dict'. instead make the field a str and then description be to return a JSON string")
        
        # Continue visiting nested classes
        self.generic_visit(node)
