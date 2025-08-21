import ast
from typing import List, Dict, Set, Optional, Any
import re

class LangGraphFormatValidator(ast.NodeVisitor):
    """
    AST Visitor that validates LangGraph code against the specified format requirements.
    Checks for proper state schema, node implementations, edge definitions, and more.
    """
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
        # Track discovered elements
        self.classes: Dict[str, ast.ClassDef] = {}
        self.functions: Dict[str, ast.FunctionDef] = {}
        self.imports: Set[str] = set()
        self.graph_state_class: Optional[ast.ClassDef] = None
        self.edge_calls: List[ast.Call] = []
        self.node_functions: Set[str] = set()
        self.llm_definition: Optional[ast.Assign] = None
        self.has_checkpointer: bool = False
        self.has_app_compile: bool = False
        
        # Required patterns
        self.required_imports = {
            'MessagesState', 'StateGraph', 'ChatOpenAI', 'BaseModel', 'Field',
            'AIMessage', 'HumanMessage', 'SystemMessage'
        }
        
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            for alias in node.names:
                import_name = f"{node.module}.{alias.name}" if alias.name != '*' else node.module
                self.imports.add(import_name)
                self.imports.add(alias.name)  # Also add the direct name
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef):
        self.classes[node.name] = node
        
        # Check for GraphState class
        if node.name.endswith('State') or node.name == 'GraphState':
            self._validate_state_class(node)
            
        # Check for Pydantic models
        if self._inherits_from_basemodel(node):
            self._validate_pydantic_model(node)
            
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.functions[node.name] = node
        self.generic_visit(node)
    
    def visit_Assign(self, node: ast.Assign):
        # Check for llm definition
        if (len(node.targets) == 1 and 
            isinstance(node.targets[0], ast.Name) and 
            node.targets[0].id == 'llm'):
            self.llm_definition = node
            self._validate_llm_definition(node)
        
        # Check for checkpointer
        if (len(node.targets) == 1 and 
            isinstance(node.targets[0], ast.Name) and 
            node.targets[0].id == 'checkpointer'):
            self.has_checkpointer = True
            
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call):
        if (isinstance(node.func, ast.Attribute)) and node.func.attr in ["add_node"]:
            self.node_functions.add(node.args[1].id)
        
        # Track graph edge calls
        if (isinstance(node.func, ast.Attribute) and 
            node.func.attr in ['add_edge', 'add_conditional_edges']):
            self.edge_calls.append(node)
            self._validate_edge_call(node)
        
        # Check for app.compile
        if (isinstance(node.func, ast.Attribute) and 
            node.func.attr == 'compile'):
            self.has_app_compile = True
            
        self.generic_visit(node)
    
    def _validate_state_class(self, node: ast.ClassDef):
        """Validate GraphState class definition"""
        # Check inheritance from MessagesState
        inherits_messages_state = False
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == 'MessagesState':
                inherits_messages_state = True
                break
        
        if not inherits_messages_state:
            self.errors.append(f"State class '{node.name}' must inherit from MessagesState")
            return
        
        self.graph_state_class = node
        
        # Check for explicit messages field (should not exist)
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                if item.target.id == 'messages':
                    self.errors.append(
                        f"State class '{node.name}' should not explicitly define 'messages' field "
                        f"when inheriting from MessagesState. "
                        f"Fix: Remove the explicit messages field definition."
                    )
        
        # Validate field types
        for item in node.body:
            if isinstance(item, ast.AnnAssign):
                self._validate_state_field_type(item, node.name)
    
    def _validate_state_field_type(self, field: ast.AnnAssign, class_name: str):
        """Validate state field type annotations"""
        if isinstance(field.target, ast.Name):
            field_name = field.target.id
            
            # Check for vague types like 'any' or 'dict'
            if isinstance(field.annotation, ast.Name):
                if field.annotation.id in ['any', 'dict', 'Dict']:
                    self.errors.append(
                        f"Field '{field_name}' in class '{class_name}' uses vague type '{field.annotation.id}'. "
                        f"Fix: Use specific types like str, int, List[str], or custom Pydantic models."
                    )
    
    def _validate_pydantic_model(self, node: ast.ClassDef):
        """Validate Pydantic model definitions"""
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                field_name = item.target.id
                
                # Check for dict/Dict usage in Pydantic models
                if isinstance(item.annotation, ast.Name) and item.annotation.id in ['dict', 'Dict']:
                    self.errors.append(
                        f"Pydantic model '{node.name}' field '{field_name}' uses 'dict' type. "
                        f"Fix: Use str and store serialized JSON instead, or create a proper Pydantic model."
                    )
                
                # Check for List[Dict] patterns
                if (isinstance(item.annotation, ast.Subscript) and 
                    isinstance(item.annotation.value, ast.Name) and 
                    item.annotation.value.id == 'List'):
                    if (isinstance(item.annotation.slice, ast.Name) and 
                        item.annotation.slice.id in ['dict', 'Dict']):
                        self.errors.append(
                            f"Pydantic model '{node.name}' field '{field_name}' uses 'List[dict]' type. "
                            f"Fix: Create a proper Pydantic model and use List[YourModel]."
                        )
    
    def _validate_node_function(self, node: ast.FunctionDef):
        """Validate node function implementation"""
        # Check parameters
        if not node.args.args:
            self.errors.append(f"Node function '{node.name}' must have a 'state' parameter")
            return
        
        state_param = node.args.args[0].arg
        if state_param != 'state':
            self.warnings.append(f"Node function '{node.name}' should use 'state' as parameter name")
        
        # Check for docstring
        if not (isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, ast.Constant) and 
                isinstance(node.body[0].value.value, str)):
            self.warnings.append(f"Node function '{node.name}' should have a docstring describing its purpose")
        
        # Check return statement
        has_return = False
        returns_messages = False
        
        for item in ast.walk(node):
            if isinstance(item, ast.Return) and item.value:
                has_return = True
                if isinstance(item.value, ast.Dict):
                    # Check if messages key is present
                    for key in item.value.keys:
                        if (isinstance(key, ast.Constant) and key.value == 'messages'):
                            returns_messages = True
                            break
        
        if not has_return:
            self.errors.append(f"Node function '{node.name}' must return a dictionary")
        elif not returns_messages:
            self.errors.append(
                f"Node function '{node.name}' must include 'messages' key in return dictionary. "
                f"Fix: Add 'messages': [AIMessage(content='...')] to your return statement."
            )
        
        # Check for state access patterns
        self._validate_state_access(node)
    
    def _validate_state_access(self, node: ast.FunctionDef):
        """Validate proper state access patterns"""
        for item in ast.walk(node):
            # Check for attribute access on state (should use dict access)
            if (isinstance(item, ast.Attribute) and 
                isinstance(item.value, ast.Name) and 
                item.value.id == 'state'):
                self.errors.append(
                    f"In function '{node.name}': Use dict access state['field'] instead of state.field. "
                    f"Fix: Change state.{item.attr} to state['{item.attr}']"
                )
    
    def _validate_edge_call(self, node: ast.Call):
        """Validate edge method calls"""
        method_name = node.func.attr
        
        if method_name == 'add_edge':
            if len(node.args) != 2:
                self.errors.append(
                    f"add_edge requires exactly 2 arguments (source, target). "
                    f"Fix: graph.add_edge('source_node', 'target_node')"
                )
        
        elif method_name == 'add_conditional_edges':
            if len(node.args) < 2:
                self.errors.append(
                    f"add_conditional_edges requires at least 2 arguments (source, routing_function). "
                    f"Fix: graph.add_conditional_edges('node', routing_function) or "
                    f"graph.add_conditional_edges('node', routing_function, mapping_dict)"
                )
    
    def _validate_llm_definition(self, node: ast.Assign):
        """Validate LLM definition"""
        if not isinstance(node.value, ast.Call):
            self.errors.append("LLM must be defined as a function call (e.g., ChatOpenAI(...))")
            return
        
        if not (isinstance(node.value.func, ast.Name) and 
                node.value.func.id == 'ChatOpenAI'):
            self.warnings.append("Consider using ChatOpenAI for LLM definition")
    
    def _inherits_from_basemodel(self, node: ast.ClassDef) -> bool:
        """Check if class inherits from BaseModel"""
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == 'BaseModel':
                return True
        return False
    
    def validate_overall_structure(self):
        """Validate overall code structure"""
        # Check for required imports
        missing_imports = []
        for req_import in self.required_imports:
            if not any(req_import in imp for imp in self.imports):
                missing_imports.append(req_import)
        
        if missing_imports:
            self.errors.append(
                f"Missing required imports: {missing_imports}. "
                f"Fix: Add appropriate import statements for these components."
            )
        
        # Check for GraphState class
        if not self.graph_state_class:
            self.errors.append(
                "No GraphState class found. "
                "Fix: Define a class that inherits from MessagesState."
            )
        
        # Check for LLM definition
        if not self.llm_definition:
            self.errors.append(
                "No LLM definition found. "
                "Fix: Add 'llm = ChatOpenAI(model=\"gpt-4o-mini\", temperature=0)'"
            )
        
        # Check if it's a node function (returns dict or Command, takes state param)
        for node_name in self.node_functions:
            self._validate_node_function(self.functions[node_name])

        # Check for edges
        if not self.edge_calls:
            self.warnings.append("No edge definitions detected")
        
        # Check for compilation
        if not self.has_app_compile:
            self.errors.append(
                "No graph compilation found. "
                "Fix: Add 'app = workflow.compile(checkpointer=checkpointer)'"
            )
        
        if not self.has_checkpointer:
            self.warnings.append(
                "No checkpointer definition found. "
                "Consider adding: checkpointer = InMemorySaver()"
            )
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate validation report"""
        return {
            "errors": self.errors,
            "warnings": self.warnings,
            "summary": {
                "total_errors": len(self.errors),
                "total_warnings": len(self.warnings),
                "classes_found": len(self.classes),
                "functions_found": len(self.functions),
                "node_functions": len(self.node_functions),
                "edge_calls": len(self.edge_calls),
                "has_graph_state": self.graph_state_class is not None,
                "has_llm_definition": self.llm_definition is not None,
                "has_compilation": self.has_app_compile
            }
        }


def validate_langgraph_code(code: str) -> Dict[str, Any]:
    """
    Validate LangGraph code against format requirements.
    
    Args:
        code: Python code string to validate
        
    Returns:
        Dictionary containing validation results with errors and warnings
    """
    try:
        tree = ast.parse(code)
        validator = LangGraphFormatValidator()
        validator.visit(tree)
        validator.validate_overall_structure()
        return validator.generate_report()
    except SyntaxError as e:
        return {
            "errors": [f"Syntax Error: {e.msg} at line {e.lineno}"],
            "warnings": [],
            "summary": {"total_errors": 1, "total_warnings": 0}
        }


# Example usage and test
if __name__ == "__main__":
    # Test code with common issues
    test_code = '''
from langgraph.graph import MessagesState
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class GraphState(MessagesState):
    messages: str  # ERROR: Should not define messages explicitly
    data: any      # ERROR: Vague type

class BadModel(BaseModel):
    info: dict     # ERROR: Should not use dict in Pydantic models

def bad_node(state):
    result = state.field  # ERROR: Should use state['field']
    return {"result": result}  # ERROR: Missing messages key

llm = "not_a_function_call"  # ERROR: Should be function call
'''
    
    report = validate_langgraph_code(test_code)
    print("Validation Report:")
    print(f"Errors: {len(report['errors'])}")
    for error in report['errors']:
        print(f"  - {error}")
    print(f"Warnings: {len(report['warnings'])}")
    for warning in report['warnings']:
        print(f"  - {warning}")