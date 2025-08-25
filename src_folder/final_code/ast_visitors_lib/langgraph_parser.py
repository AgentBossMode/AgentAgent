import ast
from typing import List, Dict, Set, Optional, Any
from collections import deque

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
        self.graph: Dict[str, List[str]] = {}
        self.conditional_edges: Set[str] = set()
        
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
        
        # Check for GraphState class - look for MessagesState inheritance
        if self._inherits_from_messages_state(node):
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
            if len(node.args) > 1 and isinstance(node.args[1], ast.Name):
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
    
    def _inherits_from_messages_state(self, node: ast.ClassDef) -> bool:
        """Check if class inherits from MessagesState"""
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == 'MessagesState':
                return True
        return False
    
    def _validate_state_class(self, node: ast.ClassDef):
        """Validate GraphState class definition"""
        # Check inheritance from MessagesState
        if not self._inherits_from_messages_state(node):
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
        
    def _validate_pydantic_model(self, node: ast.ClassDef):
        """Validate Pydantic model definitions"""
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                field_name = item.target.id
                annotation_node = item.annotation

                # Walk through the annotation to find any instance of 'dict' or 'Dict'
                for sub_node in ast.walk(annotation_node):
                    if isinstance(sub_node, ast.Name) and sub_node.id in ['dict', 'Dict']:
                        self.errors.append(
                            f"Pydantic model '{node.name}' field '{field_name}' uses a 'dict' type in annotation '{ast.unparse(annotation_node)}'. "
                            f"Fix: Use a specific Pydantic model for structured data, make changes in the code referencing this field to accomodate the change in type"
                        )
                        # Found an error for this field, no need to check further in the same annotation.
                        break
    
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

    def _extract_string_arg(self, arg_node: ast.expr) -> Optional[str]:
        if isinstance(arg_node, ast.Constant) and isinstance(arg_node.value, str):
            return arg_node.value
        if isinstance(arg_node, ast.Name) and arg_node.id in ["START", "END"]:
            return arg_node.id
        if isinstance(arg_node, ast.Name):
            return arg_node.id
        return None

    def _build_graph(self):
        """Build graph representation from edge calls"""
        self.graph = {}
        self.conditional_edges = set()

        # Initialize with all nodes
        for node in self.node_functions:
            self.graph[node] = []
        self.graph["START"] = []

        # Process edge calls
        for call in self.edge_calls:
            if call.func.attr == 'add_edge' and len(call.args) >= 2:
                source = self._extract_string_arg(call.args[0])
                target = self._extract_string_arg(call.args[1])
                
                if source and target:
                    if source not in self.graph:
                        self.graph[source] = []
                    self.graph[source].append(target)

            elif call.func.attr == 'add_conditional_edges' and len(call.args) >= 2:
                source = self._extract_string_arg(call.args[0])
                if source:
                    self.conditional_edges.add(source)
                    if source not in self.graph:
                        self.graph[source] = []
                    
                    # Check if there's a mapping dict
                    has_mapping = len(call.args) > 2 and isinstance(call.args[2], ast.Dict)
                    if has_mapping:
                        mapping_dict = call.args[2]
                        for value_node in mapping_dict.values:
                            target = self._extract_string_arg(value_node)
                            if target:
                                self.graph[source].append(target)
                    else:
                        # No mapping dict - we can't analyze this statically
                        self.warnings.append(
                            f"Cannot statically analyze conditional edges from '{source}'. "
                            f"State validation may be incomplete."
                        )

    def _analyze_node_state_io(self, func_def: ast.FunctionDef) -> Dict[str, Set[str]]:
        """Analyze what state variables a node accesses and assigns"""
        accessed = set()
        assigned = set()
        
        class StateAnalyzer(ast.NodeVisitor):
            def visit_Subscript(self, node):
                # Looking for state["key"] access
                if (isinstance(node.value, ast.Name) and 
                    node.value.id == 'state' and
                    isinstance(node.slice, ast.Constant) and 
                    isinstance(node.slice.value, str)):
                    accessed.add(node.slice.value)
                self.generic_visit(node)
            
            def visit_Return(self, node):
                # Looking for return {"key": value} assignments  
                if isinstance(node.value, ast.Dict):
                    for key in node.value.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            assigned.add(key.value)
                self.generic_visit(node)
        
        analyzer = StateAnalyzer()
        analyzer.visit(func_def)
        
        return {"accessed": accessed, "assigned": assigned}

    def _get_initial_state_vars(self) -> Set[str]:
        """Get state variables that have default values"""
        initial_vars = set()
        if self.graph_state_class:
            for item in self.graph_state_class.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    # Only consider variables with default values as initially available
                    if item.value is not None:
                        initial_vars.add(item.target.id)
        return initial_vars

    def _validate_state_flow(self):
        """Validate that state variables are properly assigned before access"""
        self._build_graph()
        
        # Get initial state variables
        initial_vars = self._get_initial_state_vars()
        
        # Analyze each node's state usage
        node_io = {}
        for node_name in self.node_functions:
            if node_name in self.functions:
                node_io[node_name] = self._analyze_node_state_io(self.functions[node_name])
        
        # Check each node for improper state access
        for node_name, io in node_io.items():
            for accessed_var in io['accessed']:
                # Skip built-in state vars and initially available vars
                if accessed_var in initial_vars or accessed_var == 'messages':
                    continue
                
                # Check if this variable is guaranteed to be assigned before this node
                if not self._is_var_guaranteed_before_node(node_name, accessed_var, node_io):
                    # Find a path where the variable is not assigned
                    problematic_path = self._find_unassigned_path(node_name, accessed_var, node_io)
                    if problematic_path:
                        path_str = " -> ".join(problematic_path)
                        self.errors.append(
                            f"In node '{node_name}', state variable '{accessed_var}' is accessed but might not be assigned. "
                            f"There is a path from START to '{node_name}' that does not guarantee '{accessed_var}' is assigned: {path_str}"
                        )

    def _is_var_guaranteed_before_node(self, target_node: str, var: str, node_io: Dict) -> bool:
        """Check if variable is guaranteed to be assigned on ALL paths to target_node"""
        # BFS to explore all paths from START to target_node
        queue = deque([("START", set())])  # (current_node, assigned_vars_so_far)
        visited = set()
        
        while queue:
            current_node, assigned_vars = queue.popleft()
            
            # Avoid infinite loops
            state_key = (current_node, frozenset(assigned_vars))
            if state_key in visited:
                continue
            visited.add(state_key)
            
            # If we reached target node, check if var is assigned
            if current_node == target_node:
                if var not in assigned_vars:
                    return False  # Found path where var is not guaranteed
                continue
            
            # Add variables assigned by current node
            new_assigned_vars = assigned_vars.copy()
            if current_node in node_io:
                new_assigned_vars.update(node_io[current_node]['assigned'])
            
            # Continue to successors
            for successor in self.graph.get(current_node, []):
                queue.append((successor, new_assigned_vars))
        
        return True

    def _find_unassigned_path(self, target_node: str, var: str, node_io: Dict) -> Optional[List[str]]:
        """Find a path from START to target_node where var is not assigned"""
        queue = deque([("START", ["START"], set())])  # (node, path, assigned_vars)
        visited = set()
        
        while queue:
            current_node, path, assigned_vars = queue.popleft()
            
            state_key = (current_node, frozenset(assigned_vars))
            if state_key in visited:
                continue
            visited.add(state_key)
            
            # If we reached target and var is not assigned, return this path
            if current_node == target_node:
                if var not in assigned_vars:
                    return path
                continue
            
            # Add variables assigned by current node
            new_assigned_vars = assigned_vars.copy()
            if current_node in node_io:
                new_assigned_vars.update(node_io[current_node]['assigned'])
            
            # Continue to successors
            for successor in self.graph.get(current_node, []):
                new_path = path + [successor]
                queue.append((successor, new_path, new_assigned_vars))
        
        return None

    def validate_overall_structure(self):
        """Validate overall code structure"""
        # Check for required imports
        missing_imports = []
        for req_import in self.required_imports:
            if not any(req_import in imp for imp in self.imports):
                missing_imports.append(req_import)
        
        if missing_imports:
            missing_imports = sorted(list(missing_imports))
            error_msg = f"Missing required imports: {missing_imports}. Fix: Add appropriate import statements for these components."
            self.errors.append(error_msg)
        
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
            if node_name in self.functions:
                self._validate_node_function(self.functions[node_name])
            else:
                self.warnings.append(f"Node '{node_name}' is added to the graph but the function implementation is not found.")

        # Check for edges
        if not self.edge_calls:
            self.errors.append("No edge definitions detected")
        
        self._validate_state_flow()
        
        # Check for compilation
        if not self.has_app_compile:
            self.errors.append(
                "No graph compilation found. "
                "Fix: Add 'app = workflow.compile(checkpointer=checkpointer)'"
            )
        
        if not self.has_checkpointer:
            self.errors.append(
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