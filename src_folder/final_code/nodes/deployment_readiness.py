from langchain_core.runnables import RunnableConfig
from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.utils.get_filtered_file import get_filtered_file
import ast

class StateInheritanceTransformer(ast.NodeTransformer):
    """
    An AST transformer that finds classes inheriting from 'MessagesState'
    and replaces the base class with 'CopilotKitState'. It also ensures
    that 'from copilotkit.langgraph import CopilotKitState' is present.
    """

    def __init__(self):
        self.transformation_made = False
        self.copilotkit_import_exists = False

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        # Check if the required import already exists
        if node.module == 'copilotkit.langgraph':
            if any(alias.name == 'CopilotKitState' for alias in node.names):
                self.copilotkit_import_exists = True
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        # Recursively visit children first to handle nested classes
        self.generic_visit(node)

        new_bases = []
        class_was_modified = False
        for base in node.bases:
            # Check for 'MessagesState' as a simple name (e.g., from ... import MessagesState)
            if isinstance(base, ast.Name) and base.id == 'MessagesState':
                # Replace with 'CopilotKitState'
                new_bases.append(ast.Name(id='CopilotKitState', ctx=base.ctx))
                self.transformation_made = True
                class_was_modified = True
            # Check for '...MessagesState' as an attribute (e.g., graph.MessagesState)
            elif isinstance(base, ast.Attribute) and base.attr == 'MessagesState':
                # Replace with 'CopilotKitState'
                new_bases.append(ast.Name(id='CopilotKitState', ctx=ast.Load()))
                self.transformation_made = True
                class_was_modified = True
            else:
                new_bases.append(base)

        if class_was_modified:
            node.bases = new_bases

        return node

    def visit_Module(self, node: ast.Module) -> ast.AST:
        # Visit all nodes in the module first to find imports and transform classes
        self.generic_visit(node)

        # If a transformation was made and the import doesn't exist, add it
        if self.transformation_made and not self.copilotkit_import_exists:
            new_import = ast.ImportFrom(
                module='copilotkit.langgraph',
                names=[ast.alias(name='CopilotKitState')],
                level=0
            )
            # Insert the new import at the beginning of the module body
            node.body.insert(0, new_import)
            ast.fix_missing_locations(node)

        return node


def refactor_code(source_code: str) -> str:
    """
    Parses Python code to find classes inheriting from MessagesState,
    replaces them with CopilotKitState, and adds the necessary import.
    """
    try:
        tree = ast.parse(source_code)
        transformer = StateInheritanceTransformer()
        new_tree = transformer.visit(tree)
        return ast.unparse(new_tree)
    except SyntaxError as e:
        return f"Error parsing code: {e}"


def deployment_readiness(state: AgentBuilderState, config: RunnableConfig):
    python_code = refactor_code(get_filtered_file(state["python_code"]))
    return { "python_code": python_code }