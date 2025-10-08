from langchain_core.runnables import RunnableConfig
from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.utils.get_filtered_file import get_filtered_file
import ast
import traceback
from langchain_core.messages import AIMessage
from langgraph.types import Command
from typing import Literal

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

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # Recursively visit children first
        self.generic_visit(node)

        # Check for `*.compile()` calls and remove `checkpointer` keyword argument
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'compile':
            original_keyword_count = len(node.keywords)
            node.keywords = [kw for kw in node.keywords if kw.arg != 'checkpointer']
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


class ToolVarCollector(ast.NodeVisitor):
    """
    A NodeVisitor to collect variable names used in the 'tools'
    parameter of `create_react_agent` calls. This is the first pass.
    """
    def __init__(self):
        self.react_agent_tool_vars = set()

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == 'create_react_agent':
            for kw in node.keywords:
                if kw.arg == 'tools' and isinstance(kw.value, ast.Name):
                    self.react_agent_tool_vars.add(kw.value.id)
        self.generic_visit(node)



class ToolsListTransformer(ast.NodeTransformer):
    """
    An AST transformer that converts tool list definitions from using list
    literals to using addition for concatenation. This is the second pass.
    e.g., `my_tools = [tool1, tool2]` becomes `my_tools = tool1 + tool2`.
    """
    def __init__(self, tool_vars_to_transform: set[str]):
        self.tool_vars_to_transform = tool_vars_to_transform
        self.transformation_made = False

    def _build_addition_chain(self, elements: list[ast.expr]) -> ast.expr:
        """Recursively builds a chain of ast.BinOp(op=ast.Add) nodes."""
        if not elements:
            # This case should ideally not be hit if we check len > 1
            return ast.List(elts=[], ctx=ast.Load())
        if len(elements) == 1:
            return elements[0]
        
        left_node = self._build_addition_chain(elements[:-1])
        return ast.BinOp(left=left_node, op=ast.Add(), right=elements[-1])

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        # Check for single target assignment, e.g., `my_tools = ...`
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id
            # If this variable is one of our tool variables and is assigned a list
            if target_name in self.tool_vars_to_transform and isinstance(node.value, ast.List) and len(node.value.elts) > 0:
                if len(node.value.elts) == 1:
                    # If only one element, just assign that element directly
                    node.value = node.value.elts[0]
                else:
                    node.value = self._build_addition_chain(node.value.elts)
                self.transformation_made = True
                ast.fix_missing_locations(node)
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # Handle inline list definitions, e.g., `tools=[t1, t2]`
        if isinstance(node.func, ast.Name) and node.func.id == 'create_react_agent':
            for kw in node.keywords:
                if kw.arg == 'tools' and isinstance(kw.value, ast.List) and len(kw.value.elts) > 0:
                    if len(kw.value.elts) == 1:
                        # If only one element, just assign that element directly
                        kw.value = kw.value.elts[0]
                    kw.value = self._build_addition_chain(kw.value.elts)
                    self.transformation_made = True
                    ast.fix_missing_locations(node)
        return self.generic_visit(node)



def refactor_code(source_code: str) -> str:
    """
    Parses Python code to find classes inheriting from MessagesState,
    replaces them with CopilotKitState, and adds the necessary import.
    """
    try:
        tree = ast.parse(source_code)
        transformer = StateInheritanceTransformer()
        new_tree = transformer.visit(tree)
        # transformer = ToolVarCollector()
        # transformer.visit(new_tree)
        # transformer = ToolsListTransformer(transformer.react_agent_tool_vars)
        # new_tree = transformer.visit(new_tree)
        return ast.unparse(new_tree)
    except SyntaxError as e:
        return f"Error parsing code: {e}"
    except Exception as e:
        return f"Error refactoring code: {e}"


def deployment_readiness(state: AgentBuilderState, config: RunnableConfig) -> Command[Literal["__end__"]]:
    try:
        python_code = refactor_code(get_filtered_file(state["python_code"]))
        return Command(
            goto="__end__",
            update={"python_code": python_code}
        )
    except Exception as e:
        return Command(
            goto="__end__",
            update={
                "exception_caught": f"{e}\n{traceback.format_exc()}",
                "messages": [AIMessage(content="An error occurred during deployment readiness preparation. Please try again.")]
            }
        )