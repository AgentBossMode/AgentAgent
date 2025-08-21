#!/usr/bin/env python3
"""
LangGraph Code Validation Script

This script validates LangGraph implementations against the specified format requirements
using AST analysis to catch common issues and ensure code quality.
"""

import ast
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List
import json

# Import the validator (assuming it's in the same directory)
from src_folder.final_code.ast_visitors_lib.langgraph_parser import validate_langgraph_code, LangGraphFormatValidator


def validate_file(file_path: str) -> Dict[str, Any]:
    """Validate a Python file containing LangGraph code"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        return validate_langgraph_code(code)
    except FileNotFoundError:
        return {
            "errors": [f"File not found: {file_path}"],
            "warnings": [],
            "summary": {"total_errors": 1, "total_warnings": 0}
        }
    except Exception as e:
        return {
            "errors": [f"Error reading file {file_path}: {str(e)}"],
            "warnings": [],
            "summary": {"total_errors": 1, "total_warnings": 0}
        }


def print_validation_report(report: Dict[str, Any], file_path: str = None):
    """Print formatted validation report"""
    print("=" * 60)
    if file_path:
        print(f"VALIDATION REPORT FOR: {file_path}")
    else:
        print("VALIDATION REPORT")
    print("=" * 60)
    
    summary = report.get("summary", {})
    errors = report.get("errors", [])
    warnings = report.get("warnings", [])
    
    print(f"📊 SUMMARY:")
    print(f"   Total Errors: {summary.get('total_errors', 0)}")
    print(f"   Total Warnings: {summary.get('total_warnings', 0)}")
    print(f"   Classes Found: {summary.get('classes_found', 0)}")
    print(f"   Functions Found: {summary.get('functions_found', 0)}")
    print(f"   Node Functions: {summary.get('node_functions', 0)}")
    print(f"   Edge Calls: {summary.get('edge_calls', 0)}")
    print(f"   Has GraphState: {summary.get('has_graph_state', False)}")
    print(f"   Has LLM Definition: {summary.get('has_llm_definition', False)}")
    print(f"   Has Compilation: {summary.get('has_compilation', False)}")
    
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for i, error in enumerate(errors, 1):
            print(f"   {i}. {error}")
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for i, warning in enumerate(warnings, 1):
            print(f"   {i}. {warning}")
    
    if not errors and not warnings:
        print("\n✅ All validations passed!")
    elif not errors:
        print("\n✅ No errors found, only warnings.")
    else:
        print(f"\n❌ Validation failed with {len(errors)} error(s).")
    
    print("=" * 60)


def validate_sample_codes():
    """Validate various sample codes to demonstrate the validator"""
    
    samples = {
        "Good Code": '''
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GraphState(MessagesState):
    query: str
    result: Optional[str] = None

class OutputModel(BaseModel):
    response: str = Field(description="Generated response")

def process_node(state: GraphState) -> GraphState:
    """Process user query and generate response"""
    structured_llm = llm.with_structured_output(OutputModel)
    result = structured_llm.invoke(state["messages"][-1].content)
    return {
        "messages": [AIMessage(content=result.response)],
        "result": result.response
    }

workflow = StateGraph(GraphState)
workflow.add_node("process", process_node)
workflow.add_edge(START, "process")
workflow.add_edge("process", END)

checkpointer = InMemorySaver()
app = workflow.compile(checkpointer=checkpointer)
        ''',
        
        "Bad Code with Multiple Issues": '''
from langgraph.graph import MessagesState

class GraphState(MessagesState):
    messages: str  # ERROR: Should not define messages explicitly
    data: any      # ERROR: Vague type

class BadModel:  # ERROR: Should inherit from BaseModel
    info: dict   # ERROR: Should not use dict

def bad_node(state):
    result = state.field  # ERROR: Should use state['field']
    return {"result": result}  # ERROR: Missing messages key

# ERROR: No LLM definition
# ERROR: No graph construction
# ERROR: No compilation
        ''',
        
        "Code with Tool Agent": '''
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class ReactAgentState(MessagesState):
    remaining_steps: int
    structured_response: str

class SearchResult(BaseModel):
    query: str = Field(description="Search query")
    results: List[str] = Field(description="Search results")

def search_node(state: ReactAgentState) -> ReactAgentState:
    """Node using react agent with tools"""
    tools = []  # Assume tools are defined elsewhere
    
    agent = create_react_agent(
        model=llm,
        prompt="Search for information based on user query",
        tools=tools,
        state_schema=ReactAgentState,
        response_format=SearchResult
    )
    
    result = agent.invoke({"messages": state["messages"]})["structured_response"]
    
    return {
        "messages": [AIMessage(content="Search completed")],
        "structured_response": result
    }

workflow = StateGraph(ReactAgentState)
workflow.add_node("search", search_node)
workflow.add_edge("__start__", "search")
workflow.add_edge("search", "__end__")

app = workflow.compile()
        '''
    }
    
    print("🧪 VALIDATING SAMPLE CODES:")
    print("=" * 80)
    
    for name, code in samples.items():
        print(f"\n📝 Sample: {name}")
        report = validate_langgraph_code(code)
        print_validation_report(report)


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Validate LangGraph code against format requirements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate.py my_langgraph.py
  python validate.py --json my_langgraph.py > report.json
  python validate.py --samples
        """
    )
    
    parser.add_argument(
        'file',
        nargs='?',
        help='Python file to validate'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output report in JSON format'
    )
    
    parser.add_argument(
        '--samples',
        action='store_true',
        help='Validate sample codes for demonstration'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only show errors, suppress warnings'
    )
    
    args = parser.parse_args()
    
    if args.samples:
        validate_sample_codes()
        return
    
    if not args.file:
        parser.print_help()
        return
    
    # Validate the specified file
    report = validate_file(args.file)
    
    if args.quiet:
        # Remove warnings in quiet mode
        report['warnings'] = []
        report['summary']['total_warnings'] = 0
    
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_validation_report(report, args.file)
    
    # Exit with error code if there are errors
    if report.get('summary', {}).get('total_errors', 0) > 0:
        sys.exit(1)


class DetailedLangGraphValidator(LangGraphFormatValidator):
    """
    Extended validator with more detailed checks for specific LangGraph patterns
    """
    
    def __init__(self):
        super().__init__()
        self.react_agent_patterns = []
        self.structured_output_patterns = []
        self.interrupt_patterns = []
        self.command_patterns = []
    
    def visit_Call(self, node: ast.Call):
        super().visit_Call(node)
        
        # Check for create_react_agent usage
        if (isinstance(node.func, ast.Name) and 
            node.func.id == 'create_react_agent'):
            self._validate_react_agent_call(node)
        
        # Check for structured output usage
        if (isinstance(node.func, ast.Attribute) and 
            node.func.attr == 'with_structured_output'):
            self._validate_structured_output_call(node)
        
        # Check for interrupt usage
        if (isinstance(node.func, ast.Name) and 
            node.func.id == 'interrupt'):
            self._validate_interrupt_call(node)
    
    def _validate_react_agent_call(self, node: ast.Call):
        """Validate create_react_agent call structure"""
        state_schema_structure= """
<STATE_SCHEMA>
class ReactAgentState(MessagesState):
    remaining_steps: int
    structured_response: any
</STATE_SCHEMA>
        """
        # Check for required parameters
        required_params = {'model', 'tools', 'state_schema', 'response_format'}
        
        # Get keyword argument names
        provided_params = set()
        for keyword in node.keywords:
            if keyword.arg:
                provided_params.add(keyword.arg)
        
        missing_params = required_params - provided_params
        if missing_params:
            self.errors.append(
                f"create_react_agent missing required parameters: {missing_params}. "
                f"Fix: Add the missing parameters to your create_react_agent call."
            )
        
        # Check response_format is a Pydantic model (not dict or list)
        for keyword in node.keywords:
            if keyword.arg == 'state_schema':
                    if isinstance(keyword.value, ast.Name):
                        state_schema_name = keyword.value.id
                        if state_schema_name != "ReactAgentState":
                            self.errors.append(f"create_react_agent state_schema must be ReactAgentState and not {state_schema_name}. \n {state_schema_structure} ")

                        if state_schema_name in self.classes:
                            state_schema_node = self.classes[state_schema_name]
                            inherits_message_state = False
                            for node in state_schema_node.bases:
                                if isinstance(node, ast.Name) and node.id == 'MessagesState':
                                    inherits_message_state = True
                                    break
                            if not inherits_message_state:
                                self.errors.append(
                                    f"create_react_agent state_schema '{state_schema_name}' must inherit from MessagesState. {state_schema_structure}"
                                )
                            # Check for remaining_steps and structured_response fields
                            has_remaining_steps = False
                            has_structured_response = False
                            for item in state_schema_node.body:
                                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                                    if item.target.id == 'remaining_steps':
                                        has_remaining_steps = True
                                    if item.target.id == 'structured_response':
                                        has_structured_response = True
                            if not has_remaining_steps:
                                self.errors.append(
                                    f"create_react_agent state_schema '{state_schema_name}' must include 'remaining_steps: int'. {state_schema_structure}"
                                )
                            if not has_structured_response:
                                self.errors.append(
                                    f"create_react_agent state_schema '{state_schema_name}' must include 'structured_response: any'. {state_schema_structure}"
                                )
                        else:
                            self.errors.append(
                                f"create_react_agent state_schema '{state_schema_name}' is not a defined class. {state_schema_structure}"
                            )
                    else:
                        self.errors.append(
                            f"create_react_agent state_schema must be a class name. {state_schema_structure}"
                        )
            if keyword.arg == 'response_format':
                if isinstance(keyword.value, ast.Name):
                    # Good - using a class name
                    continue
                elif isinstance(keyword.value, ast.Subscript):
                    # Check if it's List[SomeModel] which is not allowed
                    if (isinstance(keyword.value.value, ast.Name) and 
                        keyword.value.value.id == 'List'):
                        self.errors.append(
                            "create_react_agent response_format should not be List[Model]. "
                            "Fix: Use a single Pydantic model instead of a list."
                        )
                else:
                    self.errors.append(
                        "create_react_agent response_format should be a Pydantic model class. "
                        "Fix: Define a proper Pydantic model and use it as response_format."
                    )
    
    def _validate_structured_output_call(self, node: ast.Call):
        """Validate structured output usage"""
        if not node.args:
            self.errors.append(
                "with_structured_output requires a Pydantic model argument. "
                "Fix: llm.with_structured_output(YourPydanticModel)"
            )
    
    def _validate_interrupt_call(self, node: ast.Call):
        """Validate interrupt usage patterns"""
        if not node.args:
            self.errors.append(
                "interrupt() should include data to surface to human. "
                "Fix: interrupt({{'prompt': 'message', 'data': value}})"
            )


def run_detailed_validation(code: str) -> Dict[str, Any]:
    """Run validation with detailed LangGraph-specific checks"""
    try:
        tree = ast.parse(code)
        validator = DetailedLangGraphValidator()
        validator.visit(tree)
        validator.validate_overall_structure()
        return validator.generate_report()
    except SyntaxError as e:
        return {
            "errors": [f"Syntax Error: {e.msg} at line {e.lineno}"],
            "warnings": [],
            "summary": {"total_errors": 1, "total_warnings": 0}
        }


# Additional utility functions
def check_environment_variables(code: str) -> List[str]:
    """Extract required environment variables from code"""
    env_vars = []
    
    # Common patterns for environment variables
    patterns = [
        r'os\.environ\[[\'"](.*?)[\'"]\]',
        r'os\.getenv\([\'"](.*?)[\'"]',
        r'getenv\([\'"](.*?)[\'"]',
    ]
    
    import re
    for pattern in patterns:
        matches = re.findall(pattern, code)
        env_vars.extend(matches)
    
    return list(set(env_vars))


def suggest_improvements(report: Dict[str, Any]) -> List[str]:
    """Generate improvement suggestions based on validation report"""
    suggestions = []
    
    summary = report.get('summary', {})
    
    if not summary.get('has_graph_state'):
        suggestions.append(
            "Define a GraphState class that inherits from MessagesState to manage workflow state"
        )
    
    if not summary.get('has_llm_definition'):
        suggestions.append(
            "Add LLM definition: llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)"
        )
    
    if summary.get('node_functions', 0) == 0:
        suggestions.append(
            "Define node functions that take state parameter and return state updates"
        )
    
    if summary.get('edge_calls', 0) == 0:
        suggestions.append(
            "Add edges to connect your nodes using add_edge() or add_conditional_edges()"
        )
    
    if not summary.get('has_compilation'):
        suggestions.append(
            "Compile your workflow: app = workflow.compile(checkpointer=checkpointer)"
        )
    
    return suggestions


if __name__ == "__main__":
    main()