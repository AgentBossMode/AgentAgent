
import ast
import unittest
from src_folder.tests.ast_visitors_lib.ImportVisitor import ImportVisitor


methods_to_check = [
            ("create_react_agent", "from langgraph.prebuilt import create_react_agent"),
        ]

def test_success_case_all_present():
    code = """
from langgraph.prebuilt import create_react_agent
from y import abc

result = create_react_agent(None, None)
res2 = abc(1)
"""
    tree = ast.parse(code)
    visitor = ImportVisitor(methods_to_check)
    visitor.visit(tree)
    results = visitor.get_results()
    assert len(results["missing_imports"]) == 0
    assert len(results["unnecessary_imports"]) == 0

def test_success_case_one_present():
    code = """
from langgraph.prebuilt import create_react_agent

result = create_react_agent(None, None)
"""
    tree = ast.parse(code)
    visitor = ImportVisitor(methods_to_check)
    visitor.visit(tree)
    results = visitor.get_results()
    assert len(results["missing_imports"]) == 0
    assert len(results["unnecessary_imports"]) == 0

def test_success_case_none_present():
    code = """
print('hello')
"""
    tree = ast.parse(code)
    visitor = ImportVisitor(methods_to_check)
    visitor.visit(tree)
    results = visitor.get_results()
    assert len(results["missing_imports"]) == 0
    assert len(results["unnecessary_imports"]) == 0

def test_missing_import():
    code = """
result = create_react_agent(None, None)
"""
    tree = ast.parse(code)
    visitor = ImportVisitor(methods_to_check)
    visitor.visit(tree)
    results = visitor.get_results()
    assert len(results["missing_imports"]) == 1
    assert len(results["unnecessary_imports"]) == 0

def test_unnecessary_import():
    code = """
from langgraph.prebuilt import create_react_agent
"""
    tree = ast.parse(code)
    visitor = ImportVisitor(methods_to_check)
    visitor.visit(tree)
    results = visitor.get_results()
    assert len(results["missing_imports"]) == 0
    assert len(results["unnecessary_imports"]) == 1
    assert ("create_react_agent", "from langgraph.prebuilt import create_react_agent") in results["unnecessary_imports"]