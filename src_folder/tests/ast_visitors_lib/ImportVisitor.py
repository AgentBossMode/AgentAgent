import ast

class ImportVisitor(ast.NodeVisitor):
    def __init__(self, methods_to_check: list[tuple[str, str]]):
        self.methods_to_check = {method: imp for method, imp in methods_to_check}
        self.called_methods = set()
        self.found_imports = set()

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in self.methods_to_check:
            self.called_methods.add(node.func.id)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            full_import = f"import {alias.name}"
            if full_import in self.methods_to_check.values():
                self.found_imports.add(full_import)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            full_import = f"from {node.module} import {alias.name}"
            if full_import in self.methods_to_check.values():
                self.found_imports.add(full_import)
        self.generic_visit(node)

    def check_imports(self):
        missing_imports = set()
        for method in self.called_methods:
            expected_import = self.methods_to_check.get(method)
            if expected_import not in self.found_imports:
                missing_imports.add((method, expected_import))
        
        unnecessary_imports = set()
        for imp in self.found_imports:
            method_found = False
            for method, expected_import in self.methods_to_check.items():
                if imp == expected_import and method in self.called_methods:
                    method_found = True
                    break
            if not method_found:
                for method, expected_import in self.methods_to_check.items():
                    if imp == expected_import:
                        unnecessary_imports.add((method, imp))
                        break

        return missing_imports, unnecessary_imports

    def get_results(self):
        missing_imports, unnecessary_imports = self.check_imports()
        
        return {
            "called_methods": list(self.called_methods),
            "found_imports": list(self.found_imports),
            "missing_imports": list(missing_imports),
            "unnecessary_imports": list(unnecessary_imports)
        }