EXTRACT_IMPORT_NAMES="""
import ast
import sys

BUILTIN_MODULES = set(sys.stdlib_module_names)

def extract_import_names(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        source_code = f.read()
        if "from tools_code" in source_code:
            source_code = "\\n".join([line for line in source_code.splitlines() if "from tools_code" not in line])
    
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    python_imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                import_path = name.name
                if not import_path.startswith((".", "/")):
                    base_package = import_path.split(".")[0]
                    if base_package not in python_imports and base_package not in BUILTIN_MODULES:
                        python_imports.append(base_package)
        elif isinstance(node, ast.ImportFrom):
            if node.module and not node.module.startswith((".", "/")):
                base_package = node.module.split(".")[0]
                if base_package not in python_imports and base_package not in BUILTIN_MODULES:
                    python_imports.append(base_package)

    return python_imports

file_path = "./app.py"
imports = extract_import_names(file_path)
print("\\n".join(imports))
"""

def create_e2b_execution_command(
    *,
    execution_command: str = "python",
    install_req: bool = True
) -> str:
    if install_req:
        return (" && ").join(
        [
            f"echo '{EXTRACT_IMPORT_NAMES}' > extract_import_names.py",
            "export PIP_DISABLE_PIP_VERSION_CHECK=1",
            "python3 extract_import_names.py > openevals_requirements.txt",
            'pip install -r openevals_requirements.txt',
        ]
    )
    else:
        return (" && ").join(
            [
                f"echo '{EXTRACT_IMPORT_NAMES}' > extract_import_names.py",
                "export PIP_DISABLE_PIP_VERSION_CHECK=1",
                "python3 extract_import_names.py > requirements.txt",
            ]
        )