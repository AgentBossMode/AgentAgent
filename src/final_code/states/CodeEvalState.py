from copilotkit import CopilotKitState
from typing import List
from src.final_code.states.DryRunState import UseCaseAnalysis
from pydantic import Field
from typing import Literal

class CodeEvalState(CopilotKitState):
    reflection_code: str
    python_code: str = Field(description="The Python code to evaluate")
    json_dict: str
    use_cases: List[UseCaseAnalysis]
    pytest_code: str = Field(description="The pytest file")
    mock_tools_code: str = Field(description="The mock tools file")
    tools_code: str = Field(description="The tools file")
    packages: list[str]
    pytest_results: str = Field(description="The pytest results")
    issue_type: Literal["syntax_error", "runtime_error", "assertion_fail"] = Field(description="identify the type of issue")
    file_that_needs_fixes: Literal["python_code", "mock_tools_code", "pytest_code"] = Field(description="identify the file to fix")
    fix_needed: str = Field(description="detailed explanation of fixes needed, in a diff format")
