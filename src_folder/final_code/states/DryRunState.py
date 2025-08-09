from pydantic import BaseModel, Field
from typing import List
from final_code.states.NodesAndEdgesSchemas import JSONSchema

class UseCaseAnalysis(BaseModel):
    name: str = Field(description="Name of the use case")
    description: str = Field(description="Description of the use case")
    dry_run: str = Field(description="Dry run of the use case, which is a string representation of the dry run results")

class DryRunResults(BaseModel):
    use_cases: List[UseCaseAnalysis] = Field(default_factory=list,description="List of use cases with their names, descriptions, and dry runs.")
    updated_json_schema: JSONSchema | None = Field(default=None, description="Updated JSON schema if dry run fails")
    justification: str | None = Field(default=None, description="Justification for updated JSON schema if applicable")
