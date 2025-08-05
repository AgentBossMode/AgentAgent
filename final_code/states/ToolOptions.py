from pydantic import BaseModel, Field
from typing import List
class PythonSdkInfo(BaseModel):
    tool_sdk: str = Field(description="the python sdk identified")
    url: str = Field(description="the url for reference")
    code_implementation: str = Field("the python sdk code implementation from the url")

class NativeTool(BaseModel):
    tool_name: str = Field(description="The tool_name provided by the user")
    tools_dentified: List[PythonSdkInfo] = Field(description="The list of python sdks identified")

class ToolOptions(BaseModel):
    native_tools: List[NativeTool] = Field(description="List of the the tool_names along with python sdks identified")
