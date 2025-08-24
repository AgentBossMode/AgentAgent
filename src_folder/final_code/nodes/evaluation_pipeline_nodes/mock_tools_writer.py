from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.utils.create_react_agent_temp import create_react_agent
from final_code.llms.model_factory import get_model, ModelName
from final_code.nodes.tools.composio_info_tools import get_raw_tool_schema
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_customize_config
from final_code.states.ReactCopilotKitState import ReactCopilotState
from final_code.states.NodesAndEdgesSchemas import get_tools_info, get_nodes_and_edges_info
from src_folder.tests.validators_lib.validate_struct_output import validate_struct_output
from src_folder.tests.validators_lib.validate_ast_parse import validate_ast_parse
from src_folder.final_code.ast_visitors_lib.PydanticDictVisitor import PydanticDictVisitor
from final_code.utils.copilotkit_emit_status import append_in_progress_to_list, update_last_status
from typing import Any

async def fix_code_gen(final_code: str, config: RunnableConfig, errors: Any):

    FIX_PROMPT = """
You are a langgraph expert, user will provide you with a python code and a list of errors/warning with fixes. Your job is to make the fixes.
"""
    PYTHON_PROMPT = """
<python_code>
{python_code}
</python_code>
"""
    ERROR_REPORT = """
<error_report>
{fixes}
</error_report>
"""
    if len(errors) >0:
        llm = get_model()
        response = await llm.ainvoke(input=[SystemMessage(content=FIX_PROMPT),
                                             HumanMessage(content=PYTHON_PROMPT.format(python_code=final_code)),
                                             HumanMessage(content=ERROR_REPORT.format(fixes=errors))], config=config)
        final_code = response.content
    return final_code


async def mock_tools_writer(state: AgentBuilderState, config: RunnableConfig):
    MOCK_TOOLS_WRITER = """
You are provided the tools_code.py file
<tools_code.py>
{tools_code}
</tools_code.py>

<json_schema>
{json_schema}
</json_schema>

<tools_info>
{tools_info}
</tools_info>

<INSTRUCTIONS>
1. You will first write mock code stubs for composio tools and any method with the @tool decorator in <tools_code.py> section.
2. Identify the type of tool --> and identify docstring accordingly:
    a. In case of a composio tool --> Follow 'COMPOSIOMOCKINSTRUCTIONS' below:
    <COMPOSIOMOCKINSTRUCTION>
        Let's say you see the following composio tool being initialized

        tool_name = composio.tools.get(user_id=os.environ(\"USER_ID\"), tools=[\"TOOL_NAME_ABC\"])
    
        Instruction:
        1. call 'get_raw_tool_schema' tool, this will fetch information about the TOOL_NAME_ABC
        2. Now using this schema write a python function declaration as follows:
            def tool_name(required input parameters as per the schema output from step 1)
                \"\"\"
    Multiline PEP 257 Docstring
    Description based on the get_raw_tool_chema
    Also read the description from 'tools_info' and the 'json_schema' section, based on it add an example section in the docstring.
    Args:
        inputs: type

    Returns:
        # whatever output
    \"\"\"
    </COMPOSIOMOCKINSTRUCTION>
    b. In case of any method with @tool decorator --> Follow 'METHODMOCKINSTRUCTIONS' below:
    <METHODMOCKINSTRUCTIONS>
        Read the method docstring, analyze the code, and generate the doc string.
        Also read the description from 'tools_info' and the 'json_schema' section, based on it add an example section in the docstring.
    </METHODMOCKINSTRUCTIONS>
3. now use llm.with_structured_output to return mock data. Follow LLMSTRUCTOUTPUTINSTRUCTIONS below:
    <LLMSTRUCTOUTPUTINSTRUCTIONS>
        1. Use pydantic models for structured output.
        2. Use llm.with_structured_output to return the mock data.
        3. Use model_dump_json to return the output in JSON format.
        4. Follow the example below for structured output:

<EXAMPLE>
MOCK_TOOL_PROMPT = \"\"\"
You are a helpful assistant that generates mock data for tool outputs.
Given the tool's purpose and expected output, generate a realistic mock response.
\"\"\"

INPUT_PROMPT = \"\"\"
Tool Docstring: {{description}}
Input: {{input}}
Generate a mock output for this tool.
\"\"\"

@tool
def Web_Scraper_Read(url: str):
    \"\"\"
    PEP 257 docstring for the tool, generated in step 2 of INSTRUCTIONS.
    \"\"\"

    # NO USE OF DICT TYPE HINTS IN PYDANTIC MODELS, RECURSIVELY.
    class SkillInfo(BaseModel):
        skill_name: str = Field(description="Name of the skill or technology")
        proficiency: Optional[str] = Field(description="Proficiency level required for the skill")

    class JobDetails(BaseModel):
        job_title: str = Field(description="The full title of the job position")
        company: str = Field(description="Name of the hiring company")
        location: str = Field(description="Job location, including remote options")
        salary_range: Optional[str] = Field(description="Salary range if provided")
        required_skills: List[SkillInfo] = Field(description="List of required skills and technologies")
        posting_date: str = Field(description="When the job was posted")

    input_str = f"url: {{url}}"
    description = Web_Scraper_Read.__doc__
    
    result = llm.with_structured_output(JobDetails).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)
</EXAMPLE>

    </LLMSTRUCTOUTPUTINSTRUCTIONS>
4. Only relevant imports should remain, remove comosio.
5. Top of file add 
```python 
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```
</INSTRUCTIONS>

<OUTPUT>
You are supposed to generate a compilable python file with the mock code.
    <OUTPUT_FORMAT>
        - ONLY THE FINAL PYTHON CODE, NO MARKDOWN BLOCKS or explanations.
        - Code should be compilable python code without errors, no formatting errors
        - No SyntaxError
        - No unnecessary imports, if they are not used in the code
    </OUTPUT_FORMAT>
</OUTPUT>
<NOTE>
1. Make sure to use proper pydantic models for structured output.
2. DONOT use dict for struct output, instead make the type hint str and the Field description as "JSON string of the output".
3. Multiline PEP 257 docstring is a MUST for each generated tool.
4. In any class which inherits from BaseModel, do not use 'dict' as a type hint for any field.
</NOTE>
""" 
    customized_config= copilotkit_customize_config(config, emit_messages=False)
    app = create_react_agent(model= get_model(ModelName.GEMINI25FLASH),
    tools=[get_raw_tool_schema],
    name="mock_tools_writer",
    config_schema=customized_config,
    state_schema=ReactCopilotState)

    await append_in_progress_to_list(customized_config, state, "Writing a mock tools file...")
 
    new_state = state
    new_state["messages"] = [HumanMessage(content=MOCK_TOOLS_WRITER.format(tools_code=state["tools_code"], tools_info=get_tools_info(state["json_schema"].tools), json_schema=get_nodes_and_edges_info(state["json_schema"])))]
    final_response = await app.ainvoke(input=new_state)
    final_code = final_response["messages"][-1].content
    module = validate_ast_parse(final_code)
    errors = validate_struct_output(module)
    visitor = PydanticDictVisitor()
    visitor.visit(module)
    errors += visitor.errors

    final_code = await fix_code_gen(final_code, customized_config, errors)

    await update_last_status(customized_config, state, "Mock tools generated.", True)
    return {"mock_tools_code": final_code, "agent_status_list": state["agent_status_list"]}

