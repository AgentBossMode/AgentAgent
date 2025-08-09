from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.utils.create_react_agent_temp import create_react_agent
from final_code.llms.model_factory import get_model, ModelName
from final_code.nodes.tools.composio_info_tools import get_raw_tool_schema
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_customize_config, copilotkit_emit_state
from final_code.states.ReactCopilotKitState import ReactCopilotState
from final_code.prompt_lib.node_info.struct_output import struct_output
async def mock_test_writer(state: AgentBuilderState, config: RunnableConfig):
    MOCK_TEST_WRITER = """
You are provided the tools_code.py file
<tools_code.py>
{tools_code}
</tools_code.py>

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
                \"\"\"Docstring including what the tool does, as per the get_raw_tool_schema output \"\"\"
    </COMPOSIOMOCKINSTRUCTION>
    b. In case of any method with @tool decorator --> Follow 'METHODMOCKINSTRUCTIONS' below:
    <METHODMOCKINSTRUCTIONS>
        Read the method docstring, analyze the code, and generate the code again but with mock implementation.
        Remove any related imports, any initializations done to support the tool etc.
    </METHODMOCKINSTRUCTIONS>
3. now use llm.with_structured_output to return mock data, refer to STRUCT_OUTPUT section.
4. Only relevant imports should remain, remove comosio imports, langchain import like tools are fine.
5. Top of file add 
```python 
from langchain_openai import ChatOpenAI
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
    Extracts specific data points like company name, salary, skills, and job type from various job posting websites.

    Args:
        url (str): The URL of the job posting to scrape.

    Returns:
        dict: A dictionary containing extracted job details such as job_title, company, location, salary_range, required_skills, and posting_date.
    \"\"\"
    class JobDetails(BaseModel):
        job_title: str = Field(description="The full title of the job position")
        company: str = Field(description="Name of the hiring company")
        location: str = Field(description="Job location, including remote options")
        salary_range: Optional[str] = Field(description="Salary range if provided")
        required_skills: List[str] = Field(description="List of required skills and technologies")
        posting_date: str = Field(description="When the job was posted")

    input_str = f"url: {{url}}"
    description = \"\"\"
    Extracts specific data points like company name, salary, skills, and job type from various job posting websites.

     Args:
        url (str): The URL of the job posting to scrape.

    Returns:
    Job details such as job_title, company, location, salary_range, required_skills, and posting_date.
    \"\"\"
    
    result = llm.with_structured_output(JobDetails).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)
</EXAMPLE
""" + f"""
<STRUCT_OUTPUT>
{struct_output}
</STRUCT_OUTPUT>
"""
    customized_config= copilotkit_customize_config(config, emit_messages=False)
    app = create_react_agent(model= get_model(ModelName.GEMINI25FLASH),
    tools=[get_raw_tool_schema],
    name="mock_test_writer",
    config_schema=customized_config,
    state_schema=ReactCopilotState)
    state["current_status"] = {"inProcess":True ,"status": "Writing a mock tools file.."}
    await copilotkit_emit_state(state=state, config=customized_config)
    new_state = state
    new_state["messages"] = [HumanMessage(content=MOCK_TEST_WRITER.format(tools_code=state["tools_code"]))]
    final_response = await app.ainvoke(input=new_state)
    state["current_status"] = {"inProcess":False ,"status": "Mock tools generated."}
    await copilotkit_emit_state(state=state, config=customized_config)
    return {"mock_tools_code": final_response["messages"][-1].content, "messages": [AIMessage(content="Mock tools have been generated (mocked_tools.py)")]}

