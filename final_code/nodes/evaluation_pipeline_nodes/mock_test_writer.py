from final_code.states.CodeEvalState import CodeEvalState
from final_code.utils.create_react_agent_temp import create_react_agent
from final_code.llms.model_factory import get_model, ModelName
from final_code.nodes.tools.composio_info_tools import get_raw_tool_schema
from langchain_core.messages import HumanMessage
def mock_test_writer(state: CodeEvalState):
    MOCK_TEST_WRITER = """
You are provided the tools_code.py file
<tools_code.py>
{tools_code}
</tools_code.py>

<INSTRUCTIONS>
1. You will first write mock code stubs for composio tools and any method with the @tool decorator in <tools_code.py> section.
    a. In case of a composio tool --> Follow 'COMPOSIOMOCKINSTRUCTIONS' below:
    <COMPOSIOMOCKINSTRUCTION>
        Let's say you see the following composio tool being initialized

        tool_name = composio.tools.get(user_id=os.environ(\"USER_ID\"), tools=[\"TOOL_NAME_ABC\"])
    
        Instruction:
        1. call 'get_raw_tool_schema' tool, this will fetch information about the TOOL_NAME_ABC
        2. Now using this schema write a python function as follows:
            def tool_name(required input parameters as per the schema output from step 1)
                \"\"\"Docstring including what the tool does, as per the get_raw_tool_schema output \"\"\"
                logic that mocks the tasks of the tool and returns output as per the schema output from step 1 ...
    </COMPOSIOMOCKINSTRUCTION>
    b. In case of any method with @tool decorator --> Follow 'METHODMOCKINSTRUCTIONS' below:
    <METHODMOCKINSTRUCTIONS>
        Read the method docstring, analyze the code, and generate the code again but with mock implementation.
        Remove any related imports, any initializations done to support the tool etc.

Example:
Lets say tool looks like this:    
from import1 import module1

@tool
def SimulationEngineTool(N: int, width: int, height: int) -> dict:
    class RandomClass(Agent):
        def __init__(self, unique_id, model):
            super().__init__(unique_id, model)

        def step(self):
            pass  # Define agent behavior here

    class RandomModel(module1):
        # some model implementation

        def step(self):
            self.datacollector.collect(self)
            self.schedule.step()
    
    # Simulate a basic run and return dummy results for demonstration
    model = RandomModel(N, width, height)
    # some invocation to this model and returning the results
    
    
Mocked output:
@tool
def SimulationEngineTool(N: int, width: int, height: int) -> dict:
    # In a real scenario, this would return actual simulation data
    return {{"metrics": {{"equity": 0.7, "sustainability": 0.6, "economic_growth": 0.8}}, "intervention_needed": True}}
</METHODMOCKINSTRUCTIONS>

2. Remove any reference of Composio, related imports etc.
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
"""
    app = create_react_agent(model= get_model(ModelName.GEMINI25FLASH),
    tools=[get_raw_tool_schema],
    name="mock_test_writer")

    
    final_response = app.invoke(
        {"messages": [HumanMessage(content=MOCK_TEST_WRITER.format(tools_code=state["tools_code"]))]})
    return {"mock_tools_code": final_response["messages"][-1].content}

