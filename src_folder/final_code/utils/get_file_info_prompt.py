from final_code.states.AgentBuilderState import AgentBuilderState
def get_file_info_prompt(state: AgentBuilderState):
    FILE_INFO = """
<python_code.py code>
{python_code}
</python_code.py code>

<mock_tools_code.py code>
{mock_tools_code}
</mock_tools_code.py code>

<test_app.py code>
{pytest_code}
</test_app.py code>

<RELATION_OF_FILES>
test_app.py imports app from python_code.py 
python_code.py imports mock_tools_code.py
</RELATION_OF_FILES>
"""
    return FILE_INFO.format(python_code= state["python_code"], mock_tools_code=state["mock_tools_code"], pytest_code=state["pytest_code"])
