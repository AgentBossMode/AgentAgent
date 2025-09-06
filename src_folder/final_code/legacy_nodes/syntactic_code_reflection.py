from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.legacy_nodes.code_reflection_node import code_reflection_node_updated
#from final_code.nodes.evaluation_pipeline_nodes.mock_files import mock_tools_py, main_py, fitness_tools, fitness_main, pytest

async def reflection_node(state: AgentBuilderState):
    result = await code_reflection_node_updated.ainvoke(state)
    #return {"python_code": fitness_main, "tools": fitness_tools, "mock_tools_code":fitness_tools, "pytest_code": pytest}
    return {"python_code": result["python_code"]}

