from final_code.states.CodeEvalState import CodeEvalState
from final_code.nodes.code_reflection_node import code_reflection_node_updated

def reflection_node(state: CodeEvalState):
    result = code_reflection_node_updated.invoke({"code_to_reflect": state["python_code"], "mock_tools_code": state["mock_tools_code"]})
    return {"python_code": result["reflection_code"]}

