

from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.nodes.tools.composio_info_tools import get_raw_tool_schema
from final_code.llms.model_factory import get_model, ModelName
from langchain_core.messages import SystemMessage, HumanMessage
from final_code.states.NodesAndEdgesSchemas import JSONSchema
from final_code.pydantic_models.Questions import Questions
from langgraph.types import interrupt
def generate_additional_info_questions(state: AgentBuilderState):
    GENERATE_QUESTIONS_PROMPT = """
You are a software developer working with a client who has provided you with the following:
1. JSONSchema outlining the information regarding a proposed langgraph workflow
2. COMPOSIO_TOOL_SCHEMAS - for tools fetched via 'composio' tool sdk
3. DRYRUNS - the sample dry_runs for the proposed JSONSCHEMA langgraph workflow

<INSTRUCTIONS>
1. Understand the JSONSCHEMA and the DRYRUNS, along with all the TOOLs in JSONSCHEMA
2. If the Tool is marked as composio then also refer to COMPOSIO_TOOL_SCHEMAS
3. Now see if there are any pieces of non-API-KEY logical information that need to be provided by the user. 
4. You are not to ask questions which are asking for a 'key' of any sorts, that data is critical and will be asked sometime in the future, you are not supposed to ask for any key.
5. Do not ask questions which would suggest adding/removing tools, or any major changes in the workflow, these pieces of information were already confirmed with client.
</INSTRUCTIONS>
"""
    llm = get_model(ModelName.GPT41MINI)
    tool_schemas = ""
    json_schema: JSONSchema = state["json_schema"]
    for tool in json_schema.tools:
        if tool.is_composio_tool:
            tool_schemas+=f"""
<TOOL_NAME_{tool.composio_tool_slug}>
{get_raw_tool_schema(tool.composio_tool_slug)}
</TOOL_NAME_{tool.composio_tool_slug}>
\n\n\n
"""



    questions_llm = llm.with_structured_output(Questions)
    questions: Questions =  questions_llm.invoke([SystemMessage(content=GENERATE_QUESTIONS_PROMPT),
                                                        HumanMessage(content=f"<JSONSCHEMA>\n{state['json_schema'].model_dump_json(indent=2)}\n</JSONSCHEMA>"),
                                                        HumanMessage(content=f"<COMPOSIO_TOOL_SCHEMAS>\n{tool_schemas}\n</COMPOSIO_TOOL_SCHEMAS>"),
                                                        HumanMessage(content=f"<DRYRUNS>\n{state['dry_runs'].model_dump_json(indent=2)}\n</DRYRUNS>")]
                                                       )
    return {"questions": questions}

def additional_info_interrupt(state: AgentBuilderState):
    value = interrupt(value={"type":"additional_info", "payload": state["questions"]})
    return {"answers": value}