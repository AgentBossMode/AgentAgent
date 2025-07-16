from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Literal
from langgraph.types import Command, interrupt
from final_code.states.AgentBuilderState import AgentBuilderState
from final_code.states.NodesAndEdgesSchemas import JSONSchema
from final_code.pydantic_models.EndOrContinue import EndOrContinue
from final_code.nodes.tools.composio_info_tools import get_all_raw_tool_schemas_for_a_toolkit, get_all_toolkits
from final_code.utils.copilotkit_interrupt_temp import copilotkit_interrupt
from langchain_core.runnables import RunnableConfig
from final_code.states.ReactCopilotKitState import ReactCopilotState
from copilotkit.langgraph import copilotkit_customize_config


llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

TOOL_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant, user will provide you with a list of tool names along with their descriptions
You are also provided with a list of Composio TOOLKITS along with their descriptions below.
<COMPOSIO_TOOLKITS>
{toolkit_list}                                               
</COMPOSIO_TOOLKITS>

<RESPONSEFORMAT>
For suggestions/final response follow this format:
tool_name_1:
TOOLKIT_SLUG_NAME: Apollo
TOOL_SLUG_NAME:APOLLO_UPDATE_CONTACT_STATUS_IN_SEQUENCE
justification for choosing APOLLO_UPDATE_CONTACT_STATUS_IN_SEQUENCE for tool_name_1

tool_name_2:
No tool TOOLKIT found
</RESPONSEFORMAT>
                                               
1. Respond with suggestions of a TOOLKIT SLUG in the COMPOSIO_TOOLKITS list for each tool_name provided by the user. Refer to RESPONSEFORMAT section
2. Understand the user response, figure out which TOOLKIT SLUG in COMPOSIO_TOOLKITS are identified for each tool_name
3. For only the TOOLKIT SLUGs retrieved in above step, check the list of TOOLS corresponding to the TOOLKIT using 'get_all_raw_tool_schemas_for_a_toolkit' tool. The tool may return with an empty list, in that case mention that the tool is not available in the COMPOSIO_TOOLKITS list and Suggest an alternative, call the tool again to get the TOOL SLUG for this alternative TOOLKIT slug.
4. check which of the raw_tool_schema best suits the requirements, donot suggest TOOL SLUG which are remotely connected to the task at hand.
5. Your job is not to forcefully find a composio TOOLKIT if it is not available, do not impose on user. Understand what they say, if you donot have the app they require in the COMPOSIO_TOOLKITS list, just go and provide the final response.
6. Get confirmation from the user about the tools you are going ahead with, use the RESPONSEFORMAT as reference.
7. Now output the final user approved response, mentioning that the user has approved the response
""")

tools =  [get_all_raw_tool_schemas_for_a_toolkit]

def select_tool_or_human_review(state: AgentBuilderState, config: RunnableConfig) -> Command[Literal["select_final_tool", "human_review"]]:
    modifiedConfig = copilotkit_customize_config(
        config,
        emit_messages=False, # if you want to disable message streaming 
        emit_tool_calls=False # if you want to disable tool call streaming 
    )
    llm_with_struct = llm.with_structured_output(EndOrContinue)
    should_continue: EndOrContinue = llm_with_struct.invoke(
        [SystemMessage(content="You are supposed to analyze if the given AI message is asking user for any inputs or approvals, if yes then mark should_end_conversation as false, else if AI message says that the user has approved the suggestions, mark should_end_conversation as true"), state["messages"][-1]]
        , config=modifiedConfig)
    if should_continue.should_end_conversation:
        return Command(goto="select_final_tool")
    else:
        return Command(goto="human_review")

def get_human_review(state: AgentBuilderState, config: RunnableConfig) -> Command[Literal["composio_tool_fetch"]]:
    answer: dict = interrupt(state["messages"][-1].content)
    for key in answer.keys():
        return Command(goto="composio_tool_fetch", update={"messages": [HumanMessage(content=answer[key])]}) 

def select_final_tool(state: AgentBuilderState, config: RunnableConfig):
    modifiedConfig = copilotkit_customize_config(
        config,
        emit_messages=False, # if you want to disable message streaming 
        emit_tool_calls=False # if you want to disable tool call streaming 
    )
    json_schema = state["json_schema"]
    llm_with_struct = llm.with_structured_output(JSONSchema)
    prompt= "User will provide the representation of the JSONSchema object, and also provide a list of functions, along with the commposio TOOLKIT_SLUG_NAME and TOOL_SLUG_NAME pair that corresponds to the function, in some cases there might not be any composio_action provided. You are to generate a JSONSchema object with the updated information. "
    updated_json_schema: JSONSchema = llm_with_struct.invoke(
        [SystemMessage(content=prompt)] + [HumanMessage(content=json_schema.model_dump_json())] + [state["messages"][-1]],
        config=modifiedConfig)
    return {"json_schema": updated_json_schema}

composio_tool_fetch_app = create_react_agent(llm, prompt=TOOL_PROMPT.format(toolkit_list=get_all_toolkits()), tools=tools, name="composio_tool_fetch")

workflow = StateGraph(AgentBuilderState)
workflow.add_node("human_review", get_human_review)
workflow.add_node("composio_tool_fetch", composio_tool_fetch_app)
workflow.add_node("select_final_tool", select_final_tool)
workflow.add_node("select_tool_or_human_review", select_tool_or_human_review)
workflow.add_edge(START, "composio_tool_fetch")
workflow.add_edge("composio_tool_fetch", "select_tool_or_human_review")
workflow.add_edge("select_final_tool", END)
composio_tool_builder= workflow.compile()