from final_code.states.AgentBuilderState import AgentBuilderState
from langgraph.graph import StateGraph, START, END  # Core LangGraph components for building stateful graphs
from langgraph.types import interrupt
from final_code.llms.model_factory import get_model
from langchain_tavily import TavilySearch, TavilyExtract
from langchain_openai import ChatOpenAI
from final_code.states.ReactCopilotKitState import ReactCopilotState
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_customize_config
from final_code.utils.create_react_agent_temp import create_react_agent
from typing import List
from final_code.states.NodesAndEdgesSchemas import Tool
from final_code.states.ToolOptions import ToolOptions
from pydantic import Field, BaseModel



tavily_extract_tool = TavilyExtract(
    extract_depth="advanced",
    include_images=False)

tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced",
)



TOOL_PROMPT_2 = """
You are a helpful assistant, user will provide you with a list of tool_name along with their tool_description

Instructions:
1. For each tool_description, call the TavilySearch tool to find "API" or "python sdk" which would achieve the tool objective based on the description
2. Once you identify some toolings, then use the TavilyExtract tool to find the python code snippets for those tooling which would help in implementation.
3. Try to return more than 1 tooling for tool_name, if you dont find any, just write a custom implementation.
"""

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
native_react_agent_2 = create_react_agent(
    model=llm,
    prompt = TOOL_PROMPT_2,
    tools=[tavily_search_tool, tavily_extract_tool],
    state_schema=ReactCopilotState,
    response_format=ToolOptions)

class ToolList(BaseModel):
    tools: List[Tool] = Field(description="Updated list of tools")

async def call_native_agents_to_get_tools(state: AgentBuilderState, config: RunnableConfig):
    modifiedConfig = copilotkit_customize_config(
        config,
        emit_messages=False,
          emit_tool_calls=True)
    answer = await native_react_agent_2.ainvoke(input={"messages":state["messages"]}, config=modifiedConfig)
    return {"tool_options": answer["structured_response"]}

async def get_user_input(state: AgentBuilderState, config: RunnableConfig):
    answer:dict = interrupt({"type": "select_non_composio_tools", "payload": state["tool_options"]})
    # {'Automated_FollowUp_Sender': 'Custom Python script using Gmail API, OpenAI API, AWS Lambda', 'Email_Engagement_Monitor': 'pytracking'}
    for tool in state["json_schema"].tools:
        if tool.name in answer.keys():
            sdk_name = answer[tool.name]
            for native_tool in state["tool_options"].native_tools:
                if tool.name == native_tool.tool_name:
                    for sdk in native_tool.tools_dentified:
                        if sdk.tool_sdk == sdk_name:
                                tool.py_code = sdk.code_implementation
                                break

native_tool_workflow = StateGraph(AgentBuilderState)
native_tool_workflow.add_node("call_native_agents_to_get_tools", call_native_agents_to_get_tools)
native_tool_workflow.add_node("get_user_input", get_user_input)

native_tool_workflow.add_edge(START, "call_native_agents_to_get_tools")
native_tool_workflow.add_edge("call_native_agents_to_get_tools", "get_user_input")
native_tool_workflow.add_edge("get_user_input", END)

native_tool_builder = native_tool_workflow.compile()