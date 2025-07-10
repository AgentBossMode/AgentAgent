from final_code.states.AgentBuilderState import AgentBuilderState
from langgraph.graph import StateGraph, START, END  # Core LangGraph components for building stateful graphs
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt
from typing import Literal
from final_code.pydantic_models.EndOrContinue import EndOrContinue
from langchain_core.messages import HumanMessage, SystemMessage
from final_code.llms.model_factory import get_model
from final_code.states.NodesAndEdgesSchemas import JSONSchema
from langchain_tavily import TavilySearch, TavilyExtract
from langchain_openai import ChatOpenAI
from final_code.utils.copilotkit_interrupt_temp import copilotkit_interrupt
from final_code.states.ReactCopilotKitState import ReactCopilotState
from langchain_core.runnables import RunnableConfig
from copilotkit.langgraph import copilotkit_customize_config



tavily_extract_tool = TavilyExtract(
    extract_depth="advanced",
    include_images=False)

tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
    # include_answer=False,
    # include_raw_content=False,
    # include_images=False,
    # include_image_descriptions=False,
    search_depth="advanced",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None
)

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)


TOOL_PROMPT = """
You are responsible for generating the right tool for the given requirements by user.

Follow the following instructions:
1. Respond to the user to understand if they already have any apps/tools that they would like to use for the given requirements. 
2. If the user says donot build tools, you will just respond with "No tools required for the task at hand, please proceed with the next step."
3. Use TavilySearch to find the websites which would provide the python code sdk or code samples for the given tools, this should be called only if the user has provided the name of a proper tool/service/library.
4. Use TavilyExtract with extract_depth advanced to extract the python code snippets from the search results.
5. Donot respond with your own knowledge, if you were not able to find anything from TavilySearch or TavilyExtract, just respond with "No tools found for the task at hand, please proceed with the next step."
6. If you are able to find the python code snippets, do mention the url in the output

Output:
Corresponding to each input tool description: provide the python_code snippet based on the search results to implement the tool.

                                               
Example:

Input:
tool_name_1 and tool_description 1
tool_name_2 and tool_description 2

Output:

*For tool_name_1, here are the following proposed tools:

def tool_name_1():
    # Python code snippet based on the search results for tool_name_1

*For tool_name_2, here are the following proposed tools:
def tool_name_2():
    # Python code snippet based on the search results for tool_name_2


7. If you get human input, responding to the above, now finally provide the final tools to be used corresponding to each user requirement.
"""

native_react_agent = create_react_agent(
    model=llm,
    prompt=TOOL_PROMPT,tools=[tavily_search_tool, tavily_extract_tool],
    state_schema=ReactCopilotState)

def select_tool_or_human_review(state: AgentBuilderState, config: RunnableConfig) -> Command[Literal["compile_final_tool", "human_review"]]:
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
        return Command(goto="compile_final_tool")
    else:
        return Command(goto="human_review")

def get_human_review(state: AgentBuilderState, config: RunnableConfig) -> Command[Literal["native_react_agent"]]:
    answer: dict = interrupt(state["messages"][-1].content)
    for key in answer.keys():
        return Command(goto="native_react_agent", update={"messages": [HumanMessage(content=answer[key])]}) 



def compile_final_tool(state: AgentBuilderState, config: RunnableConfig):
    modifiedConfig = copilotkit_customize_config(
        config,
        emit_messages=False, # if you want to disable message streaming 
        emit_tool_calls=False # if you want to disable tool call streaming 
    )
    json_schema = state["json_schema"]
    llm_with_struct = llm.with_structured_output(JSONSchema)
    prompt= "User will provide the representation of the JSONSchema object, and also provide a list of functions, along with the python code that corresponds to the function. You are to generate a JSONSchema object with the updated information. "
    updated_json_schema: JSONSchema = llm_with_struct.invoke([SystemMessage(content=prompt)] + [HumanMessage(content=json_schema.model_dump_json())] + [state["messages"][-1]], config = modifiedConfig)
    return {"json_schema": updated_json_schema}



native_tool_workflow = StateGraph(AgentBuilderState)
native_tool_workflow.add_node("human_review", get_human_review)
native_tool_workflow.add_node("native_react_agent", native_react_agent)
native_tool_workflow.add_node("compile_final_tool", compile_final_tool)
native_tool_workflow.add_node("select_tool_or_human_review", select_tool_or_human_review)

native_tool_workflow.add_edge(START, "native_react_agent")
native_tool_workflow.add_edge("native_react_agent", "select_tool_or_human_review")
native_tool_workflow.add_edge("compile_final_tool", END)

native_tool_builder = native_tool_workflow.compile()



