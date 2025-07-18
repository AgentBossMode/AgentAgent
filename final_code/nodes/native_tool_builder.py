from final_code.states.AgentBuilderState import AgentBuilderState
from langgraph.graph import StateGraph, START, END  # Core LangGraph components for building stateful graphs
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
from final_code.utils.create_react_agent_temp import create_react_agent
# from langgraph.prebuilt import create_react_agent --> not working due to bug in langgraph, using custom create_react_agent function



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
You are a helpful assistant, user will provide you with a list of tool names along with their descriptions
<RESPONSEFORMAT>
For suggestions/final response follow this format:

tool_name_1:
TOOL_LIBRARY_IDENTIFIED: tool_lib_1

FUNCTION IMPLEMENTATION:

```python
def tool_name_1():
    # Python code snippet based on the search results for tool_name_1, using tool_lib_1
```

URL: https://example.com/tool_lib_1 (this is the url from where you got the python code snippet for tool_name_1)
JUSTIFICATION: Justification for choosing tool_lib_1 for tool_name_1

tool_name_2:
TOOL_LIBRARY_IDENTIFIED: tool_lib_2

FUNCTION IMPLEMENTATION:

```python
def tool_name_2():
    # Python code snippet based on the search results for tool_name_1, using tool_lib_2
```

URL: https://example.com/tool_lib_2 (this is the url from where you got the python code snippet for tool_name_2)
JUSTIFICATION: Justification for choosing tool_lib_2 for tool_name_2

</RESPONSEFORMAT>

Follow the following instructions:
1. Respond to the user with suggestions of a TOOL_LIBRARY_IDENTIFIED for each tool_name provided by the user. Refer to RESPONSEFORMAT section
2. Based on response of human, Use TavilySearch to find the websites which would provide the python code sdk or code samples for the given tools, this should be called only if the user has provided the name of a proper tool/service/library.
3. Use TavilyExtract with extract_depth advanced to extract the python code snippets from the search results.
4. Donot respond with your own knowledge, if you were not able to find anything from TavilySearch or TavilyExtract, just respond with "No tools found for the task at hand, please proceed with the next step." for that particular tool_name.
5. Get confirmation from the user about the tools you are going ahead with, use the RESPONSEFORMAT as reference.
6. Now output the final user approved response, mentioning that the user has approved the response
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
    print(answer)
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