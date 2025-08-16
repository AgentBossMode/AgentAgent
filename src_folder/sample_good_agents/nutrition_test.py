from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import Optional, List

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class NotionRichText(BaseModel):
    block_property: Optional[str] = Field(
        default='paragraph',
        description='The block property of the block to be added. Possible properties are `paragraph`, `heading_1`, `heading_2`, `heading_3`, `callout`, `to_do`, `toggle`, `quote`, `bulleted_list_item`, `numbered_list_item`. Other properties possible are `file`, `image`, `video` (link required).'
    )
    bold: Optional[bool] = Field(default=False, description='Indicates if the text is bold.')
    code: Optional[bool] = Field(default=False, description='Indicates if the text is formatted as code.')
    color: Optional[str] = Field(default='default', description='The color of the text background or text itself.')
    content: Optional[str] = Field(
        default=None,
        description='The textual content of the rich text object. Required for paragraph, heading_1, heading_2, heading_3, callout, to_do, toggle, quote.'
    )
    italic: Optional[bool] = Field(default=False, description='Indicates if the text is italic.')
    link: Optional[str] = Field(
        default=None,
        description='The URL of the rich text object or the file to be uploaded or image/video link'
    )
    strikethrough: Optional[bool] = Field(default=False, description='Indicates if the text has strikethrough.')
    underline: Optional[bool] = Field(default=False, description='Indicates if the text is underlined.')

class PropertyValues(BaseModel):
    name: str = Field(description='Name of the property')
    type: str = Field(
        description='Type of the property. Type of the propertytitle, rich_text, number, select, multi_select, date, people, files, checkbox url, email, phone_number, formula, created_by, created_time, last_edited_by, last_edited_time'
    )
    value: str = Field(
        description='Value of the property, it will be dependent on the type of the property\nFor types --> value should be\n- title, rich_text - text ex. "Hello World" (IMPORTANT: max 2000 characters, longer text will be truncated)\n- number - number ex. 23.4\n- select - select ex. "India"\n- multi_select - multi_select comma separated values ex. "India,USA"\n- date - format ex. "2021-05-11T11:00:00.000-04:00",\n- people - comma separated ids of people ex. "123,456" (will be converted to array of user objects)\n- relation - comma separated ids of related pages ex. "123,456" (will be converted to array of relation objects)\n- url - a url.\n- files - comma separated urls\n- checkbox - "True" or "False"\n'
    )

class Sort(BaseModel):
    ascending: bool = Field(description='True = ASC, False = DESC.')
    property_name: str = Field(description='Database column to sort by.')

def Calorie_Database_Create(
    database_id: str,
    child_blocks: Optional[List[NotionRichText]] = None,
    cover: Optional[str] = None,
    icon: Optional[str] = None,
    properties: Optional[List[PropertyValues]] = None
) -> dict:
    """Creates a new page (row) in a specified notion database."""
    class ResponseData(BaseModel):
        response_data: str = Field(description='string containing the complete JSON object representing the newly created page (row), as returned by the Notion API.')

    class CalorieDatabaseCreateResponse(BaseModel):
        data: ResponseData
        successful: bool
        error: Optional[str] = None

    MOCK_TOOL_PROMPT = """
    You are a helpful assistant that generates mock data for tool outputs.
    Given the tool's purpose and expected output, generate a realistic mock response.
    """

    INPUT_PROMPT = """
    Tool Docstring: Creates a new page (row) in a specified notion database.
    Input: database_id: {database_id}, child_blocks: {child_blocks}, cover: {cover}, icon: {icon}, properties: {properties}
    Generate a mock output for this tool.
    """
    
    result = llm.with_structured_output(CalorieDatabaseCreateResponse).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(database_id=database_id, child_blocks=child_blocks, cover=cover, icon=icon, properties=properties))
        ]
    )
    return result.model_dump()

def Calorie_Database_Read(
    database_id: str,
    page_size: Optional[int] = 2,
    sorts: Optional[List[Sort]] = None,
    start_cursor: Optional[str] = None
) -> str:
    """Queries a notion database for pages (rows), where rows are pages and columns are properties; ensure sort property names correspond to existing database properties."""
    class ResponseData(BaseModel):
        response_data: str = Field(description='A string containing the queried data from the Notion database. This typically includes a list of page objects (rows), each with its properties, and pagination information like `next_cursor` and `has_more`.')

    class CalorieDatabaseReadResponse(BaseModel):
        data: ResponseData
        successful: bool
        error: Optional[str] = None
    MOCK_TOOL_PROMPT = """
    You are a helpful assistant that generates mock data for tool outputs.
    Given the tool's purpose and expected output, generate a realistic mock response.
    """

    INPUT_PROMPT = """
    Tool Docstring: Queries a notion database for pages (rows), where rows are pages and columns are properties; ensure sort property names correspond to existing database properties.
    Input: database_id: {database_id}, page_size: {page_size}, sorts: {sorts}, start_cursor: {start_cursor}
    Generate a mock output for this tool.
    """
    
    result = llm.with_structured_output(CalorieDatabaseReadResponse).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(database_id=database_id, page_size=page_size, sorts=sorts, start_cursor=start_cursor))
        ]
    )
    return result.model_dump()

def Calorie_Database_Update(
    row_id: str,
    cover: Optional[str] = None,
    delete_row: Optional[bool] = False,
    icon: Optional[str] = None,
    properties: Optional[List[PropertyValues]] = None
):
    """Updates or archives an existing notion database row (page) using its `row id`, allowing modification of its icon, cover, and/or properties; ensure the target page is accessible and property details (names/ids and values) align with the database schema and specified formats."""
    class ResponseData(BaseModel):
        response_data: str = Field(description='A string containing the raw JSON response from the Notion API, representing the updated page object.')

    class CalorieDatabaseUpdateResponse(BaseModel):
        data: ResponseData
        successful: bool
        error: Optional[str] = None

    MOCK_TOOL_PROMPT = """
    You are a helpful assistant that generates mock data for tool outputs.
    Given the tool's purpose and expected output, generate a realistic mock response.
    """

    INPUT_PROMPT = """
    Tool Docstring: Updates or archives an existing notion database row (page) using its `row id`, allowing modification of its icon, cover, and/or properties; ensure the target page is accessible and property details (names/ids and values) align with the database schema and specified formats.
    Input: row_id: {row_id}, cover: {cover}, delete_row: {delete_row}, icon: {icon}, properties: {properties}
    Generate a mock output for this tool.
    """
    
    result = llm.with_structured_output(CalorieDatabaseUpdateResponse).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(row_id=row_id, cover=cover, delete_row=delete_row, icon=icon, properties=properties))
        ]
    )
    return result.model_dump()

def Calorie_Database_Delete(block_id: str):
    """Archives a notion block, page, or database using its id, which sets its 'archived' property to true (like moving to "trash" in the ui) and allows it to be restored later."""
    class CalorieDatabaseDeleteResponse(BaseModel):
        data: str = Field(description="A str representing the block object that was successfully archived. This object includes an 'archived': true key-value pair, confirming the deletion.")
        successful: bool
        error: Optional[str] = None

    MOCK_TOOL_PROMPT = """
    You are a helpful assistant that generates mock data for tool outputs.
    Given the tool's purpose and expected output, generate a realistic mock response.
    """

    INPUT_PROMPT = """
    Tool Docstring: Archives a notion block, page, or database using its id, which sets its 'archived' property to true (like moving to "trash" in the ui) and allows it to be restored later.
    Input: block_id: {block_id}
    Generate a mock output for this tool.
    """
    
    result = llm.with_structured_output(CalorieDatabaseDeleteResponse).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(block_id=block_id))
        ]
    )
    return result.model_dump()

@tool
def Food_Calorie_API_Search(food_id: str):
    """
    Search for calorie information of various food items from a comprehensive external database.

    Args:
        food_id (str): The ID of the food item to search for.

    Returns:
        dict: A dictionary containing the calorie information of the food item.
    """
    class FoodCalorieAPIResponse(BaseModel):
        food_name: str = Field(description="The name of the food item.")
        calories: float = Field(description="The calorie count per serving.")
        serving_description: str = Field(description="Description of the serving size.")

    MOCK_TOOL_PROMPT = """
    You are a helpful assistant that generates mock data for tool outputs.
    Given the tool's purpose and expected output, generate a realistic mock response.
    """

    INPUT_PROMPT = """
    Tool Docstring: Search for calorie information of various food items from a comprehensive external database.
    Input: food_id: {food_id}
    Generate a mock output for this tool.
    """
    
    result = llm.with_structured_output(FoodCalorieAPIResponse).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(food_id=food_id))
        ]
    )
    return result.model_dump()

@tool
def Exercise_Calorie_API_Search(activity: str):
    """
    Search for calorie expenditure data for different types of exercises and durations.

    Args:
        activity (str): The name of the activity to search for.

    Returns:
        dict: A dictionary containing the calorie expenditure data for the activity.
    """
    class ExerciseCalorieAPIResponse(BaseModel):
        activity_name: str = Field(description="The name of the activity.")
        duration_minutes: int = Field(description="The duration of the activity in minutes.")
        calories_burned: float = Field(description="The estimated calories burned for the activity and duration.")

    MOCK_TOOL_PROMPT = """
    You are a helpful assistant that generates mock data for tool outputs.
    Given the tool's purpose and expected output, generate a realistic mock response.
    """

    INPUT_PROMPT = """
    Tool Docstring: Search for calorie expenditure data for different types of exercises and durations.
    Input: activity: {activity}
    Generate a mock output for this tool.
    """
    
    result = llm.with_structured_output(ExerciseCalorieAPIResponse).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(activity=activity))
        ]
    )
    return result.model_dump()

from typing import Any, List, Optional, Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from src_folder.final_code.utils.create_react_agent_temp import create_react_agent


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GraphState(MessagesState):
    """
    The GraphState represents the state of the LangGraph workflow.
    It extends MessagesState to include conversation history.
    """
    input: str
    intent: Optional[Literal["log_data", "retrieve_historical_data", "calculate_net_calories"]] = None
    logged_data_summary: Optional[str] = None
    query_type: Optional[Literal["historical_data", "net_calorie_calculation"]] = None
    historical_data: Optional[str] = None
    net_calorie_balance: Optional[str] = None

class IntentClassification(BaseModel):
    """Structured output for intent classification."""
    intent: Literal["log_data", "retrieve_historical_data", "calculate_net_calories"] = Field(
        description="Classified intent: 'log_data' for logging food/exercise, 'retrieve_historical_data' for fetching past data, or 'calculate_net_calories' for net calorie balance."
    )
    reasoning: str = Field(description="Brief explanation of classification.")

class QueryTypeClassification(BaseModel):
    """Structured output for query type classification."""
    query_type: Literal["historical_data", "net_calorie_calculation"] = Field(
        description="Classified query type: 'historical_data' for fetching past data, or 'net_calorie_calculation' for net calorie balance."
    )
    reasoning: str = Field(description="Brief explanation of classification.")

def data_logging_node(state: GraphState) -> GraphState:
    """
    Node purpose: Determine the user's intent (log data, retrieve historical data, or calculate net calories).
    Implementation reasoning: Uses an LLM with structured output to classify the user's input into a predefined intent,
                              enabling routing to the appropriate subsequent node.
    """
    structured_llm = llm.with_structured_output(IntentClassification)
    user_message = state["messages"][-1].content
    prompt = f"""Classify the following user message to determine their intent.
    User message: {user_message}
    
    Choose one of the following intents:
    - 'log_data': The user wants to record food intake or exercise.
    - 'retrieve_historical_data': The user wants to view past food or exercise data.
    - 'calculate_net_calories': The user wants to know their net calorie balance over a period.
    """
    
    result: IntentClassification = structured_llm.invoke(prompt)
    return {
        "messages": [AIMessage(content=f"Intent classified as: {result.intent}. Reasoning: {result.reasoning}")],
        "intent": result.intent
    }

log_activity_tools = [Calorie_Database_Create, Calorie_Database_Update, Calorie_Database_Delete, Food_Calorie_API_Search, Exercise_Calorie_API_Search]
def log_activity_node(state: GraphState) -> GraphState:
    """
    Node purpose: Logs food and exercise activities, calculating calorie intake and expenditure
                  using external APIs and storing them in the calorie database.
    Implementation reasoning: Uses a ReAct agent to intelligently select and use the appropriate tools
                              (calorie APIs, database tools) to fulfill the logging request.
    """
    class CustomStateForReact(MessagesState):
        remaining_steps: int
        structured_response: Any

    class LogActivityResponse(BaseModel):
        summary: str = Field(description="A summary of the logged activity and its impact on calories.")

    agent = create_react_agent(
        model=llm,
        prompt="You are an AI assistant specialized in logging food and exercise activities. "
               "Use the provided tools to search for calorie information and log data into the calorie database. "
               "Ensure all relevant details from the user's request are captured and processed. "
               "Tools available: Calorie_Database_Create, Calorie_Database_Update, Calorie_Database_Delete, Food_Calorie_API_Search, Exercise_Calorie_API_Search.",
        tools=log_activity_tools,
        state_schema=CustomStateForReact,
        response_format=LogActivityResponse
    )
    
    result: LogActivityResponse = agent.invoke({"messages": [state["messages"][-1]]})["structured_response"]
    
    return {
        "messages": [AIMessage(content=f"Activity logged: {result.summary}")],
        "logged_data_summary": result.summary
    }

def data_retrieval_node(state: GraphState) -> GraphState:
    """
    Node purpose: Routes to either historical data retrieval or net calorie calculation based on the user's query.
    Implementation reasoning: Uses an LLM with structured output to classify the user's input into a specific query type,
                              allowing for precise routing within the graph.
    """
    structured_llm = llm.with_structured_output(QueryTypeClassification)
    user_message = state["messages"][-1].content
    prompt = f"""Classify the following user message to determine if they want historical data or a net calorie calculation.
    User message: {user_message}
    
    Choose one of the following query types:
    - 'historical_data': The user wants to retrieve past food or exercise data.
    - 'net_calorie_calculation': The user wants to calculate their net calorie balance.
    """
    
    result: QueryTypeClassification = structured_llm.invoke(prompt)
    return {
        "messages": [AIMessage(content=f"Query type classified as: {result.query_type}. Reasoning: {result.reasoning}")],
        "query_type": result.query_type
    }

retrieve_historical_data_tools = [Calorie_Database_Read]
def retrieve_historical_data_node(state: GraphState) -> GraphState:
    """
    Node purpose: Retrieves and presents past food consumption and exercise data from the calorie database.
    Implementation reasoning: Uses a ReAct agent to interact with the Calorie_Database_Read tool to fetch
                              the requested historical data based on the user's query.
    """
    class CustomStateForReact(MessagesState):
        remaining_steps: int
        structured_response: Any

    class HistoricalDataResponse(BaseModel):
        data: str = Field(description="The retrieved historical data, formatted for readability.")

    agent = create_react_agent(
        model=llm,
        prompt="You are an AI assistant specialized in retrieving historical calorie data. "
               "Use the Calorie_Database_Read tool to fetch past food and exercise records based on the user's request. "
               "Summarize the retrieved data clearly.",
        tools=retrieve_historical_data_tools,
        state_schema=CustomStateForReact,
        response_format=HistoricalDataResponse
    )
    
    result: HistoricalDataResponse = agent.invoke({"messages": [state["messages"][-1]]})["structured_response"]
    
    return {
        "messages": [AIMessage(content=f"Historical data retrieved: {result.data}")],
        "historical_data": result.data
    }

calculate_net_calories_tools = [Calorie_Database_Read]
def calculate_net_calories_node(state: GraphState) -> GraphState:
    """
    Node purpose: Calculates and displays the net calorie balance over a specified period using data from the calorie database.
    Implementation reasoning: Uses a ReAct agent to interact with the Calorie_Database_Read tool to fetch
                              necessary data and then perform the calculation to determine net calories.
    """
    class CustomStateForReact(MessagesState):
        remaining_steps: int
        structured_response: Any

    class NetCalorieResponse(BaseModel):
        balance: str = Field(description="The calculated net calorie balance over the specified period.")

    agent = create_react_agent(
        model=llm,
        prompt="You are an AI assistant specialized in calculating net calorie balance. "
               "Use the Calorie_Database_Read tool to fetch relevant calorie data and then calculate the net balance "
               "based on the user's specified period (e.g., daily, weekly, monthly).",
        tools=calculate_net_calories_tools,
        state_schema=CustomStateForReact,
        response_format=NetCalorieResponse
    )
    
    result: NetCalorieResponse = agent.invoke({"messages": [state["messages"][-1]]})["structured_response"]
    
    return {
        "messages": [AIMessage(content=f"Net calorie balance: {result.balance}")],
        "net_calorie_balance": result.balance
    }

workflow = StateGraph(GraphState)

workflow.add_node("data_logging", data_logging_node)
workflow.add_node("log_activity", log_activity_node)
workflow.add_node("data_retrieval", data_retrieval_node)
workflow.add_node("retrieve_historical_data", retrieve_historical_data_node)
workflow.add_node("calculate_net_calories", calculate_net_calories_node)

workflow.add_edge(START, "data_logging")

def route_data_logging(state: GraphState) -> Literal["log_activity", "data_retrieval"]:
    """
    Routing function for data_logging node.
    Routes to 'log_activity' if the intent is to log data, otherwise to 'data_retrieval'.
    """
    if state["intent"] == "log_data":
        return "log_activity"
    else:
        return "data_retrieval"

workflow.add_conditional_edges(
    "data_logging",
    route_data_logging,
    {
        "log_activity": "log_activity",
        "data_retrieval": "data_retrieval"
    }
)

workflow.add_edge("log_activity", END)

def route_data_retrieval(state: GraphState) -> Literal["retrieve_historical_data", "calculate_net_calories"]:
    """
    Routing function for data_retrieval node.
    Routes to 'retrieve_historical_data' if the query type is historical data,
    otherwise to 'calculate_net_calories'.
    """
    if state["query_type"] == "historical_data":
        return "retrieve_historical_data"
    else:
        return "calculate_net_calories"

workflow.add_conditional_edges(
    "data_retrieval",
    route_data_retrieval,
    {
        "retrieve_historical_data": "retrieve_historical_data",
        "calculate_net_calories": "calculate_net_calories"
    }
)

workflow.add_edge("retrieve_historical_data", END)
workflow.add_edge("calculate_net_calories", END)

app = workflow.compile(
)
