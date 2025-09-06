from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from typing import List, Optional
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class NotionRichText(BaseModel):
    block_property: Optional[str] = Field(
        default="paragraph",
        description="The block property of the block to be added. Possible properties are `paragraph`, `heading_1`, `heading_2`, `heading_3`, `callout`, `to_do`, `toggle`, `quote`, `bulleted_list_item`, `numbered_list_item`. Other properties possible are `file`, `image`, `video` (link required).",
        examples=["paragraph", "heading_1", "heading_2", "heading_3", "bulleted_list_item", "numbered_list_item", "to_do", "callout", "toggle", "quote"],
    )
    bold: Optional[bool] = Field(default=False, description="Indicates if the text is bold.", examples=[True, False])
    code: Optional[bool] = Field(default=False, description="Indicates if the text is formatted as code.", examples=[True, False])
    color: Optional[str] = Field(
        default="default",
        description="The color of the text background or text itself.",
        examples=["blue_background", "yellow_background", "gray", "purple"],
    )
    content: Optional[str] = Field(
        default=None,
        description="The textual content of the rich text object. Required for paragraph, heading_1, heading_2, heading_3, callout, to_do, toggle, quote.",
        examples=["Hello World", "This is a heading", "This is a paragraph"],
    )
    italic: Optional[bool] = Field(default=False, description="Indicates if the text is italic.", examples=[True, False])
    link: Optional[str] = Field(
        default=None,
        description="The URL of the rich text object or the file to be uploaded or image/video link",
        examples=["https://www.google.com"],
    )
    strikethrough: Optional[bool] = Field(default=False, description="Indicates if the text has strikethrough.", examples=[True, False])
    underline: Optional[bool] = Field(default=False, description="Indicates if the text is underlined.", examples=[True, False])

class PropertyValues(BaseModel):
    name: str = Field(description="Name of the property")
    type: str = Field(
        description="Type of the property. Type of the propertytitle, rich_text, number, select, multi_select, date, people, files, checkbox url, email, phone_number, formula, created_by, created_time, last_edited_by, last_edited_time",
        enum=[
            "title",
            "rich_text",
            "number",
            "select",
            "multi_select",
            "date",
            "people",
            "files",
            "checkbox",
            "url",
            "email",
            "phone_number",
            "formula",
            "relation",
            "rollup",
            "status",
            "created_time",
            "created_by",
            "last_edited_time",
        ],
    )
    value: str = Field(
        description='Value of the property, it will be dependent on the type of the property\nFor types --> value should be\n- title, rich_text - text ex. "Hello World" (IMPORTANT: max 2000 characters, longer text will be truncated)\n- number - number ex. 23.4\n- select - select ex. "India"\n- multi_select - multi_select comma separated values ex. "India,USA"\n- date - format ex. "2021-05-11T11:00:00.000-04:00",\n- people - comma separated ids of people ex. "123,456" (will be converted to array of user objects)\n- relation - comma separated ids of related pages ex. "123,456" (will be converted to array of relation objects)\n- url - a url.\n- files - comma separated urls\n- checkbox - "True" or "False"\n'
    )

class InsertRowDatabaseResponseData(BaseModel):
    response_data: str = Field(
        description="Dictionary containing the complete JSON object representing the newly created page (row), as returned by the Notion API."
    )

class InsertRowDatabaseResponseWrapper(BaseModel):
    data: InsertRowDatabaseResponseData = Field(description="Data from the action execution")
    error: Optional[str] = Field(default=None, description="Error if any occurred during the execution of the action")
    successful: bool = Field(description="Whether or not the action execution was successful or not")

class Sort(BaseModel):
    ascending: bool = Field(description="True = ASC, False = DESC.", examples=[True])
    property_name: str = Field(description="Database column to sort by.", examples=["Task Name"])

class QueryDatabaseResponseData(BaseModel):
    response_data: str = Field(
        description="A dictionary containing the queried data from the Notion database. This typically includes a list of page objects (rows), each with its properties, and pagination information like `next_cursor` and `has_more`."
    )

class QueryDatabaseResponseWrapper(BaseModel):
    data: QueryDatabaseResponseData = Field(description="Data from the action execution")
    error: Optional[str] = Field(default=None, description="Error if any occurred during the execution of the action")
    successful: bool = Field(description="Whether or not the action execution was successful or not")

MOCK_TOOL_PROMPT = """
You are a helpful assistant that generates mock data for tool outputs.
Given the tool's purpose and expected output, generate a realistic mock response.
"""

INPUT_PROMPT = """
Tool Docstring: {description}
Input: {input}
Generate a mock output for this tool.
"""

def CreateCalorieData(
    database_id: str,
    child_blocks: Optional[List[NotionRichText]] = None,
    cover: Optional[str] = None,
    icon: Optional[str] = None,
    properties: Optional[List[PropertyValues]] = None,
) -> InsertRowDatabaseResponseWrapper:
    """
    Creates a new page (row) in a specified notion database.

    Args:
        database_id (str): Identifier (UUID) of the Notion database where the new page (row) will be inserted.
        child_blocks (Optional[List[NotionRichText]]): A list of `NotionRichText` objects defining content blocks.
        cover (Optional[str]): URL of an external image to set as the page cover.
        icon (Optional[str]): Emoji to be used as the page icon.
        properties (Optional[List[PropertyValues]]): Property values for the new page.

    Returns:
        InsertRowDatabaseResponseWrapper: A wrapper containing the response data, success status, and any error.
    """
    input_str = f"database_id: {database_id}, child_blocks: {child_blocks}, cover: {cover}, icon: {icon}, properties: {properties}"
    description = CreateCalorieData.__doc__
    print(description)
    result = llm.with_structured_output(InsertRowDatabaseResponseWrapper).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description)) # type: ignore
        ]
    )
    return result

def ReadCalorieData(
    database_id: str,
    page_size: Optional[int] = 2,
    sorts: Optional[List[Sort]] = None,
    start_cursor: Optional[str] = None,
) -> QueryDatabaseResponseWrapper:
    """
    Queries a notion database for pages (rows), where rows are pages and columns are properties; ensure sort property names correspond to existing database properties.

    Args:
        database_id (str): Identifier of the Notion database to query.
        page_size (Optional[int]): The maximum number of items (pages or rows) to return.
        sorts (Optional[List[Sort]]): List of sort rules.
        start_cursor (Optional[str]): An opaque cursor for pagination.

    Returns:
        QueryDatabaseResponseWrapper: A wrapper containing the queried data, success status, and any error.
    """
    input_str = f"database_id: {database_id}, page_size: {page_size}, sorts: {sorts}, start_cursor: {start_cursor}"
    description = ReadCalorieData.__doc__
    result = llm.with_structured_output(QueryDatabaseResponseWrapper).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description)) # type: ignore
        ]
    )
    return result

class SearchFoodCalorieAPIResponse(BaseModel):
    """
    A dictionary containing product information, including nutriments.
    """
    product_name: str = Field(description="The name of the food product.")
    nutriments: str = Field(description="A dictionary of nutrient information, e.g., {'energy-kcal_100g': 150, 'proteins_100g': 10, 'carbohydrates_100g': 20, 'fat_100g': 5}.")
    
def SearchFoodCalorieAPI(food_name: str) -> SearchFoodCalorieAPIResponse:
    """
    Search for calorie information of various food items from an external database.

    Args:
        food_name (str): The name of the food item to search for.

    Returns:
        dict: A dictionary containing product information, including nutriments.
    """
    input_str = f"food_name: {food_name}"
    description = SearchFoodCalorieAPI.__doc__
    result = llm.with_structured_output(SearchFoodCalorieAPIResponse).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description)) # type: ignore
        ]
    )
    return result

class SearchExerciseCalorieAPIResponse(BaseModel):
    """
    A dictionary containing calorie expenditure data.
    """
    activity: str = Field(description="The name of the exercise activity.")
    calories_per_hour: float = Field(description="Estimated calories burned per hour for the activity.")
    duration_minutes: int = Field(description="The duration of the activity in minutes.")
    total_calories_burned: float = Field(description="Total calories burned for the given duration.")

def SearchExerciseCalorieAPI(activity: str) -> SearchExerciseCalorieAPIResponse:
    """
    Search for calorie expenditure data for different types of exercises and durations.

    Args:
        activity (str): The name of the exercise activity.

    Returns:
        dict: A dictionary containing calorie expenditure data.
    """
    input_str = f"activity: {activity}"
    description = SearchExerciseCalorieAPI.__doc__
    result = llm.with_structured_output(SearchExerciseCalorieAPIResponse).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description)) # type: ignore
        ]
    )
    return result


from typing import Dict, Any, List, Optional, Literal, TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GraphState(MessagesState):
    """
    The GraphState represents the state of the LangGraph workflow.
    It extends MessagesState to include conversation history.
    """
    input: Optional[str] = None
    action: Optional[str] = None
    food_item: Optional[str] = None
    exercise_activity: Optional[str] = None
    calories_consumed: Optional[int] = None
    calories_burned: Optional[int] = None
    query_type: Optional[str] = None
    historical_data: Optional[dict] = None
    net_calories: Optional[int] = None

class ActionClassifier(BaseModel):
    """Structured output for classifying user intent."""
    action: Literal["log_data", "retrieve_data", "calculate_net_calories"] = Field(
        description="Classified action based on user input."
    )
    reasoning: str = Field(description="Brief explanation for the classification.")

class DataRetrievalClassifier(BaseModel):
    """Structured output for classifying data retrieval intent."""
    query_type: Literal["historical_data", "net_calories"] = Field(
        description="Classified query type for data retrieval."
    )
    reasoning: str = Field(description="Brief explanation for the classification.")

def data_logging(state: GraphState) -> GraphState:
    """
    Node purpose: Initial node to determine if the user wants to log data, retrieve historical data, or calculate net calories.
    Implementation reasoning: Uses structured output to classify the user's intent for routing.
    """
    structured_llm = llm.with_structured_output(ActionClassifier)
    user_message = state["messages"][-1].content
    
    prompt = f"""Classify the user's request: "{user_message}".
    Is the user trying to:
    - 'log_data' (e.g., log food, log exercise, record calories)
    - 'retrieve_data' (e.g., show past activities, what did I eat yesterday, show my exercise history)
    - 'calculate_net_calories' (e.g., what's my net calories, calculate my calorie balance)
    """
    
    classification: ActionClassifier = structured_llm.invoke(prompt)
    
    return {
        "action": classification.action,
        "messages": [AIMessage(content=f"Intent classified as: {classification.action}. Reasoning: {classification.reasoning}")]
    }

log_activity_tools = [CreateCalorieData, SearchFoodCalorieAPI, SearchExerciseCalorieAPI]
def log_activity(state: GraphState) -> GraphState:
    """
    Node purpose: Logs food and exercise activities, calculating calorie intake and expenditure.
    Implementation reasoning: Uses a ReAct agent to intelligently use the available tools to log data.
    """
    class LogActivityState(MessagesState):
        food_item: Optional[str] = None
        exercise_activity: Optional[str] = None
        calories_consumed: Optional[int] = None
        calories_burned: Optional[int] = None
        remaining_steps: Optional[List[Any]] = Field(default_factory=list)
        structured_response: Optional[Any] = None

    class LogActivityOutput(BaseModel):
        food_item: Optional[str] = Field(description="The food item logged, if any.")
        exercise_activity: Optional[str] = Field(description="The exercise activity logged, if any.")
        calories_consumed: Optional[int] = Field(description="Calories consumed, if food was logged.")
        calories_burned: Optional[int] = Field(description="Calories burned, if exercise was logged.")
        message: str = Field(description="A summary message of the logging action.")

    agent = create_react_agent(
        model=llm,
        prompt="You are a helpful assistant for logging calorie data. Use the provided tools to log food consumption and exercise activities. If the user mentions a food item, use SearchFoodCalorieAPI to find its calories and then CreateCalorieData to log it. If the user mentions an exercise, use SearchExerciseCalorieAPI to find calories burned and then CreateCalorieData to log it. Always provide a summary of what was logged.",
        tools=log_activity_tools,
        state_schema=LogActivityState,
        response_format=LogActivityOutput
    )
    
    user_message = state["messages"][-1].content
    result: LogActivityOutput = agent.invoke({"messages": [HumanMessage(content=user_message)]})["structured_response"]
    
    return {
        "food_item": result.food_item,
        "exercise_activity": result.exercise_activity,
        "calories_consumed": result.calories_consumed,
        "calories_burned": result.calories_burned,
        "messages": [AIMessage(content=result.message)]
    }

def data_retrieval(state: GraphState) -> GraphState:
    """
    Node purpose: Routes to either historical data retrieval or net calorie calculation based on user query.
    Implementation reasoning: Uses structured output to classify the user's specific data retrieval intent.
    """
    structured_llm = llm.with_structured_output(DataRetrievalClassifier)
    user_message = state["messages"][-1].content
    
    prompt = f"""Classify the user's data retrieval request: "{user_message}".
    Is the user trying to:
    - 'historical_data' (e.g., show past activities, what did I eat yesterday, show my exercise history)
    - 'net_calories' (e.g., what's my net calories, calculate my calorie balance)
    """
    
    classification: DataRetrievalClassifier = structured_llm.invoke(prompt)
    
    return {
        "query_type": classification.query_type,
        "messages": [AIMessage(content=f"Query type classified as: {classification.query_type}. Reasoning: {classification.reasoning}")]
    }

retrieve_historical_data_tools = [ReadCalorieData]
def retrieve_historical_data(state: GraphState) -> GraphState:
    """
    Node purpose: Retrieves and presents past food consumption and exercise data.
    Implementation reasoning: Uses a ReAct agent to query historical data using the ReadCalorieData tool.
    """
    class RetrieveHistoricalDataState(MessagesState):
        historical_data: Optional[str] = None
        remaining_steps: Optional[List[Any]] = Field(default_factory=list)
        structured_response: Optional[Any] = None

    class HistoricalDataOutput(BaseModel):
        data: str = Field(description="The retrieved historical data.")
        message: str = Field(description="A summary message of the retrieved data.")

    agent = create_react_agent(
        model=llm,
        prompt="You are an assistant for retrieving historical calorie data. Use the ReadCalorieData tool to fetch past food consumption and exercise data based on the user's query. Summarize the retrieved data for the user.",
        tools=retrieve_historical_data_tools,
        state_schema=RetrieveHistoricalDataState,
        response_format=HistoricalDataOutput,
        version="v1"
    )
    
    user_message = state["messages"][-1].content
    result: HistoricalDataOutput = agent.invoke({"messages": [HumanMessage(content=user_message)]})["structured_response"]
    
    return {
        "historical_data": result.data,
        "messages": [AIMessage(content=result.message)]
    }

calculate_net_calories_tools = [ReadCalorieData]
def calculate_net_calories(state: GraphState) -> GraphState:
    """
    Node purpose: Calculates and displays the net calorie balance over a specified period.
    Implementation reasoning: Uses a ReAct agent to retrieve data and then calculate net calories.
    """
    class CalculateNetCaloriesState(MessagesState):
        net_calories: Optional[int] = None
        remaining_steps: Optional[List[Any]] = Field(default_factory=list)
        structured_response: Optional[Any] = None

    class NetCaloriesOutput(BaseModel):
        net_calories: int = Field(description="The calculated net calorie balance.")
        message: str = Field(description="A summary message of the net calorie calculation.")

    agent = create_react_agent(
        model=llm,
        prompt="You are an assistant for calculating net calorie balance. Use the ReadCalorieData tool to fetch relevant calorie data and then calculate the net calorie balance. Provide the final net calorie count to the user.",
        tools=calculate_net_calories_tools,
        state_schema=CalculateNetCaloriesState,
        response_format=NetCaloriesOutput,
        version="v1"
    )
    
    user_message = state["messages"][-1].content
    result: NetCaloriesOutput = agent.invoke({"messages": [HumanMessage(content=user_message)]})["structured_response"]
    
    return {
        "net_calories": result.net_calories,
        "messages": [AIMessage(content=result.message)]
    }

workflow = StateGraph(GraphState)

workflow.add_node("data_logging", data_logging)
workflow.add_node("log_activity", log_activity)
workflow.add_node("data_retrieval", data_retrieval)
workflow.add_node("retrieve_historical_data", retrieve_historical_data)
workflow.add_node("calculate_net_calories", calculate_net_calories)

workflow.add_edge(START, "data_logging")

def route_data_logging(state: GraphState) -> Literal["log_activity", "data_retrieval"]:
    """
    Routing function for data_logging node.
    Routes to 'log_activity' if the action is 'log_data'.
    Routes to 'data_retrieval' if the action is 'retrieve_data' or 'calculate_net_calories'.
    """
    if state["action"] == "log_data":
        return "log_activity"
    elif state["action"] in ["retrieve_data", "calculate_net_calories"]:
        return "data_retrieval"
    else:
        # Fallback or error handling, though structured output should prevent this
        return "data_retrieval" # Default to data retrieval if classification is ambiguous

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
    Routes to 'retrieve_historical_data' if the query_type is 'historical_data'.
    Routes to 'calculate_net_calories' if the query_type is 'net_calories'.
    """
    if state["query_type"] == "historical_data":
        return "retrieve_historical_data"
    elif state["query_type"] == "net_calories":
        return "calculate_net_calories"
    else:
        # Fallback or error handling
        return "retrieve_historical_data" # Default to historical data if classification is ambiguous

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

app = workflow.compile()