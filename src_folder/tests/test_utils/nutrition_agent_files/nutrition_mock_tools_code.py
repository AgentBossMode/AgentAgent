nutrition_mock_tools_code = r'''
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

MOCK_TOOL_PROMPT = """
You are a helpful assistant that generates mock data for tool outputs.
Given the tool's purpose and expected output, generate a realistic mock response.
"""

INPUT_PROMPT = """
Tool Docstring: {description}
Input: {input}
Generate a mock output for this tool.
"""

class NotionRichText(BaseModel):
    block_property: Optional[str] = Field(
        default='paragraph',
        description='The block property of the block to be added. Possible properties are `paragraph`, `heading_1`, `heading_2`, `heading_3`, `callout`, `to_do`, `toggle`, `quote`, `bulleted_list_item`, `numbered_list_item`. Other properties possible are `file`, `image`, `video` (link required).'
    )
    bold: Optional[bool] = Field(default=False, description='Indicates if the text is bold.')
    code: Optional[bool] = Field(default=False, description='Indicates if the text is formatted as code.')
    color: Optional[str] = Field(
        default='default',
        description='The color of the text background or text itself.'
    )
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

class CreateCalorieDataResponse(BaseModel):
    data: str = Field(description="JSON string of the output data from the action execution, including response_data which is a dictionary containing the complete JSON object representing the newly created page (row), as returned by the Notion API.")
    successful: bool = Field(description="Whether or not the action execution was successful or not")
    error: Optional[str] = Field(default=None, description="Error if any occurred during the execution of the action")

def CreateCalorieData(
    database_id: str,
    child_blocks: Optional[List[NotionRichText]] = None,
    cover: Optional[str] = None,
    icon: Optional[str] = None,
    properties: Optional[List[PropertyValues]] = None
) -> str:
    """
    Creates a new page (row) in a specified notion database.
    Tool to create new calorie-related data entries.

    Args:
        database_id (str): Identifier (UUID) of the Notion database where the new page (row) will be inserted.
        child_blocks (Optional[List[NotionRichText]]): A list of `NotionRichText` objects defining content blocks to append to the new page's body.
        cover (Optional[str]): URL of an external image to set as the page cover.
        icon (Optional[str]): Emoji to be used as the page icon.
        properties (Optional[List[PropertyValues]]): Property values for the new page.

    Returns:
        str: A JSON string representing the result of the operation, including success status and data.

    Example:
        CreateCalorieData(database_id="your_database_id", properties=[{"name": "Food Item", "type": "title", "value": "Apple"}, {"name": "Calories", "type": "number", "value": "95"}])
    """
    class Data(BaseModel):
        response_data: str = Field(description="JSON string of the complete JSON object representing the newly created page (row), as returned by the Notion API.")

    class CreateCalorieDataResponse(BaseModel):
        data: Data = Field(description="Data from the action execution")
        successful: bool = Field(description="Whether or not the action execution was successful or not")
        error: Optional[str] = Field(default=None, description="Error if any occurred during the execution of the action")

    input_str = f"database_id: {database_id}, child_blocks: {child_blocks}, cover: {cover}, icon: {icon}, properties: {properties}"
    description = CreateCalorieData.__doc__

    result = llm.with_structured_output(CreateCalorieDataResponse).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

class ReadCalorieDataResponse(BaseModel):
    data: str = Field(description="JSON string of the output data from the action execution, including response_data which is a dictionary containing the queried data from the Notion database.")
    successful: bool = Field(description="Whether or not the action execution was successful or not")
    error: Optional[str] = Field(default=None, description="Error if any occurred during the execution of the action")

class Sort(BaseModel):
    property_name: str = Field(description='Database column to sort by.')
    ascending: bool = Field(description='True = ASC, False = DESC.')

def ReadCalorieData(
    database_id: str,
    page_size: Optional[int] = 2,
    sorts: Optional[List[Sort]] = None,
    start_cursor: Optional[str] = None
) -> str:
    """
    Queries a notion database for pages (rows), where rows are pages and columns are properties; ensure sort property names correspond to existing database properties.
    Tool to read existing calorie-related data.

    Args:
        database_id (str): Identifier of the Notion database to query.
        page_size (Optional[int]): The maximum number of items (pages or rows) to return in a single response.
        sorts (Optional[List[Sort]]): List of sort rules.
        start_cursor (Optional[str]): An opaque cursor for pagination.

    Returns:
        str: A JSON string representing the result of the operation, including success status and data.

    Example:
        ReadCalorieData(database_id="your_database_id", page_size=10, sorts=[{"property_name": "Date", "ascending": False}])
    """
    class Data(BaseModel):
        response_data: str = Field(description="JSON string of a dictionary containing the queried data from the Notion database. This typically includes a list of page objects (rows), each with its properties, and pagination information like `next_cursor` and `has_more`.")

    class ReadCalorieDataResponse(BaseModel):
        data: Data = Field(description="Data from the action execution")
        successful: bool = Field(description="Whether or not the action execution was successful or not")
        error: Optional[str] = Field(default=None, description="Error if any occurred during the execution of the action")

    input_str = f"database_id: {database_id}, page_size: {page_size}, sorts: {sorts}, start_cursor: {start_cursor}"
    description = ReadCalorieData.__doc__

    result = llm.with_structured_output(ReadCalorieDataResponse).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

class SearchFoodCalorieAPIResponse(BaseModel):
    product_info: str = Field(description="JSON string of a dictionary containing product information, including nutriments.")

def SearchFoodCalorieAPI(food_name: str) -> str:
    """
    Search for calorie information of various food items from an external database.
    Tool to search for calorie information of various food items from an external database.

    Args:
        food_name (str): The name of the food item to search for.

    Returns:
        str: A JSON string representing a dictionary containing product information, including nutriments.

    Example:
        SearchFoodCalorieAPI(food_name="apple")
    """
    input_str = f"food_name: {food_name}"
    description = SearchFoodCalorieAPI.__doc__

    result = llm.with_structured_output(SearchFoodCalorieAPIResponse).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

class SearchExerciseCalorieAPIResponse(BaseModel):
    calorie_expenditure_data: str = Field(description="JSON string of a dictionary containing calorie expenditure data.")

def SearchExerciseCalorieAPI(activity: str) -> str:
    """
    Search for calorie expenditure data for different types of exercises and durations.
    Tool to search for calorie expenditure data for different types of exercises and durations.

    Args:
        activity (str): The name of the exercise activity.

    Returns:
        str: A JSON string representing a dictionary containing calorie expenditure data.

    Example:
        SearchExerciseCalorieAPI(activity="running")
    """
    input_str = f"activity: {activity}"
    description = SearchExerciseCalorieAPI.__doc__

    result = llm.with_structured_output(SearchExerciseCalorieAPIResponse).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)'''