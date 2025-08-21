nutrition_mock_tools =r'''
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

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

@tool
def Food_Database_API_Search(food_query: str) -> str:
    """
    Searches for food items and retrieves their nutritional information, including calorie content.

    Args:
        food_query (str): The food item to search for.

    Returns:
        str: JSON string of the output.

    Example:
        Food_Database_API_Search(food_query="apple")
    """
    class FoodItem(BaseModel):
        food_id: str = Field(description="The unique ID of the food item.")
        label: str = Field(description="The common name of the food item.")
        calories: float = Field(description="The calorie content per 100g of the food item.")
        brand: Optional[str] = Field(description="The brand name of the food item, if applicable.")

    class SearchResults(BaseModel):
        results: List[FoodItem] = Field(description="A list of food items matching the query.")

    input_str = f"food_query: {food_query}"
    description = Food_Database_API_Search.__doc__

    result = llm.with_structured_output(SearchResults).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

@tool    
def Food_Database_API_Read(food_id: str, quantity: float = 1.0, measure_uri: str = 'http://www.edamam.com/ontologies/edamam.owl#Measure_unit') -> str:
    """
    Reads nutritional information for a specific food item from the food database.

    Args:
        food_id (str): The ID of the food item.
        quantity (float): The quantity of the food item. Defaults to 1.0.
        measure_uri (str): The URI for the measure unit. Defaults to 'http://www.edamam.com/ontologies/edamam.owl#Measure_unit'.

    Returns:
        str: JSON string of the output.

    Example:
        Food_Database_API_Read(food_id="food_abc123", quantity=100.0, measure_uri="http://www.edamam.com/ontologies/edamam.owl#Measure_gram")
    """
    class NutrientInfo(BaseModel):
        label: str = Field(description="The name of the nutrient (e.g., 'Energy', 'Protein').")
        quantity: float = Field(description="The quantity of the nutrient.")
        unit: str = Field(description="The unit of measurement for the nutrient (e.g., 'kcal', 'g').")

    class FoodNutrients(BaseModel):
        food_id: str = Field(description="The unique ID of the food item.")
        total_calories: float = Field(description="The total calories for the specified quantity.")
        nutrients: List[NutrientInfo] = Field(description="A list of detailed nutritional information.")

    input_str = f"food_id: {food_id}, quantity: {quantity}, measure_uri: {measure_uri}"
    description = Food_Database_API_Read.__doc__

    result = llm.with_structured_output(FoodNutrients).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

@tool    
def Exercise_Database_API_Search(gender: str, age: int, weight: float, heart_rate: float, time_hours: float) -> str:
    """
    Searches for exercise activities and retrieves information about calories burned.
    Calculates calories burned based on gender, age, weight, heart rate, and exercise duration.

    Args:
        gender (str): The gender of the individual ('male' or 'female').
        age (int): The age of the individual in years.
        weight (float): The weight of the individual in kilograms.
        heart_rate (float): The average heart rate during the exercise.
        time_hours (float): The duration of the exercise in hours.

    Returns:
        str: JSON string of the output.

    Example:
        Exercise_Database_API_Search(gender="male", age=30, weight=70.5, heart_rate=150.0, time_hours=1.0)
    """
    class CaloriesBurned(BaseModel):
        estimated_calories_burned: float = Field(description="The estimated calories burned during the exercise.")
        gender: str = Field(description="The gender used for calculation.")
        age: int = Field(description="The age used for calculation.")
        weight: float = Field(description="The weight used for calculation.")
        heart_rate: float = Field(description="The heart rate used for calculation.")
        time_hours: float = Field(description="The duration in hours used for calculation.")

    input_str = f"gender: {gender}, age: {age}, weight: {weight}, heart_rate: {heart_rate}, time_hours: {time_hours}"
    description = Exercise_Database_API_Search.__doc__

    result = llm.with_structured_output(CaloriesBurned).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

@tool
def Exercise_Database_API_Read() -> str:
    """
    Reads information about calories burned for a specific exercise activity from the exercise database.
    This tool indicates that an external script (Calories.py) is used for prediction.

    Returns:
        str: JSON string of the output.

    Example:
        Exercise_Database_API_Read()
    """
    class ExerciseReadInfo(BaseModel):
        message: str = Field(description="A message indicating how to use the external Calories.py script.")

    input_str = "No specific input for this tool."
    description = Exercise_Database_API_Read.__doc__

    result = llm.with_structured_output(ExerciseReadInfo).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

class NotionRichText(BaseModel):
    block_property: Optional[str] = Field(default='paragraph', description='The block property of the block to be added. Possible properties are `paragraph`, `heading_1`, `heading_2`, `heading_3`, `callout`, `to_do`, `toggle`, `quote`, `bulleted_list_item`, `numbered_list_item`. Other properties possible are `file`, `image`, `video` (link required).')
    bold: Optional[bool] = Field(default=False, description='Indicates if the text is bold.')
    code: Optional[bool] = Field(default=False, description='Indicates if the text is formatted as code.')
    color: Optional[str] = Field(default='default', description='The color of the text background or text itself.')
    content: Optional[str] = Field(default=None, description='The textual content of the rich text object. Required for paragraph, heading_1, heading_2, heading_3, callout, to_do, toggle, quote.')
    italic: Optional[bool] = Field(default=False, description='Indicates if the text is italic.')
    link: Optional[str] = Field(default=None, description='The URL of the rich text object or the file to be uploaded or image/video link')
    strikethrough: Optional[bool] = Field(default=False, description='Indicates if the text has strikethrough.')
    underline: Optional[bool] = Field(default=False, description='Indicates if the text is underlined.')

class PropertyValues(BaseModel):
    name: str = Field(description='Name of the property')
    type: str = Field(description='Type of the property. Type of the propertytitle, rich_text, number, select, multi_select, date, people, files, checkbox url, email, phone_number, formula, created_by, created_time, last_edited_by, last_edited_time')
    value: str = Field(description='Value of the property, it will be dependent on the type of the property\nFor types --> value should be\n- title, rich_text - text ex. "Hello World" (IMPORTANT: max 2000 characters, longer text will be truncated)\n- number - number ex. 23.4\n- select - select ex. "India"\n- multi_select - multi_select comma separated values ex. "India,USA"\n- date - format ex. "2021-05-11T11:00:00.000-04:00",\n- people - comma separated ids of people ex. "123,456" (will be converted to array of user objects)\n- relation - comma separated ids of related pages ex. "123,456" (will be converted to array of relation objects)\n- url - a url.\n- files - comma separated urls\n- checkbox - "True" or "False"\n')

@tool
def Database_Management_System_Create(database_id: str, child_blocks: Optional[List[NotionRichText]] = None, cover: Optional[str] = None, icon: Optional[str] = None, properties: Optional[List[PropertyValues]] = None) -> str:
    """
    Creates new entries for calorie data (food consumption or exercise) in the database.
    Creates a new page (row) in a specified notion database.

    Args:
        database_id (str): Identifier (UUID) of the Notion database where the new page (row) will be inserted.
        child_blocks (Optional[List[NotionRichText]]): A list of `NotionRichText` objects defining content blocks to append to the new page's body.
        cover (Optional[str]): URL of an external image to set as the page cover.
        icon (Optional[str]): Emoji to be used as the page icon.
        properties (Optional[List[PropertyValues]]): Property values for the new page.

    Returns:
        str: JSON string of the output.

    Example:
        Database_Management_System_Create(database_id="your_database_id", properties=[{"name": "Food Item", "type": "title", "value": "Apple"}, {"name": "Calories", "type": "number", "value": "95"}])
    """
    class CreateResponseData(BaseModel):
        response_data: str = Field(description="JSON string of the complete JSON object representing the newly created page (row), as returned by the Notion API.")

    class CreateResponse(BaseModel):
        data: CreateResponseData = Field(description="Data from the action execution")
        successful: bool = Field(description="Whether or not the action execution was successful or not")
        error: Optional[str] = Field(default=None, description="Error if any occurred during the execution of the action")

    input_str = f"database_id: {database_id}, child_blocks: {child_blocks}, cover: {cover}, icon: {icon}, properties: {properties}"
    description = Database_Management_System_Create.__doc__

    result = llm.with_structured_output(CreateResponse).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

class Sort(BaseModel):
    ascending: bool = Field(description='True = ASC, False = DESC.')
    property_name: str = Field(description='Database column to sort by.')

@tool
def Database_Management_System_Read(database_id: str, page_size: Optional[int] = 2, sorts: Optional[List[Sort]] = None, start_cursor: Optional[str] = None) -> str:
    """
    Reads historical calorie data (food consumption or exercise) from the database.
    Queries a notion database for pages (rows), where rows are pages and columns are properties; ensure sort property names correspond to existing database properties.

    Args:
        database_id (str): Identifier of the Notion database to query.
        page_size (Optional[int]): The maximum number of items (pages or rows) to return in a single response. Defaults to 2.
        sorts (Optional[List[Sort]]): List of sort rules.
        start_cursor (Optional[str]): An opaque cursor for pagination, used to retrieve the next set of results.

    Returns:
        str: JSON string of the output.

    Example:
        Database_Management_System_Read(database_id="your_database_id", page_size=10, sorts=[{"property_name": "Date", "ascending": False}])
    """
    class ReadResponseData(BaseModel):
        response_data: str = Field(description="JSON string of the dictionary containing the queried data from the Notion database.")

    class ReadResponse(BaseModel):
        data: ReadResponseData = Field(description="Data from the action execution")
        successful: bool = Field(description="Whether or not the action execution was successful or not")
        error: Optional[str] = Field(default=None, description="Error if any occurred during the execution of the action")

    input_str = f"database_id: {database_id}, page_size: {page_size}, sorts: {sorts}, start_cursor: {start_cursor}"
    description = Database_Management_System_Read.__doc__

    result = llm.with_structured_output(ReadResponse).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

@tool
def Database_Management_System_Update(row_id: str, cover: Optional[str] = None, delete_row: Optional[bool] = False, icon: Optional[str] = None, properties: Optional[List[PropertyValues]] = None) -> str:
    """
    Updates existing calorie data entries in the database.
    Updates or archives an existing notion database row (page) using its `row id`, allowing modification of its icon, cover, and/or properties.

    Args:
        row_id (str): Identifier (UUID) of the database row (page) to be updated.
        cover (Optional[str]): URL of an external image to be used as the cover for the page.
        delete_row (Optional[bool]): If true, the row (page) will be archived. If false, the row will be updated.
        icon (Optional[str]): The emoji to be used as the icon for the page.
        properties (Optional[List[PropertyValues]]): A list of property values to update for the page.

    Returns:
        str: JSON string of the output.

    Example:
        Database_Management_System_Update(row_id="your_row_id", properties=[{"name": "Calories", "type": "number", "value": "100"}])
    """
    class UpdateResponseData(BaseModel):
        response_data: str = Field(description="JSON string of the raw JSON response from the Notion API, representing the updated page object.")

    class UpdateResponse(BaseModel):
        data: UpdateResponseData = Field(description="Data from the action execution")
        successful: bool = Field(description="Whether or not the action execution was successful or not")
        error: Optional[str] = Field(default=None, description="Error if any occurred during the execution of the action")

    input_str = f"row_id: {row_id}, cover: {cover}, delete_row: {delete_row}, icon: {icon}, properties: {properties}"
    description = Database_Management_System_Update.__doc__

    result = llm.with_structured_output(UpdateResponse).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

'''