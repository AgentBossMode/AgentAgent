stock_mock_tools = r'''
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

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
        examples=[
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
        description='Value of the property, it will be dependent on the type of the property\nFor types --> value should be\n- title, rich_text - text ex. "Hello World" (IMPORTANT: max 2000 characters, longer text will be truncated)\n- number - number ex. 23.4\n- select - select ex. "India"\n- multi_select - multi_select comma separated values ex. "India,USA"\n- date - format ex. "2021-05-11T11:00:00.000-04:00",\n- people - comma separated ids of people ex. "123,456" (will be converted to array of user objects)\n- relation - comma separated ids of related pages ex. "123,456" (will be converted to array of relation objects)\n- url - a url.\n- files - comma separated urls\n- checkbox - "True" or "False"\n',
    )

class Sort(BaseModel):
    property_name: str = Field(description="Database column to sort by.", examples=["Task Name"])
    ascending: bool = Field(description="True = ASC, False = DESC.", examples=[True])

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
def Investment_Database_CREATE(
    database_id: str,
    properties: Optional[List[PropertyValues]] = None,
    child_blocks: Optional[List[NotionRichText]] = None,
    cover: Optional[str] = None,
    icon: Optional[str] = None,
) -> str:
    """
    Creates a new page (row) in a specified notion database.

    Args:
        database_id (str): Identifier (UUID) of the Notion database where the new page (row) will be inserted.
        properties (Optional[List[PropertyValues]]): Property values for the new page.
        child_blocks (Optional[List[NotionRichText]]): A list of NotionRichText objects defining content blocks.
        cover (Optional[str]): URL of an external image to set as the page cover.
        icon (Optional[str]): Emoji to be used as the page icon.

    Returns:
        str: JSON string of the output.

    Example:
        Investment_Database_CREATE(database_id="your_database_id", properties=[{"name": "Stock", "type": "title", "value": "AAPL"}, {"name": "Shares", "type": "number", "value": "10"}])
    """

    class InvestmentDatabaseCreateOutput(BaseModel):
        successful: bool = Field(description="Whether or not the action execution was successful or not")
        data: str = Field(description="JSON string of the output data from the action execution")
        error: Optional[str] = Field(description="Error if any occurred during the execution of the action")

    input_str = f"database_id: {database_id}, properties: {properties}, child_blocks: {child_blocks}, cover: {cover}, icon: {icon}"
    description = Investment_Database_CREATE.__doc__

    result = llm.with_structured_output(InvestmentDatabaseCreateOutput).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

@tool
def Investment_Database_READ(
    database_id: str,
    page_size: Optional[int] = 2,
    sorts: Optional[List[Sort]] = None,
    start_cursor: Optional[str] = None,
) -> str:
    """
    Queries a notion database for pages (rows), where rows are pages and columns are properties; ensure sort property names correspond to existing database properties.

    Args:
        database_id (str): Identifier of the Notion database to query.
        page_size (Optional[int]): The maximum number of items (pages or rows) to return.
        sorts (Optional[List[Sort]]): List of sort rules.
        start_cursor (Optional[str]): An opaque cursor for pagination.

    Returns:
        str: JSON string of the output.

    Example:
        Investment_Database_READ(database_id="your_database_id", page_size=10, sorts=[{"property_name": "Date", "ascending": False}])
    """

    class InvestmentDatabaseReadOutput(BaseModel):
        successful: bool = Field(description="Whether or not the action execution was successful or not")
        data: str = Field(description="JSON string of the output data from the action execution")
        error: Optional[str] = Field(description="Error if any occurred during the execution of the action")

    input_str = f"database_id: {database_id}, page_size: {page_size}, sorts: {sorts}, start_cursor: {start_cursor}"
    description = Investment_Database_READ.__doc__

    result = llm.with_structured_output(InvestmentDatabaseReadOutput).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

@tool
def Stock_Market_Data_API_READ() -> str:
    """
    To get real-time or near real-time stock prices.

    This tool connects to a real-time equities client to subscribe to stock quotes and trades for specified symbols.
    It processes incoming data for a set duration and then disconnects.

    Returns:
        str: A JSON string containing a summary of the stock market data, including trade, ask, bid, and backlog counts.

    Example:
        Stock_Market_Data_API_READ()
    """

    class StockMarketDataOutput(BaseModel):
        trade_count: int = Field(description="Number of trades received.")
        ask_count: int = Field(description="Number of ask quotes received.")
        bid_count: int = Field(description="Number of bid quotes received.")
        backlog_count: int = Field(description="Number of items in the backlog.")
        message: str = Field(description="A descriptive message about the data retrieval.")

    description = Stock_Market_Data_API_READ.__doc__
    input_str = "No specific input for this tool, it initiates a data stream."

    result = llm.with_structured_output(StockMarketDataOutput).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)

@tool
def Notification_Service_CREATE(recipient: str, message: str) -> str:
    """
    To send notifications to users.

    This tool provides a multi-provider notification service that can send messages
    via SendGrid, Twilio, Telegram, and OneSignal. It abstracts the underlying
    notification mechanisms and allows sending a message to a specified recipient.

    Args:
        recipient (str): The identifier for the recipient (e.g., email address, phone number, chat ID, player ID).
        message (str): The content of the notification message.

    Returns:
        str: A JSON string indicating the success or failure of the notification attempt.

    Example:
        Notification_Service_CREATE(recipient='user@example.com', message='Your stock alert for AAPL has been triggered!')
    """

    class NotificationServiceOutput(BaseModel):
        status: str = Field(description="The status of the notification attempt (e.g., 'success', 'failed').")
        recipient: str = Field(description="The recipient to whom the notification was attempted to be sent.")
        message: str = Field(description="The message that was attempted to be sent.")
        details: Optional[str] = Field(description="Additional details about the notification attempt, if any.")

    description = Notification_Service_CREATE.__doc__
    input_str = f"recipient: {recipient}, message: {message}"

    result = llm.with_structured_output(NotificationServiceOutput).invoke(
        [
            SystemMessage(content=MOCK_TOOL_PROMPT),
            HumanMessage(content=INPUT_PROMPT.format(input=input_str, description=description))
        ]
    )
    return result.model_dump_json(indent=2)
'''
