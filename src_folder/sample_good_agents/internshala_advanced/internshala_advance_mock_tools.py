from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@tool
def Web_Scraper_Parse(url: str) -> str:
    """
    Parses HTML content from a given URL.

    Args:
        url (str): The URL to parse.

    Returns:
        str: The prettified HTML content if successful, otherwise an error message.

    Example:
        Web_Scraper_Parse(url="https://example.com")
    """

    class HTMLContent(BaseModel):
        html_content: str = Field(description="The prettified HTML content of the webpage.")

    input_str = f"url: {url}"
    description = Web_Scraper_Parse.__doc__

    result = llm.with_structured_output(HTMLContent).invoke(
        [
            SystemMessage(content="""
You are a helpful assistant that generates mock data for tool outputs.
Given the tool's purpose and expected output, generate a realistic mock response.
"""),
            HumanMessage(content=f"""
Tool Docstring: {description}
Input: {input_str}
Generate a mock output for this tool.
""")
        ]
    )
    return result.model_dump_json(indent=2)

@tool
def Web_Scraper_Extract(url: str, extract_rules: dict) -> str:
    """
    Extracts specific data from parsed HTML content based on a schema.

    Args:
        url (str): The URL to scrape.
        extract_rules (dict): A dictionary of CSS selectors to extract data.

    Returns:
        str: The extracted content.

    Example:
        Web_Scraper_Extract(url="https://example.com", extract_rules={"title": "h1", "description": "p.description"})
    """

    class ExtractedContent(BaseModel):
        extracted_data: str = Field(description="The extracted content based on the provided rules, in JSON format.")

    input_str = f"url: {url}, extract_rules: {extract_rules}"
    description = Web_Scraper_Extract.__doc__

    result = llm.with_structured_output(ExtractedContent).invoke(
        [
            SystemMessage(content="""
You are a helpful assistant that generates mock data for tool outputs.
Given the tool's purpose and expected output, generate a realistic mock response.
"""),
            HumanMessage(content=f"""
Tool Docstring: {description}
Input: {input_str}
Generate a mock output for this tool.
""")
        ]
    )
    return result.model_dump_json(indent=2)

def Google_Sheets_Read(spreadsheet_id: str, ranges: Optional[List[str]] = None) -> str:
    """
    Retrieves data from specified cell ranges in a google spreadsheet; ensure the spreadsheet has at least one worksheet and any explicitly referenced sheet names in ranges exist.

    Args:
        spreadsheet_id (str): The unique identifier of the Google Spreadsheet from which data will be retrieved.
        ranges (Optional[List[str]]): A list of cell ranges in A1 notation (e.g., 'Sheet1!A1:B2', 'A1:C5') from which to retrieve data. If this list is omitted or empty, all data from the first sheet of the spreadsheet will be fetched. A range can specify a sheet name (e.g., 'Sheet2!A:A'); if no sheet name is provided in a range string (e.g., 'A1:B2'), it defaults to the first sheet.

    Returns:
        str: A JSON string containing the retrieved data, including the spreadsheetId and valueRanges.

    Example:
        Google_Sheets_Read(spreadsheet_id="your_spreadsheet_id", ranges=["Sheet1!A1:B2"])
    """

    class SpreadsheetData(BaseModel):
        spreadsheetId: str = Field(description="The ID of the spreadsheet.")
        valueRanges: str = Field(description="A JSON string representing a list of value range objects, each containing a range and its values.")

    input_str = f"spreadsheet_id: {spreadsheet_id}, ranges: {ranges}"
    description = Google_Sheets_Read.__doc__

    result = llm.with_structured_output(SpreadsheetData).invoke(
        [
            SystemMessage(content="""
You are a helpful assistant that generates mock data for tool outputs.
Given the tool's purpose and expected output, generate a realistic mock response.
"""),
            HumanMessage(content=f"""
Tool Docstring: {description}
Input: {input_str}
Generate a mock output for this tool.
""")
        ]
    )
    return result.model_dump_json(indent=2)

def Google_Sheets_Write(spreadsheet_id: str, sheet_name: str, values: List[List[str]], first_cell_location: Optional[str] = None, includeValuesInResponse: Optional[bool] = False, valueInputOption: Optional[str] = "USER_ENTERED") -> str:
    """
    Updates a specified range in a google sheet with given values, or appends them as new rows if `first cell location` is omitted; ensure the target sheet exists and the spreadsheet contains at least one worksheet.

    Args:
        spreadsheet_id (str): The unique identifier of the Google Sheets spreadsheet to be updated.
        sheet_name (str): The name of the specific sheet within the spreadsheet to update.
        values (List[List[str]]): A 2D list of cell values. Each inner list represents a row. Values can be strings, numbers, or booleans. Ensure columns are properly aligned across rows.
        first_cell_location (Optional[str]): The starting cell for the update range, specified in A1 notation (e.g., 'A1', 'B2'). The update will extend from this cell to the right and down, based on the provided values. If omitted, values are appended to the sheet.
        includeValuesInResponse (Optional[bool]): If set to True, the response will include the updated values from the spreadsheet.
        valueInputOption (Optional[str]): How input data is interpreted. 'USER_ENTERED': Values parsed as if typed by a user (e.g., strings may become numbers/dates, formulas are calculated); recommended for formulas. 'RAW': Values stored as-is without parsing (e.g., '123' stays string, '=SUM(A1:B1)' stays string).

    Returns:
        str: A JSON string containing details of the update operation, such as spreadsheet ID, updated range, and counts of updated rows, columns, and cells.

    Example:
        Google_Sheets_Write(spreadsheet_id="your_spreadsheet_id", sheet_name="Sheet1", values=[["Name", "Age"], ["John Doe", 30]])
    """

    class UpdateResponse(BaseModel):
        spreadsheetId: str = Field(description="The ID of the spreadsheet that was updated.")
        updatedRange: str = Field(description="The range of cells that were updated.")
        updatedRows: int = Field(description="The number of rows updated.")
        updatedColumns: int = Field(description="The number of columns updated.")
        updatedCells: int = Field(description="The number of cells updated.")

    input_str = f"spreadsheet_id: {spreadsheet_id}, sheet_name: {sheet_name}, values: {values}, first_cell_location: {first_cell_location}, includeValuesInResponse: {includeValuesInResponse}, valueInputOption: {valueInputOption}"
    description = Google_Sheets_Write.__doc__

    result = llm.with_structured_output(UpdateResponse).invoke(
        [
            SystemMessage(content="""
You are a helpful assistant that generates mock data for tool outputs.
Given the tool's purpose and expected output, generate a realistic mock response.
"""),
            HumanMessage(content=f"""
Tool Docstring: {description}
Input: {input_str}
Generate a mock output for this tool.
""")
        ]
    )
    return result.model_dump_json(indent=2)

@tool
def URL_Parser_Extract_Base_URL(url: str) -> str:
    """
    Extracts the base website URL from a given job posting URL.

    Args:
        url (str): The full URL of the job posting.

    Returns:
        str: The base URL of the website.

    Example:
        URL_Parser_Extract_Base_URL(url="https://example.com/jobs/123")
    """

    class BaseURL(BaseModel):
        base_url: str = Field(description="The base URL of the website.")

    input_str = f"url: {url}"
    description = URL_Parser_Extract_Base_URL.__doc__

    result = llm.with_structured_output(BaseURL).invoke(
        [
            SystemMessage(content="""
You are a helpful assistant that generates mock data for tool outputs.
Given the tool's purpose and expected output, generate a realistic mock response.
"""),
            HumanMessage(content=f"""
Tool Docstring: {description}
Input: {input_str}
Generate a mock output for this tool.
""")
        ]
    )
    return result.model_dump_json(indent=2)

@tool
def Email_Extractor_Extract(html_content: str) -> list:
    """
    Identifies and extracts email addresses from parsed website content.

    Args:
        html_content (str): The HTML content of the webpage.

    Returns:
        list: A list of unique email addresses found in the content.

    Example:
        Email_Extractor_Extract(html_content="<html><body>Contact us at test@example.com</body></html>")
    """

    class EmailList(BaseModel):
        emails: List[str] = Field(description="A list of unique email addresses found in the content.")

    input_str = f"html_content: {html_content}"
    description = Email_Extractor_Extract.__doc__

    result = llm.with_structured_output(EmailList).invoke(
        [
            SystemMessage(content="""
You are a helpful assistant that generates mock data for tool outputs.
Given the tool's purpose and expected output, generate a realistic mock response.
"""),
            HumanMessage(content=f"""
Tool Docstring: {description}
Input: {input_str}
Generate a mock output for this tool.
""")
        ]
    )
    return result.model_dump_json(indent=2)
