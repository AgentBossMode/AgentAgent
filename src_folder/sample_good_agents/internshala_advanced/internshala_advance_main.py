from typing import Dict, Any, List, Optional, Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver, InMemorySaver
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
import re
import json
from .internshala_advance_mock_tools import Web_Scraper_Parse, Web_Scraper_Extract, Google_Sheets_Read, Google_Sheets_Write, URL_Parser_Extract_Base_URL, Email_Extractor_Extract

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class GraphState(MessagesState):
    """ The GraphState represents the state of the LangGraph workflow.
    Below is the definition of MessagesState, the AnyMessage refers to AIMessage, HumanMessage, or SystemMessage etc.
    the add_messages is a reducer, which means that when doing return {{"messages": [AIMessage(content="...")]}}, it will append the new message to the messages variable and not override it..
    class MessagesState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
    """ 
    initial_url: str
    all_listings_html: Optional[str] = None
    internship_listing_urls: Optional[List[str]] = None
    existing_internship_data: Optional[List[dict]] = None
    new_internship_listings: Optional[List[dict]] = None
    job_posting_url: Optional[str] = None
    extracted_job_details: Optional[dict] = None
    company_website_url: Optional[str] = None
    company_website_html: Optional[str] = None
    extracted_email: Optional[str] = None
    final_internship_data: Optional[List[dict]] = None

class InternshipListingUrls(BaseModel):
    urls: List[str] = Field(description="A list of URLs for individual internship postings.")

class InternshipPostingSchema(BaseModel):
    company_name: str = Field(description="The name of the company offering the internship.")
    role: str = Field(description="The title of the internship role.")
    location: str = Field(description="The location of the internship (e.g., city, state, remote).")
    skills_required: List[str] = Field(description="A list of skills required for the internship.")
    application_link: str = Field(description="The direct link to apply for the internship.")
    description: str = Field(description="A brief description of the internship.")

def parse_initial_url(state: GraphState) -> GraphState:
    """
    Node purpose: Parses the HTML content from the provided initial website URL.
    Implementation reasoning: This node uses the Web_Scraper_Parse tool to fetch and parse the HTML of the initial URL.
                              The result is stored in 'all_listings_html' for subsequent extraction.
    """
    initial_url = state["initial_url"]
    html_content_json = Web_Scraper_Parse.invoke({"url": initial_url})
    html_content = json.loads(html_content_json)["html_content"]
    return {
        "all_listings_html": html_content,
        "messages": [AIMessage(content=f"Parsed initial URL: {initial_url}")]
    }

def extract_internship_listing_urls(state: GraphState) -> GraphState:
    """
    Node purpose: Extracts all individual internship listing URLs from the parsed HTML content.
    Implementation reasoning: This node uses an LLM with structured output to extract a list of URLs
                              from the 'all_listings_html' content. This ensures a type-safe list of URLs.
    """
    all_listings_html = state["all_listings_html"]
    
    structured_llm = llm.with_structured_output(InternshipListingUrls)
    prompt = f"Extract all internship listing URLs from the following HTML content:\n\n{all_listings_html}\n\nProvide only the URLs in a list."
    
    result: InternshipListingUrls = structured_llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "internship_listing_urls": result.urls,
        "messages": [AIMessage(content=f"Extracted {len(result.urls)} internship listing URLs.")]
    }

def read_existing_data(state: GraphState) -> GraphState:
    """
    Node purpose: Reads existing internship data from the Google Sheet.
    Implementation reasoning: This node uses the Google_Sheets_Read tool to fetch existing data,
                              which is crucial for filtering out duplicates later.
    """
    # Assuming Google_Sheets_Read returns a list of dictionaries
    existing_data_json = Google_Sheets_Read(spreadsheet_id="your_spreadsheet_id") # Assuming no specific arguments needed for a general read
    # The mock tool returns a JSON string with 'valueRanges' which is also a JSON string
    parsed_data = json.loads(existing_data_json)
    # Assuming the actual data is within 'valueRanges' and needs another json.loads
    # And assuming the structure is a list of lists, where the first list is headers
    # and subsequent lists are rows. We need to convert this to a list of dicts.
    if parsed_data and "valueRanges" in parsed_data:
        value_ranges_str = parsed_data["valueRanges"]
        value_ranges = json.loads(value_ranges_str)
        if value_ranges and "values" in value_ranges[0]: # Assuming first range contains the data
            values = value_ranges[0]["values"]
            if values:
                headers = values[0]
                existing_data = [dict(zip(headers, row)) for row in values[1:]]
            else:
                existing_data = []
        else:
            existing_data = []
    else:
        existing_data = []

    return {
        "existing_internship_data": existing_data,
        "messages": [AIMessage(content=f"Read {len(existing_data)} existing internship records from Google Sheet.")]
    }

def filter_new_listings(state: GraphState) -> GraphState:
    """
    Node purpose: Compares newly found internship listings against existing data in the Google Sheet to identify unique entries.
    Implementation reasoning: This node uses an LLM to compare the new URLs with existing data and identify
                              which ones are truly new. This prevents duplicate entries in the Google Sheet.
    """
    internship_listing_urls = state["internship_listing_urls"]
    existing_internship_data = state["existing_internship_data"]

    existing_urls = {item["application_link"] for item in existing_internship_data if "application_link" in item}
    
    new_listings = []
    for url in internship_listing_urls:
        if url not in existing_urls:
            new_listings.append({"application_link": url}) # Store as dict for consistency with later steps

    return {
        "new_internship_listings": new_listings,
        "messages": [AIMessage(content=f"Filtered down to {len(new_listings)} new internship listings.")]
    }

def extract_job_details(state: GraphState) -> GraphState:
    """
    Node purpose: Parses specific details from each individual internship posting URL, guided by the Internship Posting Schema.
    Implementation reasoning: This node iterates through 'new_internship_listings', scrapes each URL using Web_Scraper_Extract,
                              and then uses an LLM with structured output to extract details conforming to InternshipPostingSchema.
    """
    new_internship_listings = state["new_internship_listings"]
    
    all_extracted_details = []
    for listing in new_internship_listings:
        job_posting_url = listing["application_link"]
        
        # Define extract rules for Web_Scraper_Extract (example, adjust as needed)
        extract_rules = {
            "company_name": "h1.company-name::text",
            "role": "h2.job-title::text",
            "location": "span.job-location::text",
            "description": "div.job-description::text",
            "skills_required": "ul.skills li::text"
        }
        
        # Use Web_Scraper_Extract to get raw data
        raw_data_json = Web_Scraper_Extract.invoke({"url": job_posting_url, "extract_rules": extract_rules})
        raw_data = json.loads(raw_data_json)["extracted_data"]
        
        # Use LLM with structured output to parse and validate
        structured_llm = llm.with_structured_output(InternshipPostingSchema)
        prompt = f"Extract internship details from the following raw data, ensuring it conforms to the Internship Posting Schema. The application link is: {job_posting_url}\n\nRaw Data:\n{raw_data}"
        
        extracted_details: InternshipPostingSchema = structured_llm.invoke([HumanMessage(content=prompt)])
        
        all_extracted_details.append(extracted_details.model_dump()) # Convert Pydantic model to dict
    
    return {
        "new_internship_listings": all_extracted_details, # Update with full details
        "messages": [AIMessage(content=f"Extracted details for {len(all_extracted_details)} job postings.")]
    }

def parse_company_url(state: GraphState) -> GraphState:
    """
    Node purpose: Extracts the base website URL of the company from the job posting URL.
    Implementation reasoning: This node uses the URL_Parser_Extract_Base_URL tool to get the base URL,
                              which is needed for scraping the company's website for contact information.
    """
    # This node will be called for each new listing, so we need to process them iteratively
    # For simplicity, let's assume we process one at a time or update the list.
    # If this node is part of a loop, the state would need to manage the current job_posting_url.
    # For a linear graph, we'll assume it processes the first new listing's URL.
    if state["new_internship_listings"]:
        job_posting_url = state["new_internship_listings"][0]["application_link"] # Assuming we process the first one
        company_website_url_json = URL_Parser_Extract_Base_URL.invoke({"url": job_posting_url})
        company_website_url = json.loads(company_website_url_json)["base_url"]
        return {
            "company_website_url": company_website_url,
            "job_posting_url": job_posting_url, # Keep track of the current job posting URL
            "messages": [AIMessage(content=f"Extracted company website URL: {company_website_url}")]
        }
    return {
        "messages": [AIMessage(content="No new internship listings to process company URL.")]
    }


def scrape_company_website(state: GraphState) -> GraphState:
    """
    Node purpose: Scrapes the HTML content of the company's website.
    Implementation reasoning: This node uses the Web_Scraper_Parse tool to fetch the HTML of the company's website,
                              preparing it for email extraction.
    """
    company_website_url = state["company_website_url"]
    if company_website_url:
        html_content_json = Web_Scraper_Parse.invoke({"url": company_website_url})
        html_content = json.loads(html_content_json)["html_content"]
        return {
            "company_website_html": html_content,
            "messages": [AIMessage(content=f"Scraped company website: {company_website_url}")]
        }
    return {
        "messages": [AIMessage(content="No company website URL to scrape.")]
    }

def extract_email_address(state: GraphState) -> GraphState:
    """
    Node purpose: Attempts to find and extract email addresses from the parsed company website content.
    Implementation reasoning: This node uses the Email_Extractor_Extract tool to find email addresses
                              from the 'company_website_html'.
    """
    company_website_html = state["company_website_html"]
    if company_website_html:
        emails_json = Email_Extractor_Extract.invoke({"html_content": company_website_html})
        emails = json.loads(emails_json)["emails"]
        extracted_email = emails[0] if emails else None # Take the first email if found
        return {
            "extracted_email": extracted_email,
            "messages": [AIMessage(content=f"Extracted email: {extracted_email or 'None'}")]
        }
    return {
        "messages": [AIMessage(content="No company website HTML to extract email from.")]
    }

def update_google_sheet(state: GraphState) -> GraphState:
    """
    Node purpose: Stores all extracted and filtered internship data, including discovered email addresses, into the specified Google Sheet.
    Implementation reasoning: This node uses the Google_Sheets_Write tool to persist the final, enriched internship data.
    """
    new_internship_listings = state["new_internship_listings"]
    extracted_email = state["extracted_email"]
    
    # Assuming we are processing one listing at a time and adding the email to it
    # For a batch update, this logic would need to be adjusted to iterate through all new_internship_listings
    # and add the email to each relevant entry.
    if new_internship_listings and extracted_email:
        # For simplicity, let's assume the email is for the first listing processed in this cycle
        # In a real scenario, you'd need a more robust way to associate emails with specific listings.
        updated_listing = new_internship_listings[0].copy()
        updated_listing["contact_email"] = extracted_email
        final_data = [updated_listing] # Or append to existing final_internship_data
    else:
        final_data = new_internship_listings if new_internship_listings else []

    # Assuming Google_Sheets_Write expects a list of dictionaries
    # Convert list of dicts to list of lists for Google_Sheets_Write
    if final_data:
        headers = list(final_data[0].keys())
        values_to_write = [headers] + [[item[header] for header in headers] for item in final_data]
    else:
        values_to_write = []

    if values_to_write:
        Google_Sheets_Write(spreadsheet_id="your_spreadsheet_id", sheet_name="Sheet1", values=values_to_write)
    
    return {
        "final_internship_data": final_data,
        "messages": [AIMessage(content=f"Updated Google Sheet with {len(final_data)} new entries.")]
    }

def route_filter_new_listings(state: GraphState) -> str:
    """
    Routing function: Determines the next node based on whether there are new internship listings to process.
    Implementation reasoning: This function acts as a conditional router, directing the workflow
                              to either 'extract_job_details' if new listings exist or to '__END__' if not.
    """
    if state["new_internship_listings"] and len(state["new_internship_listings"]) > 0:
        return "extract_job_details"
    else:
        return "__END__"

checkpointer = InMemorySaver()
workflow = StateGraph(GraphState)

workflow.add_node("parse_initial_url", parse_initial_url)
workflow.add_node("extract_internship_listing_urls", extract_internship_listing_urls)
workflow.add_node("read_existing_data", read_existing_data)
workflow.add_node("filter_new_listings", filter_new_listings)
workflow.add_node("extract_job_details", extract_job_details)
workflow.add_node("parse_company_url", parse_company_url)
workflow.add_node("scrape_company_website", scrape_company_website)
workflow.add_node("extract_email_address", extract_email_address)
workflow.add_node("update_google_sheet", update_google_sheet)

workflow.add_edge(START, "parse_initial_url")
workflow.add_edge("parse_initial_url", "extract_internship_listing_urls")
workflow.add_edge("extract_internship_listing_urls", "read_existing_data")
workflow.add_edge("read_existing_data", "filter_new_listings")
workflow.add_conditional_edges(
    "filter_new_listings",
    route_filter_new_listings,
    {
        "extract_job_details": "extract_job_details",
        "__END__": END
    }
)
workflow.add_edge("extract_job_details", "parse_company_url")
workflow.add_edge("parse_company_url", "scrape_company_website")
workflow.add_edge("scrape_company_website", "extract_email_address")
workflow.add_edge("extract_email_address", "update_google_sheet")
workflow.add_edge("update_google_sheet", END)

app = workflow.compile(
    checkpointer=checkpointer
)