import os
import requests
from bs4 import BeautifulSoup
from scrapingbee import ScrapingBeeClient
from composio import Composio
from composio_langchain import LangchainProvider
from urllib.parse import urlparse
import asyncio
import httpx
import re
import email
from langchain_core.tools import tool

@tool
def Web_Scraper_Parse(url: str) -> str:
    """
    Parses HTML content from a given URL.

    Args:
        url (str): The URL to parse.

    Returns:
        str: The prettified HTML content if successful, otherwise an error message.
    """
    response = requests.get(url)
    if response.status_code == 200:
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.prettify()
    else:
        return f"Failed to retrieve the page. Status code: {response.status_code}"

@tool
def Web_Scraper_Extract(url: str, extract_rules: dict) -> str:
    """
    Extracts specific data from parsed HTML content based on a schema.

    Args:
        url (str): The URL to scrape.
        extract_rules (dict): A dictionary of CSS selectors to extract data.

    Returns:
        str: The extracted content.
    """
    client = ScrapingBeeClient(api_key=os.environ.get('SCRAPINGBEE_API_KEY'))
    response = client.get(
        url,
        params={
            'extract_rules': extract_rules,
        },
    )
    return response.content

composio = Composio(provider=LangchainProvider())
Google_Sheets_Read = composio.tools.get(user_id=os.environ.get("USER_ID"), tools=["GOOGLESHEETS_BATCH_GET"])
Google_Sheets_Write = composio.tools.get(user_id=os.environ.get("USER_ID"), tools=["GOOGLESHEETS_BATCH_UPDATE"])

@tool
def URL_Parser_Extract_Base_URL(url: str) -> str:
    """
    Extracts the base website URL from a given job posting URL.

    Args:
        url (str): The full URL of the job posting.

    Returns:
        str: The base URL of the website.
    """
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return base_url

@tool
def Email_Extractor_Extract(html_content: str) -> list:
    """
    Identifies and extracts email addresses from parsed website content.

    Args:
        html_content (str): The HTML content of the webpage.

    Returns:
        list: A list of unique email addresses found in the content.
    """
    emails = set()
    soup = BeautifulSoup(html_content, 'lxml')
    text_content = soup.get_text()
    EMAIL_REGEX = r"""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)])"""
    found_emails = re.findall(EMAIL_REGEX, text_content, re.IGNORECASE)
    mailto_links = soup.find_all('a', href=re.compile(r"^mailto:"))
    for link in mailto_links:
        email_address = link.get('href').split(':')[1]
        found_emails.append(email_address)
    for email_address in found_emails:
        try:
            email.utils.parseaddr(email_address)
            emails.add(email_address)
        except:
            pass
    return list(emails)