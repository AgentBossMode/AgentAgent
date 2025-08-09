from bs4 import BeautifulSoup
import html2text
import httpx

def fetch_documents(url: str) -> str:
    """Fetch a document from a URL and return the markdownified text.

    Args:
        url (str): The URL of the document to fetch.

    Returns:
        str: The markdownified text of the document.
    """
    httpx_client = httpx.Client(follow_redirects=True, timeout=10)

    try:
        response = httpx_client.get(url, timeout=10)
        response.raise_for_status()
        html_content = response
        soup = BeautifulSoup(html_content, 'html.parser')
    
        img_tags = soup.find_all('img')
        for img_tag in img_tags:
            img_tag.decompose()

        target_div = soup.find('div', class_= "theme-doc-markdown markdown") #langchain
        
        if not target_div:
            target_div = soup.find('article') #langraph

        if not target_div:
            target_div = soup.find('html') #langraph

        if not target_div:
            return html2text.html2text(str(soup))
        
        return html2text.html2text(str(target_div))
    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        return f"Encountered an HTTP error: {str(e)}"