import aiohttp
import asyncio
from bs4 import BeautifulSoup
import logging
import html2text

# Set up the logger
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        # logging.FileHandler("scraper.log"),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)

async def html_to_markdown(url):
    """Convert HTML content to Markdown, focusing on a specific target div."""
    timeout = aiohttp.ClientTimeout(total=10)  # Set timeout to 3 seconds
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout= timeout) as response:
            if response.status != 200:
                logger.warning(f"Failed to fetch {url}: Status code {response.status}")
            html_content = await response.text()
            soup = BeautifulSoup(html_content, 'html.parser')
            target_div = soup.find('div', class_= "theme-doc-markdown markdown") #langchain
            
            if not target_div:
                target_div = soup.find('article') #langraph
            
            if not target_div:
                return
            
            print(target_div)
            markdown = html2text.html2text(str(target_div))

            
            # markdown = process_element(target_div)
            with open("url2.md", "w", encoding="utf-8") as f:
             f.write(markdown)
            return markdown

if __name__ == "__main__":
    try:
        #with open("visited_urls_langgraph.json", "r") as f:
        #    lanchain_list = json.load(f)
        
        asyncio.run(html_to_markdown("https://docs.smith.langchain.com/observability"))
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
