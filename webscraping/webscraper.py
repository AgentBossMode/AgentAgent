import aiohttp
import asyncio
from bs4 import BeautifulSoup
import logging
import json
from urllib.parse import urljoin


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

async def scrape_links(url, baseurl, visited):
    if url in visited:
        logger.debug(f"Skipping already visited URL: {url}")
        return set()
    
    visited.add(url)
    links = set()

    try:
        logger.debug(f"Fetching URL: {url}")
        timeout = aiohttp.ClientTimeout(total=10)  # Set timeout to 3 seconds
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout= timeout) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {url}: Status code {response.status}")
                    return links

                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'html.parser')

                for a_tag in soup.find_all('a', href=True):
                    link = a_tag['href']
                    link = urljoin(url, link)
                    if link.__contains__('#'):
                        [start, end] = link.split('#')
                        logger.debug(f"start is {start} and end is {end}")
                        if start in visited:
                            continue
                        else:
                            link = start                  
                            
                    if link.__contains__("/v0.1") or link.__contains__("/v0.2") or link.startswith("https://python.langchain.com/api_reference"):
                        continue

                    if link.startswith(baseurl) and link not in visited:
                        links.add(link)
                        
        tasks = [scrape_links(link, baseurl, visited) for link in links]
        results = await asyncio.gather(*tasks)
        for result in results:
            links.update(result)
        logger.debug(f'Links found for {url}: {links}')
        return links
    except Exception as e:
        logger.error(f"An error occurred while processing {url}: {e}")
    
    
    return links

async def main_task():
    start_url = "https://python.langchain.com/"
    base_url = "https://python.langchain.com/"
    visited_urls = set()

    logger.info("Starting the web scraping process...")
    await scrape_links(start_url, base_url, visited_urls)
    
    logger.info("Collected Links:")
    with open("visited_urls_langchain.json", "w") as f:
        visited_list = list(visited_urls)
        visited_list.sort()
        json.dump(visited_list, f)

if __name__ == "__main__":
    try:
        asyncio.run(main_task())
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
