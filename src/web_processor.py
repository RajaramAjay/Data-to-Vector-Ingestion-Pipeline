#src/web_processor.py
import sys, os, re, time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import Optional, List, Tuple, Dict
from langchain.schema import Document
from src.interfaces import DocumentProcessor
from src.utils import get_logger
logger = get_logger(__name__)
import toml
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.toml'))
config = toml.load(config_path)

# ---- WebProcessor Implementation ----

class WebProcessor(DocumentProcessor):
    def __init__(self):
        self.max_links = config['webscraper']['max_links']
    @staticmethod
    def normalize_url(self,base_url: str, link: str) -> Optional[str]:
        full_url = urljoin(base_url, link)
        if not full_url.startswith(('http://', 'https://')):
            return None
        parsed = urlparse(full_url)
        if any(ext in parsed.path for ext in ['.jpg', '.png', '.gif', '.pdf', '.zip']):
            return None
        if parsed.netloc != urlparse(base_url).netloc:
            return None
        return full_url
    @staticmethod
    def get_filtered_links(self, url: str, max_links: int = 100) -> List[str]:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            links = []
            for a_tag in soup.find_all('a', href=True):
                normalized_url = self.normalize_url(self, url, a_tag['href'])
                if normalized_url and normalized_url not in links:
                    links.append(normalized_url)
                    if len(links) >= max_links:
                        break
            return links
        except Exception as e:
            logger.error(f"Error fetching links from {url}: {e}")
            return []
    @staticmethod
    def extract_page_content(url: str) -> Tuple[str, Dict[str, str]]:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            metadata = {
                "source": url,
                "title": soup.title.string if soup.title else "No Title",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()

            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile('content|main'))
            if main_content:
                content_elements = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
                content = "\n\n".join([elem.get_text(strip=True) for elem in content_elements])
            else:
                content = soup.body.get_text(separator="\n\n", strip=True) if soup.body else ""

            content = re.sub(r'\s+', ' ', content).strip()

            return content, metadata
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return "", {"url": url, "error": str(e)}
        
    @staticmethod  
    def create_langchain_documents(self, url_contents: List[Tuple[str, Dict]]) -> List[Document]:
        documents = []
        for content, metadata in url_contents:
            if content:
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
        return documents
        

    def process(self, source: str) -> List[Document]:
        logger.info(f"Starting processing for: {source}")
        inner_urls = self.get_filtered_links(self,source, max_links=self.max_links)
        logger.info(f"Found {len(inner_urls)} valid links to process")

        url_contents = []
        for url in inner_urls:
            logger.info(f"Processing: {url}")
            content, metadata = self.extract_page_content(url)
            if content:
                url_contents.append((content, metadata))

        documents = self.create_langchain_documents(self,url_contents)
        return documents, len(inner_urls)
    





# processor = WebProcessor()

# # Process a URL and get LangChain documents
# documents = processor.process("https://apple.com")
# print(documents[4].met)