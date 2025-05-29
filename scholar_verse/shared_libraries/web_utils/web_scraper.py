"""Web scraping utilities for ScholarVerse."""

import requests
from typing import Dict, Any, Optional
from bs4 import BeautifulSoup

from scholar_verse.shared_libraries.logging_utils import logger


class WebScraper:
    """Web scraper for ScholarVerse.
    
    This class provides basic web scraping capabilities for the ScholarVerse agents.
    It will be expanded in Phase 5 to include more sophisticated scraping capabilities.
    """
    
    def __init__(self, user_agent: Optional[str] = None):
        """Initialize the web scraper.
        
        Args:
            user_agent: The user agent to use for requests, or None to use the default.
        """
        self.session = requests.Session()
        self.user_agent = user_agent or "ScholarVerse/0.1.0 (Research Assistant)"
        self.session.headers.update({"User-Agent": self.user_agent})
        logger.info("Initialized web scraper")
    
    def fetch_page(self, url: str) -> Optional[str]:
        """Fetch a web page.
        
        Args:
            url: The URL to fetch.
            
        Returns:
            The page content as a string, or None if the request failed.
        """
        try:
            logger.info(f"Fetching page: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching page {url}: {e}")
            return None
    
    def extract_text(self, html: str) -> str:
        """Extract text from HTML.
        
        Args:
            html: The HTML content to extract text from.
            
        Returns:
            The extracted text.
        """
        soup = BeautifulSoup(html, "html.parser")
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def extract_links(self, html: str, base_url: str) -> list:
        """Extract links from HTML.
        
        Args:
            html: The HTML content to extract links from.
            base_url: The base URL to resolve relative links against.
            
        Returns:
            A list of links.
        """
        soup = BeautifulSoup(html, "html.parser")
        links = []
        
        for link in soup.find_all('a'):
            href = link.get('href')
            if href:
                # Handle relative URLs
                if href.startswith('/'):
                    href = f"{base_url.rstrip('/')}{href}"
                elif not href.startswith(('http://', 'https://')):
                    href = f"{base_url.rstrip('/')}/{href.lstrip('/')}"
                
                links.append({
                    "url": href,
                    "text": link.text.strip(),
                })
        
        return links


# Create a default web scraper
web_scraper = WebScraper()
