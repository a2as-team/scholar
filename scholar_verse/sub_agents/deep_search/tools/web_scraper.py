"""Web Scraper Tool for Deep Search Agent.

This module provides tools for extracting content from web pages, specifically targeting scholarly content.
"""

from typing import Dict, Any, List, Optional, Tuple, Type, Union
import re
import json
import logging
import time
from datetime import datetime
import urllib.parse

# Import ADK tooling with proper error handling
try:
    from google.adk.tools import BaseTool, ToolContext, ToolParameters
    HAS_ADK = True
except ImportError:
    # For local development without ADK
    HAS_ADK = False
    
    class BaseTool:
        def __init__(self, name: str, description: str, parameters: Type['ToolParameters'] = None):
            self.name = name
            self.description = description
            self.parameters = parameters or type('ToolParameters', (), {})
        
        def _call(self, context: 'ToolContext', **kwargs) -> Any:
            raise NotImplementedError("Tool must implement _call method")
    
    class ToolContext:
        def __init__(self):
            self.state = {}
    
    class ToolParameters:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

# Import BeautifulSoup with proper error handling
try:
    from bs4 import BeautifulSoup
    HAS_BEAUTIFUL_SOUP = True
except ImportError:
    HAS_BEAUTIFUL_SOUP = False
    class BeautifulSoup:
        pass

import requests

from scholar_verse.shared_libraries.logging_utils import logger
from scholar_verse.config import get_config

# Export the tool class
__all__ = ['WebScraperTool']


class WebScraperTool(BaseTool):
    """Tool for scraping web content from scholarly sources."""
    
    def __init__(self):
        """Initialize the web scraper tool."""
        # Define parameters schema
        class WebScraperParameters(ToolParameters):
            url: str
            extraction_type: str = "full"
            depth: int = 0
        
        super().__init__(
            name="web_scraper",
            description="Scrapes content from web pages, focusing on scholarly content",
            parameters=WebScraperParameters
        )
        
        self.config = get_config()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ScholarVerse Research Assistant/1.0 (Academic Research Purpose)'
        })
    
    def _call(self, context: ToolContext, url: str, extraction_type: str = "full", depth: int = 0) -> Dict[str, Any]:
        """Execute the web scraper tool.
        
        Args:
            context: The tool context.
            url: The URL to scrape.
            extraction_type: The type of extraction to perform (full, citations, metadata, text).
            depth: The depth of links to follow from the main page.
            
        Returns:
            A dictionary containing the scraped content.
        """
        logger.info(f"Scraping URL: {url}, Type: {extraction_type}, Depth: {depth}")
        
        try:
            # Validate URL
            if not self._is_valid_url(url):
                return {
                    'success': False,
                    'error': f"Invalid URL: {url}",
                    'content': None
                }
            
            # Fetch content
            response = self._fetch_url(url)
            if not response['success']:
                return response
            
            # Parse content based on extraction type
            content = self._parse_content(response['content'], url, extraction_type)
            
            # Fetch linked content if depth > 0
            linked_content = []
            if depth > 0 and 'links' in content and content['links']:
                for i, link in enumerate(content['links'][:min(3, len(content['links']))]):  # Limit to 3 links
                    if i >= depth:
                        break
                    
                    # Only follow links from the same domain to avoid excessive scraping
                    if self._is_same_domain(url, link):
                        link_response = self._call(context, link, extraction_type, depth - 1)
                        if link_response['success']:
                            linked_content.append({
                                'url': link,
                                'content': link_response['content']
                            })
            
            # Return results
            return {
                'success': True,
                'url': url,
                'extraction_type': extraction_type,
                'content': content,
                'linked_content': linked_content if depth > 0 else [],
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'url': url,
                'content': None
            }
    def __call__(self, context: ToolContext, **kwargs) -> Any:
        return self._call(context, **kwargs)
    def _is_valid_url(self, url: str) -> bool:
        """Check if a URL is valid.
        
        Args:
            url: The URL to check.
            
        Returns:
            True if the URL is valid, False otherwise.
        """
        try:
            result = urllib.parse.urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _is_same_domain(self, url1: str, url2: str) -> bool:
        """Check if two URLs are from the same domain.
        
        Args:
            url1: The first URL.
            url2: The second URL.
            
        Returns:
            True if the URLs are from the same domain, False otherwise.
        """
        try:
            domain1 = urllib.parse.urlparse(url1).netloc
            domain2 = urllib.parse.urlparse(url2).netloc
            return domain1 == domain2
        except:
            return False
    
    def _fetch_url(self, url: str) -> Dict[str, Any]:
        """Fetch content from a URL.
        
        Args:
            url: The URL to fetch.
            
        Returns:
            A dictionary containing the fetched content.
        """
        try:
            # Add rate limiting
            time.sleep(1)  # Be respectful to servers
            
            # Fetch content
            response = self.session.get(url, timeout=10)
            
            # Check response
            if response.status_code != 200:
                return {
                    'success': False,
                    'error': f"Failed to fetch URL: {url}, Status Code: {response.status_code}",
                    'content': None
                }
            
            return {
                'success': True,
                'content': response.text,
                'status_code': response.status_code,
                'headers': dict(response.headers)
            }
        
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'content': None
            }
    
    def _parse_content(self, html_content: str, url: str, extraction_type: str) -> Dict[str, Any]:
        """Parse HTML content based on extraction type.
        
        Args:
            html_content: The HTML content to parse.
            url: The URL of the content.
            extraction_type: The type of extraction to perform.
            
        Returns:
            A dictionary containing the parsed content.
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Basic metadata
            title = soup.title.string if soup.title else "No title"
            
            # Extract main content
            main_content = self._extract_main_content(soup)
            
            # Base result
            result = {
                'title': title,
                'url': url,
                'text': main_content
            }
            
            # Add specific extractions based on type
            if extraction_type == "full" or extraction_type == "metadata":
                # Extract metadata
                meta_tags = soup.find_all('meta')
                metadata = {}
                
                for tag in meta_tags:
                    if tag.get('name'):
                        metadata[tag.get('name')] = tag.get('content')
                    elif tag.get('property'):
                        metadata[tag.get('property')] = tag.get('content')
                
                result['metadata'] = metadata
            
            if extraction_type == "full" or extraction_type == "citations":
                # Extract citations
                citations = self._extract_citations(soup, url)
                result['citations'] = citations
            
            if extraction_type == "full":
                # Extract links for potential follow-up
                links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if href.startswith('http'):
                        links.append(href)
                    elif href.startswith('/'):
                        # Convert relative URL to absolute
                        base_url = f"{urllib.parse.urlparse(url).scheme}://{urllib.parse.urlparse(url).netloc}"
                        links.append(f"{base_url}{href}")
                
                result['links'] = links
            
            return result
        
        except Exception as e:
            logger.error(f"Error parsing content from {url}: {str(e)}")
            return {
                'title': "Error parsing content",
                'text': str(e),
                'url': url
            }
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract the main content from a web page.
        
        Args:
            soup: The BeautifulSoup object.
            
        Returns:
            The extracted main content as text.
        """
        # Try to find content in common content containers
        content_elements = soup.find_all(['article', 'main', 'div', 'section'], class_=lambda c: c and any(x in str(c).lower() for x in ['content', 'article', 'main', 'text', 'body']))
        
        if content_elements:
            # Use the longest content element as the main content
            main_element = max(content_elements, key=lambda e: len(e.get_text(strip=True)))
            return main_element.get_text(separator='\n', strip=True)
        
        # Fallback to extracting from body
        if soup.body:
            # Remove script, style, header, footer, nav, ads
            for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
                element.decompose()
            
            return soup.body.get_text(separator='\n', strip=True)
        
        # Last resort: get all text
        return soup.get_text(separator='\n', strip=True)
    
    def _extract_citations(self, soup: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
        """Extract citations from a web page.
        
        Args:
            soup: The BeautifulSoup object.
            url: The URL of the page.
            
        Returns:
            A list of extracted citations.
        """
        citations = []
        
        # Look for common citation patterns in academic papers
        # DOI references
        doi_pattern = r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)'
        text = soup.get_text()
        doi_matches = re.finditer(doi_pattern, text, re.IGNORECASE)
        
        for match in doi_matches:
            doi = match.group(0)
            citations.append({
                'type': 'DOI',
                'identifier': doi,
                'url': f'https://doi.org/{doi}'
            })
        
        # Look for citation elements
        citation_elements = soup.find_all(['cite', 'div', 'p', 'li'], class_=lambda c: c and any(x in str(c).lower() for x in ['citation', 'reference', 'biblio']))
        
        for element in citation_elements:
            text = element.get_text(strip=True)
            citations.append({
                'type': 'citation_element',
                'text': text,
                'html': str(element)
            })
        
        return citations


# Export the tool class instead of an instance
__all__ = ['WebScraperTool']
