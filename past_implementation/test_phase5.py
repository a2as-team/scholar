"""Test suite for ScholarVerse Phase 5: Deep Search Implementation.

This module contains tests for the Deep Search Agent and its related tools.
"""

import unittest
import json
import os
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime
import asyncio

# Import the modules to test
from scholar_verse.sub_agents.deep_search.agent import DeepSearchAgent
from scholar_verse.sub_agents.deep_search.tools.web_search import WebSearchTool
from scholar_verse.sub_agents.deep_search.tools.web_scraper import WebScraperTool
from scholar_verse.sub_agents.deep_search.tools.validation import CitationValidationTool
from scholar_verse.sub_agents.deep_search.tools.content_extraction import ContentExtractionTool
from scholar_verse.sub_agents.deep_search.tools.rag_integration import RAGManager,RAGRetrievalTool


class TestDeepSearchAgent(unittest.TestCase):
    def setUp(self):
        self.agent = DeepSearchAgent()

    async def test_search_functionality_async(self):
        query = "artificial intelligence ethics"
        result = await self.agent.search(query, search_type="academic")
        self.assertTrue(result['success'])
        self.assertEqual(result['query'], query)
        self.assertIn('results', result)
        self.assertIn('timestamp', result)

    async def test_validate_citations_async(self):
        citations = [
            "10.1234/ai.ethics.2020",
            "Smith, J. (2020). AI Ethics. Journal of AI, 45(2), 123-145."
        ]
        result = await self.agent.validate_citations(citations)
        self.assertTrue(result['success'])
        self.assertEqual(result['total_citations'], len(citations))
        self.assertIn('validation_results', result)

    def test_search_functionality(self):
        asyncio.run(self.test_search_functionality_async())

    def test_validate_citations(self):
        asyncio.run(self.test_validate_citations_async())
class TestWebSearchTool(unittest.TestCase):
    """Test cases for the Web Search Tool."""
    
    def setUp(self):
        """Set up test environment."""
        self.web_search = WebSearchTool()
        self.mock_context = MagicMock()
        self.mock_context.state = {}
    
    def test_search_academic(self):
        """Test academic search functionality."""
        result = self.web_search._call(
            self.mock_context,
            query="artificial intelligence",
            search_type="academic",
            max_results=3
        )
        
        # Assertions
        self.assertTrue(result['success'])
        self.assertEqual(result['query'], "artificial intelligence")
        self.assertEqual(result['search_type'], "academic")
        self.assertEqual(len(result['results']), 3)
        
        # Verify search history was tracked
        self.assertIn('search_history', self.mock_context.state)
        self.assertEqual(len(self.mock_context.state['search_history']), 1)
    
    def test_search_with_domain_filtering(self):
        """Test search with domain filtering."""
        include_domains = ["academic.example.org", "scholar.example.com"]
        
        result = self.web_search._call(
            self.mock_context,
            query="machine learning",
            search_type="general",
            max_results=5,
            include_domains=include_domains
        )
        
        # Check if any results were returned
        self.assertTrue(result['success'])
        
        # For each result, check if its domain is in the include_domains list
        for item in result['results']:
            domain = item['url'].split('/')[2]  # Extract domain from URL
            domain_match = any(incl_domain in domain for incl_domain in include_domains)
            self.assertTrue(domain_match, f"Domain {domain} not in include_domains list")


class TestWebScraperTool(unittest.TestCase):
    """Test cases for the Web Scraper Tool."""
    
    def setUp(self):
        """Set up test environment."""
        self.web_scraper = WebScraperTool()
        self.mock_context = MagicMock()
        
        # Patch the _fetch_url method to avoid actual web requests
        self.fetch_patcher = patch.object(WebScraperTool, '_fetch_url')
        self.mock_fetch = self.fetch_patcher.start()
        
        # Patch BeautifulSoup with a mock implementation
        self.bs_patcher = patch('scholar_verse.sub_agents.deep_search.tools.web_scraper.BeautifulSoup')
        self.mock_bs = self.bs_patcher.start()
        
        # Configure the mock BeautifulSoup
        mock_soup = MagicMock()
        mock_title = MagicMock()
        mock_title.string = "AI Ethics Research"
        mock_soup.title = mock_title
        mock_soup.get_text.return_value = "This paper discusses ethical considerations in artificial intelligence development."
        
        # Mock the find_all method
        mock_content = MagicMock()
        mock_content.get_text.return_value = "Ethical Considerations in AI content"
        mock_soup.find_all.return_value = [mock_content]
        
        # Mock the body property
        mock_soup.body = MagicMock()
        mock_soup.body.get_text.return_value = "Body content about AI ethics"
        
        self.mock_bs.return_value = mock_soup
    
    def tearDown(self):
        """Tear down test environment."""
        self.fetch_patcher.stop()
        self.bs_patcher.stop()
    
    def test_scrape_url(self):
        """Test URL scraping functionality."""
        # Mock HTML content
        mock_html = """
        <html>
            <head><title>AI Ethics Research</title></head>
            <body>
                <article>
                    <h1>Ethical Considerations in AI</h1>
                    <p>This paper discusses ethical considerations in artificial intelligence development.</p>
                    <p>Citation: Smith, J. (2020). AI Ethics.</p>
                    <a href="https://example.com/related">Related Research</a>
                </article>
            </body>
        </html>
        """
        
        # Configure mock
        self.mock_fetch.return_value = {
            'success': True,
            'content': mock_html,
            'status_code': 200,
            'headers': {'content-type': 'text/html'}
        }
        
        # Call the scraper
        result = self.web_scraper._call(
            self.mock_context,
            url="https://example.com/ai-ethics",
            extraction_type="full",
            depth=0
        )
        
        # Assertions
        self.assertTrue(result['success'])
        self.assertEqual(result['url'], "https://example.com/ai-ethics")
        self.assertEqual(result['content']['title'], "AI Ethics Research")
        self.assertIn("ethical considerations", result['content']['text'].lower())
    
    def test_validate_url(self):
        """Test URL validation."""
        # Valid URL
        self.assertTrue(self.web_scraper._is_valid_url("https://example.com"))
        
        # Invalid URLs
        self.assertFalse(self.web_scraper._is_valid_url("not-a-url"))
        self.assertFalse(self.web_scraper._is_valid_url("http://"))


class TestCitationValidationTool(unittest.TestCase):
    """Test cases for the Citation Validation Tool."""
    
    def setUp(self):
        """Set up test environment."""
        self.citation_validator = CitationValidationTool()
        self.mock_context = MagicMock()
        self.mock_context.state = {}
    
    def test_validate_doi(self):
        """Test DOI validation."""
        doi_citation = "10.1234/ai.ethics.2020"
        
        result = self.citation_validator._call(
            self.mock_context,
            citation=doi_citation,
            validation_level="standard"
        )
        
        # Assertions
        self.assertTrue(result['success'])
        self.assertEqual(result['citation'], doi_citation)
        self.assertEqual(result['citation_type'], "doi")
        self.assertTrue(result['validation_result']['is_valid'])
    
    def test_validate_formatted_citation(self):
        """Test formatted citation validation."""
        formatted_citation = "Smith, J. (2020). AI Ethics. Journal of AI, 45(2), 123-145."
        source_url = "https://journal.example.com/article"
        
        result = self.citation_validator._call(
            self.mock_context,
            citation=formatted_citation,
            source_url=source_url,
            validation_level="thorough"
        )
        
        # Assertions
        self.assertTrue(result['success'])
        self.assertEqual(result['citation'], formatted_citation)
        self.assertIn('validation_result', result)
        self.assertIn('extracted_authors', result['validation_result'])
        self.assertIn('extracted_year', result['validation_result'])
    
    def test_citation_tracking(self):
        """Test citation tracking functionality."""
        # Validate multiple citations
        self.citation_validator._call(
            self.mock_context,
            citation="10.1234/ai.ethics.2020"
        )
        self.citation_validator._call(
            self.mock_context,
            citation="Smith, J. (2020). AI Ethics."
        )
        
        # Check if citations were tracked
        self.assertIn('citations', self.mock_context.state)
        self.assertEqual(len(self.mock_context.state['citations']['tracked']), 2)
        
        # Test stats tracking
        stats = self.citation_validator.get_citation_stats(self.mock_context)
        self.assertTrue(stats['success'])
        self.assertEqual(stats['total_tracked'], 2)


class TestContentExtractionTool(unittest.TestCase):
    """Test cases for the Content Extraction Tool."""
    
    def setUp(self):
        """Set up test environment."""
        # Create content extraction tool
        self.content_extractor = ContentExtractionTool()
        self.mock_context = MagicMock()
        
        # Mock sent_tokenize functionality directly in the content extractor
        # This is necessary because our tests may not have nltk installed
        sent_tokenize_patcher = patch('scholar_verse.sub_agents.deep_search.tools.content_extraction.sent_tokenize')
        self.mock_sent_tokenize = sent_tokenize_patcher.start()
        self.mock_sent_tokenize.side_effect = self._mock_sent_tokenize_func
        self.addCleanup(sent_tokenize_patcher.stop)
        
        # Sample content for testing
        self.test_content = """
        # Ethical Considerations in AI Development
        
        By Smith, J. (2020)
        
        ## Abstract
        This paper discusses important ethical considerations in artificial intelligence development.
        
        ## Introduction
        Artificial Intelligence (AI) has seen rapid advancement in recent years. This progress brings
        significant ethical challenges that researchers and practitioners must address.
        
        ## Key Issues
        Transparency, fairness, and privacy are fundamental ethical concerns in AI systems.
        
        ## References
        [1] Johnson, A. (2019). AI Ethics Framework.
        [2] Williams, B. et al. (2018). Privacy in Machine Learning.
        DOI: 10.1234/privacy.ml.2018
        """
    
    def _mock_sent_tokenize_func(self, text):
        """Mock implementation of sent_tokenize"""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def test_extract_key_information(self):
        """Test key information extraction."""
        result = self.content_extractor._call(
            self.mock_context,
            content=self.test_content,
            extraction_type="key_information"
        )
        
        # Assertions
        self.assertTrue(result['success'])
        self.assertEqual(result['extraction_type'], "key_information")
        self.assertIn('important_sentences', result['result'])
        self.assertIn('key_terms', result['result'])
    
    def test_extract_summary(self):
        """Test summary extraction."""
        result = self.content_extractor._call(
            self.mock_context,
            content=self.test_content,
            extraction_type="summary"
        )
        
        # Assertions
        self.assertTrue(result['success'])
        self.assertEqual(result['extraction_type'], "summary")
        self.assertIn('intro_summary', result['result'])
        self.assertIn('conclusion_summary', result['result'])
        self.assertIn('full_summary', result['result'])
    
    def test_extract_citations(self):
        """Test citation extraction."""
        result = self.content_extractor._call(
            self.mock_context,
            content=self.test_content,
            extraction_type="citations"
        )
        
        # Assertions
        self.assertTrue(result['success'])
        self.assertEqual(result['extraction_type'], "citations")
        self.assertIn('dois', result['result'])
        self.assertIn('author_year_citations', result['result'])
        self.assertIn('reference_citations', result['result'])
        
        # Verify extracted DOI
        self.assertIn("10.1234/privacy.ml.2018", result['result']['dois'])


class TestRAGIntegration(unittest.TestCase):
    """Test cases for the RAG Integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock get_config
        self.config_patcher = patch('scholar_verse.sub_agents.deep_search.tools.rag_integration.get_config')
        self.mock_get_config = self.config_patcher.start()
        self.mock_get_config.return_value = {
            'rag': {
                'default_model': 'text-embedding-004',
                'max_retrieval_results': 10,
                'min_relevance_score': 0.5
            }
        }
        
        # Mock AdaptiveStateManager
        self.state_manager_patcher = patch('scholar_verse.sub_agents.deep_search.tools.rag_integration.AdaptiveStateManager')
        self.mock_state_manager = self.state_manager_patcher.start()
        self.mock_state_manager_instance = MagicMock()
        self.mock_state_manager.return_value = self.mock_state_manager_instance
        self.mock_state_manager_instance.get.return_value = {}
        self.mock_state_manager_instance.set.return_value = None
        
        # Initialize RAGManager with mocked state manager
        self.rag_manager = RAGManager(state_manager=self.mock_state_manager_instance)
        
        # Initialize RAGRetrievalTool
        self.rag_tool = self.rag_manager.create_retrieval_tool()
        
        # Create a mock ToolContext
        self.mock_context = MagicMock()
        self.mock_context.state = {}
    
    def test_retrieve_information(self):
        """Test information retrieval using RAGRetrievalTool."""
        query = "ethical considerations in AI"
        
        # Call the tool with the mock context, using query_context
        result = self.rag_tool._call(
            self.mock_context,
            query=query,
            query_context={"domain": "academic"}  # Renamed from 'context'
        )
        
        # Assertions
        self.assertTrue(result['success'])
        self.assertIn('results', result)
        self.assertIn('metadata', result)
        self.assertIn('query_id', result)
        self.assertGreaterEqual(len(result['results']), 1)
    
    def test_retrieval_history(self):
        """Test retrieval history tracking using RAGRetrievalTool."""
        # Perform multiple retrievals
        self.rag_tool._call(
            self.mock_context,
            query="ethical considerations in AI",
            query_context={"domain": "academic"}  # Renamed from 'context'
        )
        self.rag_tool._call(
            self.mock_context,
            query="privacy in machine learning",
            query_context={"domain": "academic"}  # Renamed from 'context'
        )
        
        # Get history
        history = self.rag_tool.get_retrieval_history(self.mock_context)
        
        # Assertions
        self.assertTrue(history['success'])
        self.assertEqual(len(history['queries']), 2)
        
        # Test clearing history
        clear_result = self.rag_manager.state_manager.clear_retrieval_history(self.mock_context)
        self.assertTrue(clear_result['success'])
        
        # Verify history is cleared
        empty_history = self.rag_manager.state_manager.get_retrieval_history(self.mock_context)
        self.assertEqual(len(empty_history['queries']), 0)
    
    def tearDown(self):
        """Tear down test environment."""
        self.config_patcher.stop()
        self.state_manager_patcher.stop()
if __name__ == '__main__':
    unittest.main()
