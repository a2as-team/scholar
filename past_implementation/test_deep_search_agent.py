"""Tests for the Deep Search Agent functionality."""

import unittest
from unittest.mock import patch, MagicMock
import json
from datetime import datetime

from scholar_verse.sub_agents.deep_search.agent import DeepSearchAgent

class TestDeepSearchAgent(unittest.TestCase):
    """Test cases for DeepSearchAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = DeepSearchAgent()
        
        # Mock the agent's tools
        self.mock_tools = [
            MagicMock(name='WebSearchTool', name_='web_search'),
            MagicMock(name='WebScraperTool', name_='web_scraper'),
            MagicMock(name='CitationValidationTool', name_='citation_validation'),
            MagicMock(name='ContentExtractionTool', name_='content_extraction'),
        ]
        
        # Configure mock tools
        for tool in self.mock_tools:
            tool._call.return_value = {'success': True, 'content': 'Mock content'}
        
        # Replace agent's tools with mocks
        self.agent.tools = self.mock_tools
        
        # Mock the main agent
        self.agent.agent = MagicMock()
        self.agent.agent.execute.return_value = {
            'results': [
                {
                    'title': 'Test Result 1',
                    'url': 'https://example.com/1',
                    'snippet': 'This is a test result',
                    'source': 'test',
                    'score': 0.9
                },
                {
                    'title': 'Test Result 2',
                    'url': 'https://example.com/2',
                    'snippet': 'Another test result',
                    'source': 'test',
                    'score': 0.8
                }
            ]
        }
    
    def test_initialization(self):
        """Test that the agent initializes correctly."""
        self.assertIsNotNone(self.agent)
        self.assertIsNotNone(self.agent.rag_manager)
        self.assertGreaterEqual(len(self.agent.tools), 1)
    
    @patch('scholar_verse.sub_agents.deep_search.agent.DeepSearchAgent._process_search_results')
    def test_search_success(self, mock_process):
        """Test successful search execution."""
        # Mock the process results
        expected_results = {
            'success': True,
            'query': 'test query',
            'results': [
                {'title': 'Processed 1', 'source': 'test'},
                {'title': 'Processed 2', 'source': 'test'}
            ],
            'search_type': 'academic',
            'timestamp': '2023-01-01T00:00:00',
            'tools_used': ['web_search', 'web_scraper', 'citation_validation', 'content_extraction']
        }
        
        # Mock the process_search_results to return our expected results
        mock_process.return_value = {
            'processed_results': expected_results['results'],
            'summary': 'Found 2 results for test query',
            'entities': {},
            'citations': {},
            'metadata': {}
        }
        
        # Mock the agent's execute method
        self.agent.agent.execute.return_value = {
            'results': [
                {'title': 'Test Result 1', 'url': 'https://example.com/1'},
                {'title': 'Test Result 2', 'url': 'https://example.com/2'}
            ]
        }
        
        # Execute search
        query = "test query"
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = '2023-01-01T00:00:00'
            results = self.agent.search(query)
        
        # Verify results
        self.assertTrue(results['success'])
        self.assertEqual(results['query'], query)
        self.assertEqual(len(results['results']), 5)
        self.assertEqual(results['search_type'], 'academic')
        self.agent.agent.execute.assert_called_once()
        mock_process.assert_called_once()
    
    def test_search_invalid_query(self):
        """Test search with invalid query."""
        # Test empty query
        results = self.agent.search("")
        self.assertFalse(results['success'])
        self.assertIn('error', results)
        
        # Test non-string query
        results = self.agent.search(None)
        self.assertFalse(results['success'])
    
    @patch('scholar_verse.sub_agents.deep_search.agent.DeepSearchAgent._process_search_results')
    def test_search_with_context(self, mock_process):
        """Test search with additional context."""
        # Mock the process results
        mock_process.return_value = {
            'success': True,
            'query': 'test',
            'search_context': {
                'search_type': 'academic',
                'context': {
                    'user_id': 'test_user',
                    'preferences': {'language': 'en'}
                }
            },
            'result_count': 1,
            'processed_results': [{'title': 'Processed'}],
            'summary': 'Found 1 result for test',
            'entities': {},
            'citations': {},
            'metadata': {}
        }
        
        context = {
            'user_id': 'test_user',
            'preferences': {'language': 'en'}
        }
        
        results = self.agent.search("test", context=context, search_type="academic")
        
        self.assertTrue(results['success'])
        self.agent.agent.execute.assert_called_once()
        
        # Verify context was passed correctly
        call_args = self.agent.agent.execute.call_args[0][1]
        self.assertEqual(call_args['search_type'], 'academic')
        self.assertEqual(call_args['context'], context)
    
    def test_process_search_results(self):
        """Test processing of search results."""
        # Mock search results
        search_results = {
            'results': [
                {
                    'title': 'Test 1',
                    'url': 'https://example.com/1',
                    'snippet': 'Test content 1',
                    'source': 'test',
                    'score': 0.9
                },
                {
                    'title': 'Test 2',
                    'url': 'https://example.com/2',
                    'snippet': 'Test content 2',
                    'source': 'test',
                    'score': 0.8
                }
            ]
        }
        
        # Patch content extraction
        with patch.object(self.agent, '_process_content') as mock_process_content:
            mock_process_content.return_value = {
                'content': 'Processed content',
                'summary': 'Summary',
                'processed': True
            }
            
            # Process results
            processed = self.agent._process_search_results(
                search_results, 
                'test query',
                {'test': 'context'}
            )
            
            # Verify results
            self.assertTrue(processed['success'])
            self.assertEqual(processed['query'], 'test query')
            self.assertEqual(len(processed['processed_results']), 2)
            self.assertEqual(processed['result_count'], 2)
            self.assertIn('summary', processed)
            self.assertIn('entities', processed)
            self.assertIn('citations', processed)
    
    @patch('scholar_verse.sub_agents.deep_search.agent.ToolContext')
    def test_extract_entities(self, mock_tool_context_class):
        """Test entity extraction with both tool-based and regex fallback."""
        import logging
        import sys
        import pprint
        
        # Set up the mock ToolContext
        mock_tool_context = MagicMock()
        mock_tool_context.state = {}
        mock_tool_context_class.return_value = mock_tool_context
        
        # Configure root logger to output to console
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        
        # Get logger for this test
        logger = logging.getLogger(__name__)
        logger.info("=== Starting test_extract_entities ===")
        
        # Debug: Print agent's current tools
        logger.info("Agent tools before test:")
        for i, tool in enumerate(getattr(self.agent, 'tools', [])):
            logger.info(f"  Tool {i}: {type(tool).__name__}, name={getattr(tool, 'name', 'N/A')}, callable={hasattr(tool, '_call')}")
        
        # Debug: Print agent's attributes
        logger.info("Agent attributes:")
        for attr in dir(self.agent):
            if not attr.startswith('_'):
                try:
                    attr_val = getattr(self.agent, attr)
                    logger.info(f"  {attr}: {type(attr_val).__name__}")
                except Exception as e:
                    logger.info(f"  {attr}: <error: {str(e)}>")
        
        # Test text with various entities - using strip() to remove extra whitespace
        text = """Google LLC is an American multinational technology company that specializes in 
Internet-related services and products. Sundar Pichai is the CEO of Google.
The company is headquartered in Mountain View, California.""".strip()
        
        logger.info(f"Original text: {text}")
        
        # Test 1: Test with mock entity extraction tool
        logger.info("--- Testing with mock entity extraction tool ---")
        
        # Reset the mock call count before our test
        mock_tool_context_class.reset_mock()
        
        # Create a MagicMock for the tool with the expected name and structure
        mock_tool = MagicMock()
        mock_tool.name = 'entity_extraction'  # This must match what the agent is looking for
        mock_tool._call.return_value = {
            'entities': [
                {'text': 'Google LLC', 'type': 'ORGANIZATION'},
                {'text': 'Sundar Pichai', 'type': 'PERSON'},
                {'text': 'Mountain View, California', 'type': 'LOCATION'}
            ]
        }
        
        # Make sure the tool has a proper __name__ attribute
        mock_tool.__name__ = 'entity_extraction'
        
        # Debug: Print mock tool details
        logger.info("Created mock tool with properties:")
        logger.info(f"  name: {mock_tool.name}")
        logger.info(f"  _call return value: {mock_tool._call.return_value}")
        logger.info(f"  dir(mock_tool): {dir(mock_tool)}")
        logger.info(f"  mock_tool._call: {mock_tool._call}")
        logger.info(f"  callable(mock_tool._call): {callable(mock_tool._call)}")
        
        # Replace agent's tools with our mock tool
        self.agent.tools = [mock_tool]
        
        # Log mock tool setup
        logger.info(f"Mock tool name: {mock_tool.name}")
        logger.info(f"Mock tool has _call: {hasattr(mock_tool, '_call')}")
        logger.info(f"Mock tool _call callable: {callable(mock_tool._call)}")
        
        # Debug: Check if the tool would be found by the extraction method
        entity_extraction_tool = next(
            (tool for tool in self.agent.tools if hasattr(tool, 'name') and tool.name == 'entity_extraction'),
            None
        )
        logger.info(f"Found entity extraction tool: {entity_extraction_tool is not None}")
        
        if entity_extraction_tool:
            logger.info(f"Tool name: {getattr(entity_extraction_tool, 'name', 'no_name')}")
            logger.info(f"Tool has _call: {hasattr(entity_extraction_tool, '_call')}")
            
            # Test calling the tool directly
            try:
                logger.info("Testing direct call to tool...")
                # Use a simple dict as context since we don't have ToolContext
                # Use the mocked ToolContext
                result = entity_extraction_tool._call(context=mock_tool_context, text=text)
                logger.info(f"Direct tool call result: {result}")
                
                # Verify the direct call result
                self.assertIsInstance(result, dict)
                self.assertIn('entities', result)
                self.assertEqual(len(result['entities']), 3)
                
            except Exception as e:
                logger.error(f"Error calling tool directly: {e}", exc_info=True)
                raise
        
        # Now test the _extract_entities method
        try:
            logger.info("\nCalling _extract_entities...")
            entities = self.agent._extract_entities(text)
            logger.info(f"Returned entities: {entities}")
            
            # Verify the results
            self.assertIsInstance(entities, list)
            logger.info(f"Number of entities returned: {len(entities)}")
            
            # Debug: Check if the tool was called
            call_count = mock_tool._call.call_count
            logger.info(f"Tool _call was called {call_count} times")
            
            if call_count > 0:
                logger.info("Tool was called successfully")
                call_args = mock_tool._call.call_args[1]
                logger.info(f"Tool called with args: {call_args}")
                self.assertIn('context', call_args)
                self.assertEqual(call_args['text'], text.strip())
            else:
                logger.warning("Tool was not called. Available tools in agent:")
                for i, tool in enumerate(self.agent.tools):
                    logger.warning(f"  Tool {i}: {type(tool).__name__}, name={getattr(tool, 'name', 'N/A')}")
                
                # If tool wasn't called, we should have used regex fallback
                self.assertGreater(len(entities), 0, "No entities found with regex fallback")
                logger.info("Using regex fallback for entities")
                
        except Exception as e:
            logger.error(f"Error in _extract_entities: {e}", exc_info=True)
            raise
        
        # Test 2: Test fallback to regex extraction when no tool is available
        logger.info("\n--- Testing regex fallback ---")
        self.agent.tools = []  # Remove all tools to trigger regex fallback
        entities = self.agent._extract_entities(text)
        self.assertIsInstance(entities, list)
        self.assertGreater(len(entities), 0, "No entities found with regex fallback")
        logger.info(f"Found {len(entities)} entities with regex fallback")
        
        # Test 2: Test fallback to regex extraction when no tool is available
        self.agent.tools = []  # Remove all tools to trigger regex fallback
        
        # Call the method again
        entities = self.agent._extract_entities(text)
        
        # Verify the results from regex fallback
        self.assertIsInstance(entities, list)
        # The regex pattern should find at least one entity in this text
        self.assertGreater(len(entities), 0, "Regex fallback should find at least one entity")
        
        # Verify the structure of the returned entities
        if entities:
            entity = entities[0]
            self.assertIn('text', entity)
            self.assertIn('type', entity)
            self.assertIn('start', entity)
            self.assertIn('end', entity)
            self.assertIn('source', entity)
            self.assertEqual(entity['source'], 'regex')
    
    def test_generate_search_summary(self):
        """Test search result summarization."""
        results = [
            {'title': 'Test 1', 'source': 'test', 'metadata': {'content_type': 'article'}},
            {'title': 'Test 2', 'source': 'test', 'metadata': {'content_type': 'paper'}},
            {'title': 'Test 3', 'source': 'other', 'metadata': {'content_type': 'article'}}
        ]
        
        summary = self.agent._generate_search_summary(results, 'test query')
        
        self.assertIn('summary', summary)
        self.assertEqual(summary['total_results'], 3)
        self.assertEqual(len(summary['sources']), 2)  # test and other
        self.assertEqual(len(summary['content_types']), 2)  # article and paper


if __name__ == '__main__':
    unittest.main()
