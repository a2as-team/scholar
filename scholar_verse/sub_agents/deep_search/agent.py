"""Deep Search Agent for ScholarVerse.

This module defines the Deep Search Agent that conducts real-time web searches
and content extraction to find additional scholarly information.
"""

from typing import Dict, Any, List, Optional, AsyncGenerator
import logging
import json
from datetime import datetime

from google.adk.agents.llm_agent import Agent as LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from scholar_verse.base_agent import BaseAgent
from scholar_verse.config import DEFAULT_MODEL
from scholar_verse.sub_agents.deep_search.prompt import DEEP_SEARCH_AGENT_INSTRUCTIONS
from scholar_verse.sub_agents.deep_search.tools import (
    WebSearchTool,
    WebScraperTool,
    CitationValidationTool,
    ContentExtractionTool,
    RAGManager
)
from scholar_verse.shared_libraries.logging_utils import logger


class DeepSearchAgent(LlmAgent):
    """Deep Search Agent for conducting real-time web searches and content extraction.
    
    This agent is responsible for:
    - Performing web searches for scholarly content
    - Extracting and processing content from web pages
    - Validating citations and references
    - Integrating with RAG for enhanced retrieval
    """
    
    def __init__(self):
        """Initialize the Deep Search Agent with tools and configuration."""
        super().__init__(
            name="deep_search_agent",
            description="Agent for conducting deep web searches and content extraction",
            model=DEFAULT_MODEL,
            instruction=DEEP_SEARCH_AGENT_INSTRUCTIONS,
        )
        
        # Initialize tools
        self._tools_initialized = False
        self._tools = []
        self.rag_manager = RAGManager()
    
    async def initialize_tools(self):
        """Initialize the agent's tools asynchronously."""
        if self._tools_initialized:
            return
            
        try:
            # Initialize core tools
            self._tools = [
                WebSearchTool(),
                WebScraperTool(),
                CitationValidationTool(),
                ContentExtractionTool()
            ]
            
            # Initialize RAG retrieval tool
            rag_tool = await self.rag_manager.create_retrieval_tool_async()
            if rag_tool:
                self._tools.append(rag_tool)
            
            self._tools_initialized = True
            logger.info("DeepSearchAgent tools initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize DeepSearchAgent tools: {str(e)}")
            raise
    
    @property
    def tools(self) -> List[BaseTool]:
        """Get the list of tools available to this agent."""
        return self._tools
    
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Core implementation of the deep search agent's async execution.
        
        Args:
            ctx: The invocation context containing the search query and parameters.
            
        Yields:
            Event: Search result events.
        """
        try:
            # Initialize tools if not already done
            await self.initialize_tools()
            
            # Get search query from context
            query = ctx.get("query")
            if not query:
                yield Event(content=Content.from_text("No search query provided"))
                return
                
            logger.info(f"Processing deep search query: {query}")
            
            # Perform the search and process results
            async for result in self._process_search(query, ctx):
                yield result
                
        except Exception as e:
            logger.error(f"Error in DeepSearchAgent: {str(e)}", exc_info=True)
            yield Event(content=Content.from_text(
                f"An error occurred during the search: {str(e)}"
            ))
    
    async def _process_search(
        self, query: str, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Process a search query and yield results.
        
        Args:
            query: The search query string.
            ctx: The invocation context.
            
        Yields:
            Event: Search result events.
        """
        # This is a simplified example - implement actual search logic here
        yield Event(content=Content.from_text(
            f"Performing deep search for: {query}"
        ))
        
        # Example of using a tool
        try:
            web_search = next((t for t in self._tools if isinstance(t, WebSearchTool)), None)
            if web_search:
                search_results = await web_search._call({"query": query}, ToolContext())
                yield Event(content=Content.from_text(
                    f"Found {len(search_results.get('results', []))} results"
                ))
        except Exception as e:
            logger.warning(f"Web search failed: {str(e)}")
    
    @staticmethod
    def make_json_serializable(obj: Any) -> Any:
        """Convert non-serializable objects to a serializable format.
        
        Args:
            obj: The object to make JSON serializable.
            
        Returns:
            A JSON-serializable version of the object.
        """
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {k: DeepSearchAgent.make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, set)):
            return [DeepSearchAgent.make_json_serializable(x) for x in obj]
        elif hasattr(obj, '__dict__'):
            return DeepSearchAgent.make_json_serializable(obj.__dict__)
        elif hasattr(obj, 'isoformat'):  # Handle datetime objects
            return obj.isoformat()
        return str(obj)


class DeepSearchAgent(BaseAgent):
    """Deep Search Agent for ScholarVerse."""
    
    def __init__(self):
        """Initialize the Deep Search Agent."""
        # Initialize BaseAgent
        super().__init__(
            name="deep_search_agent",
            description="Deep Search Agent for conducting real-time web searches and content extraction"
        )
        
        # Initialize tools list
        self.tools = []
        
        # Initialize RAG Manager
        self.rag_manager = RAGManager()
        
        # Track initialization status and errors
        self._initialized = False
        self._initialization_error = None
        
    async def _fallback_async_generator(self, error: str) -> AsyncGenerator[Event, None]:
        """Fallback async generator for when initialization fails.
        
        Args:
            error: The error message from initialization.
            
        Yields:
            Error events with the initialization error.
        """
        logger.error(f"Using fallback generator due to initialization error: {error}")
        yield Event(
            content=Content.from_text(
                json.dumps({
                    'success': False,
                    'error': f'Agent initialization failed: {error}',
                    'timestamp': datetime.now().isoformat()
                }, ensure_ascii=False, default=self.make_json_serializable)
            ),
            event_type='initialization_error',
        )
        
    async def initialize(self):
        """Initialize the agent asynchronously."""
        if self._initialized:
            return
            
        try:
            # Initialize core tools
            tools_to_initialize = [
                (WebSearchTool, "WebSearchTool"),
                (WebScraperTool, "WebScraperTool"),
                (CitationValidationTool, "CitationValidationTool"),
                (ContentExtractionTool, "ContentExtractionTool")
            ]
            
            # Initialize each tool and add to tools list if valid
            for tool_class, tool_name in tools_to_initialize:
                try:
                    tool = tool_class()
                    if hasattr(tool, '_call') and callable(tool._call):
                        self.tools.append(tool)
                        logger.info(f"Initialized {tool_name}")
                    else:
                        logger.warning(f"{tool_name} is not a valid tool - missing _call method")
                except Exception as e:
                    logger.error(f"Failed to initialize {tool_name}: {str(e)}")
            
            # Initialize RAG retrieval tool
            try:
                rag_retrieval = await self.rag_manager.create_retrieval_tool()
                if rag_retrieval and hasattr(rag_retrieval, '_call') and callable(rag_retrieval._call):
                    self.tools.append(rag_retrieval)
                    logger.info("Initialized RAGRetrievalTool")
                else:
                    logger.warning("RAG retrieval tool is not properly initialized - skipping")
            except Exception as e:
                logger.error(f"Failed to initialize RAG retrieval tool: {str(e)}")
            
            logger.info(f"Successfully initialized {len(self.tools)} tools for Deep Search Agent")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize Deep Search Agent: {str(e)}")
            raise
            
    async def search(self, query: str, context: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Perform an asynchronous search with RAG-enhanced query processing.
        
        Args:
            query: The search query string.
            context: Additional context for the search.
            
        Yields:
            Dict containing search results and metadata as they become available.
        """
        if not self._initialized:
            await self.initialize()
            
        if not query or not isinstance(query, str):
            error_msg = "Invalid query: query must be a non-empty string"
            logger.error(error_msg)
            yield {
                'success': False,
                'error': error_msg,
                'query': str(query) if query else '',
                'timestamp': datetime.now().isoformat()
            }
            return
            
        # Get the web search tool
        web_search_tool = next((t for t in self.tools if isinstance(t, WebSearchTool)), None)
        if not web_search_tool:
            error_msg = "Web search tool not available"
            logger.error(error_msg)
            yield {
                'success': False,
                'error': error_msg,
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
            return
            
        # Get the RAG retrieval tool if available
        rag_tool = next((t for t in self.tools if hasattr(t, 'name') and 'rag' in t.name.lower()), None)
        
        try:
            # Enhance query using RAG if available
            enhanced_query = query
            if rag_tool and context and context.get('use_rag', True):
                try:
                    rag_context = await rag_tool._call({
                        'query': query,
                        'context': context
                    })
                    if rag_context and 'enhanced_query' in rag_context:
                        enhanced_query = rag_context['enhanced_query']
                        logger.info(f"Enhanced query using RAG: {enhanced_query}")
                except Exception as e:
                    logger.warning(f"Error enhancing query with RAG: {str(e)}")
            
            # Execute the web search
            search_results = await web_search_tool._call({
                'query': enhanced_query,
                'context': context or {}
            })
            
            if not search_results or 'results' not in search_results or not search_results['results']:
                yield {
                    'success': True,
                    'query': query,
                    'enhanced_query': enhanced_query,
                    'results': [],
                    'status': 'no_results',
                    'timestamp': datetime.now().isoformat()
                }
                return
                
            # Process and yield results as they become available
            processed_count = 0
            total_results = len(search_results['results'])
            
            for result in search_results['results']:
                try:
                    # Process the result
                    processed_result = await self._process_search_result(result, context)
                    processed_count += 1
                    
                    # Yield the processed result
                    yield {
                        'success': True,
                        'query': query,
                        'enhanced_query': enhanced_query,
                        'result': processed_result,
                        'status': 'in_progress',
                        'processed': processed_count,
                        'total': total_results,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing search result: {str(e)}")
                    yield {
                        'success': False,
                        'query': query,
                        'error': f"Error processing result: {str(e)}",
                        'status': 'error',
                        'timestamp': datetime.now().isoformat()
                    }
            
            # Signal search completion
            yield {
                'success': True,
                'query': query,
                'enhanced_query': enhanced_query,
                'status': 'completed',
                'processed': processed_count,
                'total': total_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            yield {
                'success': False,
                'query': query,
                'error': str(e),
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }
            
#         logger.info(f"Deep Search initiated for query: {query}, type: {search_type}")
        
#         try:
#             # Prepare search context
#             search_context = {
#                 "query": query,
#                 "search_type": search_type,
#                 "context": context or {},
#                 "timestamp": datetime.now().isoformat()
#             }
            
#             logger.debug(f"Executing search with context: {json.dumps(search_context, indent=2)}")
            
#             # Use web search to find relevant sources
#             search_results = self.agent.execute(
#                 "Perform a comprehensive search for the following query and find the most relevant scholarly sources:",
#                 search_context
#             )
            
#             if not search_results or not isinstance(search_results, dict):
#                 error_msg = "Invalid search results format"
#                 logger.error(f"{error_msg}: {search_results}")
#                 raise ValueError(error_msg)
            
#             # Process results using the information integration workflow
#             processed_results = self._process_search_results(search_results, query, search_context)
            
#             response = {
#                 'success': True,
#                 'query': query,
#                 'results': processed_results,
#                 'search_type': search_type,
#                 'timestamp': datetime.now().isoformat(),
#                 'tools_used': [tool.name for tool in self.tools if hasattr(tool, 'name')]
#             }
            
#             logger.info(f"Successfully completed search for query: {query}")
#             # Create a safe version of the response for logging
#             def make_json_serializable(obj):
#                 if isinstance(obj, (str, int, float, bool, type(None))):
#                     return obj
#                 elif isinstance(obj, dict):
#                     return {k: make_json_serializable(v) for k, v in obj.items()}
#                 elif isinstance(obj, (list, tuple)):
#                     return [make_json_serializable(item) for item in obj]
#                 else:
#                     # For any other type, convert to string
#                     return str(obj)
            
#             safe_response = {
#                 'success': response.get('success'),
#                 'query': response.get('query'),
#                 'search_type': response.get('search_type'),
#                 'timestamp': response.get('timestamp'),
#                 'tools_used': response.get('tools_used'),
#                 'results_count': len(response.get('results', [])),
#             }
            
#             # Convert all values to JSON-serializable types
#             safe_response = make_json_serializable(safe_response)
#             logger.debug(f"Search completed. Response summary: {json.dumps(safe_response, indent=2)}")
            
#             return response
            
#         except Exception as e:
#             error_msg = f"Error during deep search for '{query}': {str(e)}"
#             logger.error(error_msg, exc_info=True)
            
#             return {
#                 'success': False,
#                 'error': str(e),
#                 'query': query,
#                 'search_type': search_type,
#                 'timestamp': datetime.now().isoformat(),
#                 'error_type': e.__class__.__name__
#             }
    
#     def _process_search_results(self, search_results: Dict[str, Any], 
#                               query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
#         """Process search results through the information integration workflow.
        
#         This method takes raw search results, extracts relevant information, and processes them
#         using available tools to provide structured and enriched results.
        
#         Args:
#             search_results: The raw search results to process.
#             query: The original search query.
#             context: Additional context for the search, including any relevant metadata.
            
#         Returns:
#             A dictionary containing:
#             - processed_results: List of processed search results
#             - summary: A summary of the search results
#             - entities: Extracted entities from the results
#             - citations: Any citations found in the results
#             - metadata: Additional processing metadata
#         """
#         if not search_results or not isinstance(search_results, dict):
#             logger.warning("No valid search results to process")
#             return {
#                 'success': False,
#                 'error': 'No valid search results to process',
#                 'query': query,
#                 'processed_results': []
#             }
            
#         processed_results = []
#         all_entities = []
#         all_citations = []
        
#         logger.info(f"Processing {len(search_results.get('results', []))} search results")
        
#         # Extract and process each result
#         for idx, result in enumerate(search_results.get('results', []), 1):
#             try:
#                 logger.debug(f"Processing result {idx}/{len(search_results.get('results', []))}")
                
#                 # Extract content if available
#                 content = result.get('content', '')
#                 if not content and 'snippet' in result:
#                     content = result['snippet']
                
#                 # Skip results without content
#                 if not content:
#                     logger.debug(f"Skipping result {idx} - no content")
#                     continue
                
#                 # Process content using content extraction tool
#                 processed_content = self._process_content(content, result.get('url', ''), context)
                
#                 # Extract entities if content extraction was successful
#                 entities = []
#                 if processed_content and processed_content.get('success', False):
#                     try:
#                         entities = self._extract_entities(processed_content.get('content', ''))
#                         all_entities.extend(entities)
#                     except Exception as e:
#                         logger.warning(f"Failed to extract entities from result {idx}: {str(e)}")
                
#                 # Extract citations if content extraction was successful
#                 citations = []
#                 if processed_content and processed_content.get('success', False):
#                     try:
#                         citations = self._extract_citations(processed_content.get('content', ''))
#                         all_citations.extend(citations)
#                     except Exception as e:
#                         logger.warning(f"Failed to extract citations from result {idx}: {str(e)}")
                
#                 # Create result entry
#                 result_entry = {
#                     'title': result.get('title', 'No title'),
#                     'url': result.get('url', ''),
#                     'source': result.get('source', 'unknown'),
#                     'content_summary': processed_content.get('summary', '') if isinstance(processed_content, dict) else str(processed_content)[:500],
#                     'entities': entities,
#                     'citations': citations,
#                     'relevance_score': result.get('score', 0.0),
#                     'metadata': {
#                         'language': result.get('language', 'en'),
#                         'last_updated': result.get('last_updated', ''),
#                         'content_type': result.get('content_type', 'webpage'),
#                         'processed_at': datetime.now().isoformat()
#                     }
#                 }
                
#                 processed_results.append(result_entry)
#                 logger.debug(f"Successfully processed result {idx}")
                
#             except Exception as e:
#                 logger.error(f"Error processing search result {idx}: {str(e)}", exc_info=True)
#                 continue
        
#         # Generate a summary of all results
#         summary = self._generate_search_summary(processed_results, query)
        
#         # Process all entities and citations
#         processed_entities = self._process_entities(all_entities)
#         processed_citations = self._process_citations(all_citations)
        
#         return {
#             'success': True,
#             'query': query,
#             'search_context': context,
#             'result_count': len(processed_results),
#             'processed_results': processed_results,
#             'summary': summary,
#             'entities': processed_entities,
#             'citations': processed_citations,
#             'metadata': {
#                 'processing_time': f"{len(processed_results)} results processed",
#                 'timestamp': datetime.now().isoformat(),
#                 'tools_used': [tool.name for tool in self.tools if hasattr(tool, 'name')]
#             }
#         }
    
#     def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
#         """Extract named entities from the given text.
        
#         Args:
#             text: The text to extract entities from.
            
#         Returns:
#             A list of dictionaries containing entity information.
#         """
#         if not text:
#             print("No text provided for entity extraction")
#             return []
            
#         try:
#             print(f"\n=== Starting entity extraction ===")
#             print(f"Available tools: {[getattr(tool, 'name', 'no_name') for tool in self.tools]}")
            
#             # Try to find an entity extraction tool
#             entity_extraction_tool = None
#             for i, tool in enumerate(self.tools):
#                 has_name = hasattr(tool, 'name')
#                 name = getattr(tool, 'name', 'no_name')
#                 print(f"Tool {i}: name='{name}', has_name={has_name}, name=='entity_extraction': {name == 'entity_extraction'}")
#                 if has_name and name == 'entity_extraction':
#                     entity_extraction_tool = tool
#                     break
            
#             print(f"Found entity extraction tool: {entity_extraction_tool is not None}")
            
#             if entity_extraction_tool:
#                 print("Calling entity extraction tool...")
#                 try:
#                     result = entity_extraction_tool._call(
#                         context=ToolContext(),
#                         text=text
#                     )
#                     print(f"Tool returned: {result}")
#                     if isinstance(result, list):
#                         return result
#                     elif isinstance(result, dict) and 'entities' in result:
#                         return result['entities']
#                 except Exception as e:
#                     print(f"Error calling entity extraction tool: {e}")
            
#             print("Falling back to regex-based entity extraction")
#             # Fallback: simple regex-based entity extraction
#             entities = []
#             # Extract simple patterns (this is a very basic implementation)
#             patterns = {
#                 'PERSON': r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
#                 'ORGANIZATION': r'([A-Z][a-zA-Z0-9&]+(?:\s+[A-Z][a-zA-Z0-9&]+)+)',
#                 'LOCATION': r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:City|State|Country|Province|Region))',
#             }
            
#             for entity_type, pattern in patterns.items():
#                 print(f"\nMatching pattern for {entity_type}: {pattern}")
#                 matches = list(re.finditer(pattern, text))
#                 print(f"Found {len(matches)} matches for {entity_type}")
#                 for match in matches:
#                     entity_text = match.group(1)
#                     if len(entity_text.split()) > 1:  # Only include multi-word entities
#                         entities.append({
#                             'text': entity_text,
#                             'type': entity_type,
#                             'start': match.start(),
#                             'end': match.end(),
#                             'source': 'regex'
#                         })
#                         print(f"  - Added entity: {entity_text} ({entity_type}) at position {match.start()}-{match.end()}")
            
#             print(f"Total entities found: {len(entities)}")
#             return entities
            
#         except Exception as e:
#             logger.error(f"Error extracting entities: {str(e)}", exc_info=True)
#             print(f"Exception in _extract_entities: {e}")
#             return []
    
#     def _extract_citations(self, text: str) -> List[Dict[str, Any]]:
#         """Extract citations from the given text.
        
#         Args:
#             text: The text to extract citations from.
            
#         Returns:
#             A list of dictionaries containing citation information.
#         """
#         if not text:
#             return []
            
#         try:
#             # Try to find a citation extraction tool
#             citation_tool = next(
#                 (tool for tool in self.tools if hasattr(tool, 'name') and tool.name == 'citation_validation'),
#                 None
#             )
            
#             if citation_tool:
#                 result = citation_tool._call(
#                     context=ToolContext(),
#                     text=text,
#                     extract_only=True
#                 )
#                 if isinstance(result, list):
#                     return result
#                 elif isinstance(result, dict) and 'citations' in result:
#                     return result['citations']
            
#             # Fallback: simple citation pattern matching
#             citations = []
#             # Match common citation patterns (very basic implementation)
#             patterns = [
#                 # Author (Year) pattern
#                 r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+\.?)+(?:\s+et\s+al\.)?\s*\(\d{4}\))',
#                 # [1], [2-5], [6,8,10] patterns
#                 r'(\[\d+(?:[,-]?\s*\d+)*\])',
#             ]
            
#             for pattern in patterns:
#                 matches = re.finditer(pattern, text)
#                 for match in matches:
#                     citations.append({
#                         'text': match.group(1),
#                         'type': 'inline_citation',
#                         'start': match.start(),
#                         'end': match.end(),
#                         'source': 'regex'
#                     })
            
#             return citations
            
#         except Exception as e:
#             logger.error(f"Error extracting citations: {str(e)}", exc_info=True)
#             return []
    
#     def _process_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """Process and aggregate extracted entities.
        
#         Args:
#             entities: List of entity dictionaries.
            
#         Returns:
#             A dictionary containing processed entity information.
#         """
#         if not entities:
#             return {}
            
#         try:
#             # Count entity frequencies
#             entity_counter = Counter()
#             entity_types = defaultdict(list)
            
#             for entity in entities:
#                 if not isinstance(entity, dict):
#                     continue
                    
#                 entity_text = entity.get('text', '').strip()
#                 entity_type = entity.get('type', 'UNKNOWN').upper()
                
#                 if entity_text:
#                     entity_counter[(entity_text, entity_type)] += 1
#                     entity_types[entity_type].append(entity_text)
            
#             # Get most common entities
#             most_common = entity_counter.most_common(10)
            
#             return {
#                 'total_entities': len(entities),
#                 'unique_entities': len(entity_counter),
#                 'entity_types': dict(entity_types),
#                 'top_entities': [{'text': text, 'type': type_, 'count': count} 
#                                for (text, type_), count in most_common]
#             }
            
#         except Exception as e:
#             logger.error(f"Error processing entities: {str(e)}", exc_info=True)
#             return {}
    
#     def _process_citations(self, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """Process and aggregate extracted citations.
        
#         Args:
#             citations: List of citation dictionaries.
            
#         Returns:
#             A dictionary containing processed citation information.
#         """
#         if not citations:
#             return {}
            
#         try:
#             citation_types = defaultdict(int)
#             for citation in citations:
#                 if isinstance(citation, dict):
#                     citation_type = citation.get('type', 'unknown')
#                     citation_types[citation_type] += 1
            
#             return {
#                 'total_citations': len(citations),
#                 'citation_types': dict(citation_types)
#             }
            
#         except Exception as e:
#             logger.error(f"Error processing citations: {str(e)}", exc_info=True)
#             return {}
    
#     def _generate_search_summary(self, results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
#         """Generate a summary of search results.
        
#         Args:
#             results: List of processed search results.
#             query: The original search query.
            
#         Returns:
#             A dictionary containing summary information.
#         """
#         if not results:
#             return {
#                 'summary': 'No results found for the query.',
#                 'sources': [],
#                 'total_results': 0
#             }
            
#         try:
#             # Count sources
#             sources = Counter()
#             content_types = Counter()
            
#             for result in results:
#                 if not isinstance(result, dict):
#                     continue
                    
#                 source = result.get('source', 'unknown')
#                 content_type = result.get('metadata', {}).get('content_type', 'unknown')
                
#                 sources[source] += 1
#                 content_types[content_type] += 1
            
#             # Generate summary text
#             total_results = len(results)
#             top_sources = sources.most_common(3)
            
#             summary_parts = [
#                 f"Found {total_results} results for '{query}'.",
#                 f"Top sources: {', '.join([f'{source} ({count})' for source, count in top_sources])}.",
#                 f"Content types: {', '.join([f'{ctype} ({count})' for ctype, count in content_types.most_common()])}."
#             ]
            
#             return {
#                 'summary': ' '.join(summary_parts),
#                 'sources': [{'source': source, 'count': count} for source, count in sources.most_common()],
#                 'content_types': [{'type': ctype, 'count': count} for ctype, count in content_types.most_common()],
#                 'total_results': total_results,
#                 'timestamp': datetime.now().isoformat()
#             }
            
#         except Exception as e:
#             logger.error(f"Error generating search summary: {str(e)}", exc_info=True)
#             return {
#                 'summary': f'Found {len(results)} results for "{query}"',
#                 'error': str(e),
#                 'timestamp': datetime.now().isoformat()
#             }
    
#     def _process_content(self, content: str, source_url: str = '', context: Dict[str, Any] = None) -> Dict[str, Any]:
#         """Process content using available tools.
        
#         This method processes the given content using the available content extraction tools,
#         handling errors and providing fallback mechanisms when tools are not available.
        
#         Args:
#             content: The content to process (plain text or HTML).
#             source_url: The source URL where the content was retrieved from.
#             context: Additional context for processing, such as content type or language hints.
            
#         Returns:
#             A dictionary containing:
#             - content: The processed content (may be truncated if processing failed)
#             - summary: A summary of the content if extraction was successful
#             - entities: Any named entities found in the content
#             - source_url: The source URL
#             - processed: Boolean indicating if processing was successful
#             - error: Error message if processing failed
#         """
#         if not content:
#             return {
#                 'success': False,
#                 'error': 'No content provided',
#                 'source_url': source_url,
#                 'processed': False
#             }
            
#         logger.debug(f"Processing content from {source_url or 'unknown source'}, length: {len(content)} chars")
        
#         try:
#             # Find content extraction tool
#             content_extraction_tool = next(
#                 (tool for tool in self.tools if hasattr(tool, 'name') and tool.name == 'content_extraction'),
#                 None
#             )
            
#             if content_extraction_tool:
#                 try:
#                     logger.debug("Using content extraction tool")
#                     result = content_extraction_tool._call(
#                         context=ToolContext(),
#                         content=content,
#                         source_url=source_url,
#                         context_info=context or {}
#                     )
                    
#                     if result and isinstance(result, dict):
#                         result['processed'] = True
#                         result['source_url'] = source_url
#                         return result
                    
#                     logger.warning("Content extraction returned invalid result, using fallback")
                    
#                 except Exception as e:
#                     logger.error(f"Error in content extraction: {str(e)}", exc_info=True)
            
#             # Fallback to simple processing
#             logger.debug("Using fallback content processing")
#             return {
#                 'content': content[:5000],  # Truncate to avoid huge responses
#                 'summary': content[:500] + ('...' if len(content) > 500 else ''),
#                 'source_url': source_url,
#                 'processed': True,
#                 'fallback': True,
#                 'metadata': {
#                     'processing_method': 'fallback',
#                     'content_length': len(content),
#                     'timestamp': datetime.now().isoformat()
#                 }
#             }
            
#         except Exception as e:
#             error_msg = f"Unexpected error processing content: {str(e)}"
#             logger.error(error_msg, exc_info=True)
#             return {
#                 'content': content[:1000] if content else '',
#                 'source_url': source_url,
#                 'processed': False,
#                 'error': error_msg,
#                 'error_type': e.__class__.__name__,
#                 'metadata': {
#                     'error': True,
#                     'error_type': e.__class__.__name__,
#                     'timestamp': datetime.now().isoformat()
#                 }
#             }
    
#     def validate_citations(self, citations: List[str], 
#                           source_urls: List[str] = None) -> Dict[str, Any]:
#         """Validate and track a list of citations.
        
#         Args:
#             citations: List of citations to validate.
#             source_urls: List of source URLs where the citations were found.
            
#         Returns:
#             Validation results for the citations.
#         """
#         logger.info(f"Validating {len(citations)} citations")
        
#         try:
#             # Prepare source URLs if not provided
#             if source_urls is None:
#                 source_urls = [None] * len(citations)
#             elif len(source_urls) < len(citations):
#                 # Pad with None if not enough URLs
#                 source_urls.extend([None] * (len(citations) - len(source_urls)))
            
#             validation_results = []
            
#             # Validate each citation
#             for i, (citation, source_url) in enumerate(zip(citations, source_urls)):
#                 result = self.agent.execute(
#                     "Validate this citation and assess its reliability:",
#                     {"citation": citation, "source_url": source_url}
#                 )
#                 validation_results.append(result)
            
#             return {
#                 'success': True,
#                 'validation_results': validation_results,
#                 'total_citations': len(citations),
#                 'timestamp': datetime.now().isoformat()
#             }
        
#         except Exception as e:
#             logger.error(f"Error validating citations: {str(e)}")
#             return {
#                 'success': False,
#                 'error': str(e),
#                 'total_citations': len(citations)
#             }


# # Create an instance of the Deep Search Agent
# deep_search_agent_instance = DeepSearchAgent()

# # Create an AgentTool from the Deep Search Agent
# deep_search_agent_tool = AgentTool(agent=deep_search_agent_instance.agent)

"""Deep Search Agent for ScholarVerse.

This module defines the Deep Search Agent that conducts real-time web searches and content extraction to find additional scholarly information.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import re
import json
from datetime import datetime
from collections import defaultdict, Counter
from urllib.parse import urlparse
from unittest.mock import MagicMock
import asyncio
from google.adk.agents.invocation_context import InvocationContext

from google.adk import Agent
from google.adk.tools.agent_tool import AgentTool, ToolContext

from scholar_verse.config import DEFAULT_MODEL
from scholar_verse.sub_agents.deep_search.prompt import DEEP_SEARCH_AGENT_INSTRUCTIONS
from scholar_verse.sub_agents.deep_search.tools import (
    WebSearchTool,
    WebScraperTool,
    CitationValidationTool,
    ContentExtractionTool,
    RAGManager
)
from scholar_verse.shared_libraries.logging_utils import logger


class DeepSearchAgent:
    """Deep Search Agent for ScholarVerse."""
    
    def __init__(self):
        """Initialize the Deep Search Agent."""
        # Initialize tools list
        self.tools = []
        
        # Initialize RAG Manager
        self.rag_manager = RAGManager()
        
        try:
            # Initialize core tools
            tools_to_initialize = [
                (WebSearchTool, "WebSearchTool"),
                (WebScraperTool, "WebScraperTool"),
                (CitationValidationTool, "CitationValidationTool"),
                (ContentExtractionTool, "ContentExtractionTool")
            ]
            
            # Initialize each tool and add to tools list if valid
            for tool_class, tool_name in tools_to_initialize:
                try:
                    tool = tool_class()
                    if hasattr(tool, '_call') and callable(tool._call):
                        self.tools.append(tool)
                        logger.info(f"Initialized {tool_name}")
                    else:
                        logger.warning(f"{tool_name} is not a valid tool - missing _call method")
                except Exception as e:
                    logger.error(f"Failed to initialize {tool_name}: {str(e)}")
            
            # Initialize RAG retrieval tool
            try:
                rag_retrieval = self.rag_manager.create_retrieval_tool()
                if rag_retrieval and hasattr(rag_retrieval, '_call') and callable(rag_retrieval._call):
                    self.tools.append(rag_retrieval)
                    logger.info("Initialized RAGRetrievalTool")
                else:
                    logger.warning("RAG retrieval tool is not properly initialized - skipping")
            except Exception as e:
                logger.error(f"Failed to initialize RAG retrieval tool: {str(e)}")
            
            logger.info(f"Successfully initialized {len(self.tools)} tools for Deep Search Agent")
            
            # Create Agent with available tools
            self.agent = Agent(
                model=DEFAULT_MODEL,
                name="deep_search_agent",
                instruction=DEEP_SEARCH_AGENT_INSTRUCTIONS,
                tools=self.tools,
            )
            logger.info("Deep Search Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Deep Search Agent: {str(e)}")
            # Create a fallback agent with minimal functionality
            self.agent = MagicMock()
            self.agent.run_async = lambda *args, **kwargs: self._fallback_async_generator(str(e))
    
    async def _fallback_async_generator(self, error: str):
        """Fallback async generator for failed agent initialization."""
        yield {
            'success': False,
            'error': f'Agent initialization failed: {error}',
            'query': ''
        }
    
    async def __call__(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make the agent callable. This allows the agent to be used directly as a function.
        
        Args:
            query: The search query string or a dictionary containing the query.
            context: Additional context for the search.
            
        Returns:
            Dict containing search results and metadata.
        """
        return await self.search(query, context)
    
    async def search(self, query: str, context: Dict[str, Any] = None, 
                     search_type: str = "academic") -> Dict[str, Any]:
        """Perform a deep search for a query.
        
        Args:
            query: The search query string or a dictionary containing the query.
            context: Additional context for the search, including any relevant metadata.
            search_type: Type of search to perform (academic, general, etc.).
            
        Returns:
            A dictionary containing:
            - success: Boolean indicating if the search was successful
            - query: The original search query
            - results: Processed search results (if successful)
            - error: Error message (if failed)
            - search_type: The type of search performed
            - timestamp: ISO format timestamp of when the search was performed
        """
        if not query or not isinstance(query, str):
            error_msg = "Invalid query: query must be a non-empty string"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'query': str(query) if query else '',
                'search_type': search_type,
                'timestamp': datetime.now().isoformat()
            }
            
        logger.info(f"Deep Search initiated for query: {query}, type: {search_type}")
        
        try:
            # Prepare search context
            search_context = {
                "query": query,
                "search_type": search_type,
                "context": context or {},
                "timestamp": datetime.now().isoformat()
            }
            
            logger.debug(f"Executing search with context: {json.dumps(search_context, indent=2)}")
            
            # Use web search to find relevant sources
            search_results = []
            async for event in self.agent.run_async(InvocationContext()):
                if event.content and event.content.parts:
                    result_text = ''.join(part.text for part in event.content.parts if part.text)
                    try:
                        result = json.loads(result_text) if result_text else {}
                    except json.JSONDecodeError:
                        result = {'text': result_text}
                    search_results.append(result)
            
            if not search_results:
                error_msg = "No valid search results returned"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Process results using the information integration workflow
            processed_results = self._process_search_results(search_results, query, search_context)
            
            response = {
                'success': True,
                'query': query,
                'results': processed_results,
                'search_type': search_type,
                'timestamp': datetime.now().isoformat(),
                'tools_used': [tool.name for tool in self.tools if hasattr(tool, 'name')]
            }
            
            logger.info(f"Successfully completed search for query: {query}")
            # Create a safe version of the response for logging
            def make_json_serializable(obj):
                if isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                elif isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [make_json_serializable(item) for item in obj]
                else:
                    return str(obj)
            
            safe_response = {
                'success': response.get('success'),
                'query': response.get('query'),
                'search_type': response.get('search_type'),
                'timestamp': response.get('timestamp'),
                'tools_used': response.get('tools_used'),
                'results_count': len(response.get('results', [])),
            }
            
            safe_response = make_json_serializable(safe_response)
            logger.debug(f"Search completed. Response summary: {json.dumps(safe_response, indent=2)}")
            
            return response
            
        except Exception as e:
            error_msg = f"Error during deep search for '{query}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'search_type': search_type,
                'timestamp': datetime.now().isoformat(),
                'error_type': e.__class__.__name__
            }
    
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Core async implementation of the deep search agent.
        
        Args:
            ctx: The invocation context containing the search query and parameters
            
        Yields:
            Event: Search result events
        """
        try:
            # Ensure agent is initialized
            if not self._initialized:
                await self.initialize()
                
            # Extract query and context from the invocation context
            query = ctx.input.text
            context = ctx.state.get('context', {})
            
            # Check if this is a citation validation request
            if 'citation' in context:
                # Handle citation validation
                citation = context['citation']
                source_url = context.get('source_url')
                
                # Process the citation
                try:
                    # Use the citation validation tool if available
                    citation_tool = next((t for t in self.tools if isinstance(t, CitationValidationTool)), None)
                    if citation_tool:
                        result = await citation_tool._call({
                            'citation': citation,
                            'source_url': source_url
                        })
                        
                        # Yield the validation result
                        yield Event(
                            content=Content.from_text(
                                json.dumps({
                                    'success': True,
                                    'citation': citation,
                                    'source_url': source_url,
                                    'validation_result': result,
                                    'timestamp': datetime.now().isoformat()
                                }, ensure_ascii=False, default=self.make_json_serializable)
                            ),
                            event_type='citation_validation'
                        )
                    else:
                        raise ValueError("Citation validation tool not available")
                        
                except Exception as e:
                    logger.error(f"Error validating citation: {str(e)}")
                    yield Event(
                        content=Content.from_text(
                            json.dumps({
                                'success': False,
                                'citation': citation,
                                'source_url': source_url,
                                'error': str(e),
                                'timestamp': datetime.now().isoformat()
                            }, ensure_ascii=False, default=self.make_json_serializable)
                        ),
                        event_type='citation_validation_error'
                    )
                
                return
                
            # Handle regular search queries
            try:
                # Run the search asynchronously
                async for result in self.search(query, context=context):
                    # Yield each result as it's available
                    yield Event(
                        content=Content.from_text(
                            json.dumps(result, ensure_ascii=False, default=self.make_json_serializable)
                        ),
                        event_type='search_result',
                    )
                
                # Signal search completion
                yield Event(
                    content=Content.from_text(
                        json.dumps({
                            'success': True,
                            'query': query,
                            'status': 'search_completed',
                            'timestamp': datetime.now().isoformat()
                        }, ensure_ascii=False, default=self.make_json_serializable)
                    ),
                    event_type='search_complete'
                )
                
            except Exception as e:
                logger.error(f"Error in deep search: {str(e)}")
                yield Event(
                    content=Content.from_text(
                        json.dumps({
                            'success': False,
                            'error': str(e),
                            'query': query,
                            'timestamp': datetime.now().isoformat()
                        }, ensure_ascii=False, default=self.make_json_serializable)
                    ),
                    event_type='search_error',
                )
                
        except Exception as e:
            logger.error(f"Unexpected error in deep search agent: {str(e)}")
            yield Event(
                content=Content.from_text(
                    json.dumps({
                        'success': False,
                        'error': f"Unexpected error: {str(e)}",
                        'timestamp': datetime.now().isoformat()
                    }, ensure_ascii=False, default=self.make_json_serializable)
                ),
                event_type='error',
            )
    async def validate_citations(self, citations: List[str], 
                               source_urls: List[str] = None) -> Dict[str, Any]:
        """Validate and track a list of citations.
        

        Args:
            citations: List of citations to validate.
            source_urls: List of source URLs where the citations were found.
            
        Returns:
            Validation results for the citations.
        """
        if not self._initialized:
            await self.initialize()
            
        logger.info(f"Validating {len(citations)} citations")
        
        try:
            # Prepare source URLs if not provided
            if source_urls is None:
                source_urls = [None] * len(citations)
            elif len(source_urls) < len(citations):
                source_urls.extend([None] * (len(citations) - len(source_urls)))
            
            validation_results = []
            
            # Find the citation validation tool
            citation_tool = next((t for t in self.tools if isinstance(t, CitationValidationTool)), None)
            if not citation_tool:
                raise ValueError("Citation validation tool not available")
            
            # Process citations in batches to avoid overwhelming the system
            batch_size = 5  # Process 5 citations at a time
            for i in range(0, len(citations), batch_size):
                batch = citations[i:i + batch_size]
                batch_urls = source_urls[i:i + batch_size]
                
                # Process each citation in the current batch
                for citation, source_url in zip(batch, batch_urls):
                    try:
                        # Use the citation validation tool
                        result = await citation_tool._call({
                            'citation': citation,
                            'source_url': source_url,
                            'batch_mode': True
                        })
                        
                        validation_results.append({
                            'success': True,
                            'citation': citation,
                            'source_url': source_url,
                            'validation_result': result,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # Yield progress update
                        yield {
                            'status': 'in_progress',
                            'processed': len(validation_results),
                            'total': len(citations),
                            'current_citation': citation[:100] + ('...' if len(citation) > 100 else '')
                        }
                        
                    except Exception as e:
                        logger.error(f"Error validating citation: {str(e)}")
                        validation_results.append({
                            'success': False,
                            'citation': citation,
                            'source_url': source_url,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        })
            
            # Return final results
            yield {
                'status': 'completed',
                'success': True,
                'validation_results': validation_results,
                'total_citations': len(citations),
                'processed': len(validation_results),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in citation validation: {str(e)}")
            yield {
                'status': 'error',
                'success': False,
                'error': str(e),
                'total_citations': len(citations),
                'processed': len(validation_results) if 'validation_results' in locals() else 0,
                'timestamp': datetime.now().isoformat()
            }

    async def _process_search_result(self, result: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a single search result asynchronously.
        
        Args:
            result: The search result to process.
            context: Additional context for processing.
            
        Returns:
            Processed search result with extracted information.
        """
        if not result:
            return {'error': 'Empty result'}
            
        try:
            processed = {
                'title': result.get('title', ''),
                'url': result.get('url', ''),
                'snippet': result.get('snippet', ''),
                'source': result.get('source', 'unknown'),
                'metadata': {}
            }
            
            # Extract content if available
            content = result.get('content', '')
            if not content and 'snippet' in result:
                content = result['snippet']
                
            # Process content if we have any
            if content:
                # Try to extract entities
                try:
                    entity_tool = next((t for t in self.tools if isinstance(t, ContentExtractionTool)), None)
                    if entity_tool:
                        entities = await entity_tool._call({
                            'text': content,
                            'extract_entities': True
                        })
                        if entities and 'entities' in entities:
                            processed['entities'] = entities['entities']
                except Exception as e:
                    logger.warning(f"Error extracting entities: {str(e)}")
                
                # Try to extract citations
                try:
                    citation_tool = next((t for t in self.tools if isinstance(t, CitationValidationTool)), None)
                    if citation_tool:
                        citations = await citation_tool._call({
                            'text': content,
                            'extract_only': True
                        })
                        if citations and 'citations' in citations:
                            processed['citations'] = citations['citations']
                except Exception as e:
                    logger.warning(f"Error extracting citations: {str(e)}")
            
            # Add any additional metadata
            if 'metadata' in result and isinstance(result['metadata'], dict):
                processed['metadata'].update(result['metadata'])
                
            processed['metadata']['processed_at'] = datetime.now().isoformat()
            return processed
            
        except Exception as e:
            logger.error(f"Error processing search result: {str(e)}")
            return {
                'error': str(e),
                'url': result.get('url', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }
            
    @staticmethod
    def make_json_serializable(obj: Any) -> Any:
        """Convert non-serializable objects to a serializable format.
        
        Args:
            obj: The object to make JSON serializable.
            
        Returns:
            A JSON-serializable version of the object.
        """
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, (list, tuple)):
            return [DeepSearchAgent.make_json_serializable(item) for item in obj]
        if isinstance(obj, dict):
            return {str(k): DeepSearchAgent.make_json_serializable(v) for k, v in obj.items()}
        if hasattr(obj, '__dict__'):
            return DeepSearchAgent.make_json_serializable(obj.__dict__)
        if hasattr(obj, 'isoformat'):  # Handle datetime objects
            return obj.isoformat()
        return str(obj)
        
    def _process_search_results(self, search_results: List[Dict[str, Any]], 
                               query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process search results through the information integration workflow."""
        if not search_results:
            logger.warning("No valid search results to process")
            return {
                'success': False,
                'error': 'No valid search results to process',
                'query': query,
                'processed_results': []
            }
            
        processed_results = []
        all_entities = []
        all_citations = []
        
        logger.info(f"Processing {len(search_results)} search results")
        
        for idx, result in enumerate(search_results, 1):
            try:
                logger.debug(f"Processing result {idx}/{len(search_results)}")
                
                content = result.get('text', '') or result.get('content', '') or result.get('snippet', '')
                
                if not content:
                    logger.debug(f"Skipping result {idx} - no content")
                    continue
                
                processed_content = self._process_content(content, result.get('url', ''), context)
                
                entities = []
                if processed_content and processed_content.get('success', False):
                    try:
                        entities = self._extract_entities(processed_content.get('content', ''))
                        all_entities.extend(entities)
                    except Exception as e:
                        logger.warning(f"Failed to extract entities from result {idx}: {str(e)}")
                
                citations = []
                if processed_content and processed_content.get('success', False):
                    try:
                        citations = self._extract_citations(processed_content.get('content', ''))
                        all_citations.extend(citations)
                    except Exception as e:
                        logger.warning(f"Failed to extract citations from result {idx}: {str(e)}")
                
                result_entry = {
                    'title': result.get('title', 'No title'),
                    'url': result.get('url', ''),
                    'source': result.get('source', 'unknown'),
                    'content_summary': processed_content.get('summary', '') if isinstance(processed_content, dict) else str(processed_content)[:500],
                    'entities': entities,
                    'citations': citations,
                    'relevance_score': result.get('score', 0.0),
                    'metadata': {
                        'language': result.get('language', 'en'),
                        'last_updated': result.get('last_updated', ''),
                        'content_type': result.get('content_type', 'webpage'),
                        'processed_at': datetime.now().isoformat()
                    }
                }
                
                processed_results.append(result_entry)
                logger.debug(f"Successfully processed result {idx}")
                
            except Exception as e:
                logger.error(f"Error processing search result {idx}: {str(e)}", exc_info=True)
                continue
        
        summary = self._generate_search_summary(processed_results, query)
        processed_entities = self._process_entities(all_entities)
        processed_citations = self._process_citations(all_citations)
        
        return {
            'success': True,
            'query': query,
            'search_context': context,
            'result_count': len(processed_results),
            'processed_results': processed_results,
            'summary': summary,
            'entities': processed_entities,
            'citations': processed_citations,
            'metadata': {
                'processing_time': f"{len(processed_results)} results processed",
                'timestamp': datetime.now().isoformat(),
                'tools_used': [tool.name for tool in self.tools if hasattr(tool, 'name')]
            }
        }
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from the given text."""
        if not text:
            logger.debug("No text provided for entity extraction")
            return []
            
        try:
            entity_extraction_tool = next(
                (tool for tool in self.tools if hasattr(tool, 'name') and tool.name == 'entity_extraction'),
                None
            )
            
            if entity_extraction_tool:
                result = entity_extraction_tool._call(
                    context=ToolContext(),
                    text=text
                )
                if isinstance(result, list):
                    return result
                elif isinstance(result, dict) and 'entities' in result:
                    return result['entities']
            
            entities = []
            patterns = {
                'PERSON': r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                'ORGANIZATION': r'([A-Z][a-zA-Z0-9&]+(?:\s+[A-Z][a-zA-Z0-9&]+)+)',
                'LOCATION': r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:City|State|Country|Province|Region))',
            }
            
            for entity_type, pattern in patterns.items():
                matches = list(re.finditer(pattern, text))
                for match in matches:
                    entity_text = match.group(1)
                    if len(entity_text.split()) > 1:
                        entities.append({
                            'text': entity_text,
                            'type': entity_type,
                            'start': match.start(),
                            'end': match.end(),
                            'source': 'regex'
                        })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}", exc_info=True)
            return []
    
    def _extract_citations(self, text: str) -> List[Dict[str, Any]]:
        """Extract citations from the given text."""
        if not text:
            return []
            
        try:
            citation_tool = next(
                (tool for tool in self.tools if hasattr(tool, 'name') and tool.name == 'citation_validation'),
                None
            )
            
            if citation_tool:
                result = citation_tool._call(
                    context=ToolContext(),
                    text=text,
                    extract_only=True
                )
                if isinstance(result, list):
                    return result
                elif isinstance(result, dict) and 'citations' in result:
                    return result['citations']
            
            citations = []
            patterns = [
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+\.?)+(?:\s+et\s+al\.)?\s*\(\d{4}\))',
                r'(\[\d+(?:[,-]?\s*\d+)*\])',
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    citations.append({
                        'text': match.group(1),
                        'type': 'inline_citation',
                        'start': match.start(),
                        'end': match.end(),
                        'source': 'regex'
                    })
            
            return citations
            
        except Exception as e:
            logger.error(f"Error extracting citations: {str(e)}", exc_info=True)
            return []
    
    def _process_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process and aggregate extracted entities."""
        if not entities:
            return {}
            
        try:
            entity_counter = Counter()
            entity_types = defaultdict(list)
            
            for entity in entities:
                if not isinstance(entity, dict):
                    continue
                    
                entity_text = entity.get('text', '').strip()
                entity_type = entity.get('type', 'UNKNOWN').upper()
                
                if entity_text:
                    entity_counter[(entity_text, entity_type)] += 1
                    entity_types[entity_type].append(entity_text)
            
            most_common = entity_counter.most_common(10)
            
            return {
                'total_entities': len(entities),
                'unique_entities': len(entity_counter),
                'entity_types': dict(entity_types),
                'top_entities': [{'text': text, 'type': type_, 'count': count} 
                               for (text, type_), count in most_common]
            }
            
        except Exception as e:
            logger.error(f"Error processing entities: {str(e)}", exc_info=True)
            return {}
    
    def _process_citations(self, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process and aggregate extracted citations."""
        if not citations:
            return {}
            
        try:
            citation_types = defaultdict(int)
            for citation in citations:
                if isinstance(citation, dict):
                    citation_type = citation.get('type', 'unknown')
                    citation_types[citation_type] += 1
            
            return {
                'total_citations': len(citations),
                'citation_types': dict(citation_types)
            }
            
        except Exception as e:
            logger.error(f"Error processing citations: {str(e)}", exc_info=True)
            return {}
    
    def _generate_search_summary(self, results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Generate a summary of search results."""
        if not results:
            return {
                'summary': 'No results found for the query.',
                'sources': [],
                'total_results': 0
            }
            
        try:
            sources = Counter()
            content_types = Counter()
            
            for result in results:
                if not isinstance(result, dict):
                    continue
                    
                source = result.get('source', 'unknown')
                content_type = result.get('metadata', {}).get('content_type', 'unknown')
                
                sources[source] += 1
                content_types[content_type] += 1
            
            total_results = len(results)
            top_sources = sources.most_common(3)
            
            summary_parts = [
                f"Found {total_results} results for '{query}'.",
                f"Top sources: {', '.join([f'{source} ({count})' for source, count in top_sources])}.",
                f"Content types: {', '.join([f'{ctype} ({count})' for ctype, count in content_types.most_common()])}."
            ]
            
            return {
                'summary': ' '.join(summary_parts),
                'sources': [{'source': source, 'count': count} for source, count in sources.most_common()],
                'content_types': [{'type': ctype, 'count': count} for ctype, count in content_types.most_common()],
                'total_results': total_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating search summary: {str(e)}", exc_info=True)
            return {
                'summary': f'Found {len(results)} results for "{query}"',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _process_content(self, content: str, source_url: str = '', context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process content using available tools."""
        if not content:
            return {
                'success': False,
                'error': 'No content provided',
                'source_url': source_url,
                'processed': False
            }
            
        logger.debug(f"Processing content from {source_url or 'unknown source'}, length: {len(content)} chars")
        
        try:
            content_extraction_tool = next(
                (tool for tool in self.tools if hasattr(tool, 'name') and tool.name == 'content_extraction'),
                None
            )
            
            if content_extraction_tool:
                try:
                    logger.debug("Using content extraction tool")
                    result = content_extraction_tool._call(
                        context=ToolContext(),
                        content=content,
                        source_url=source_url,
                        context_info=context or {}
                    )
                    
                    if result and isinstance(result, dict):
                        result['processed'] = True
                        result['source_url'] = source_url
                        return result
                    
                    logger.warning("Content extraction returned invalid result, using fallback")
                    
                except Exception as e:
                    logger.error(f"Error in content extraction: {str(e)}", exc_info=True)
            
            logger.debug("Using fallback content processing")
            return {
                'content': content[:5000],
                'summary': content[:500] + ('...' if len(content) > 500 else ''),
                'source_url': source_url,
                'processed': True,
                'fallback': True,
                'metadata': {
                    'processing_method': 'fallback',
                    'content_length': len(content),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            error_msg = f"Unexpected error processing content: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                'content': content[:1000] if content else '',
                'source_url': source_url,
                'processed': False,
                'error': error_msg,
                'error_type': e.__class__.__name__,
                'metadata': {
                    'error': True,
                    'error_type': e.__class__.__name__,
                    'timestamp': datetime.now().isoformat()
                }
            }


# Create an instance of the Deep Search Agent
deep_search_agent_instance = DeepSearchAgent()

# Create an AgentTool from the Deep Search Agent
deep_search_agent_tool = AgentTool(agent=deep_search_agent_instance.agent)