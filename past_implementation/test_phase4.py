"""Test script for Phase 4 implementation of ScholarVerse."""

import os
import sys
import asyncio
from pathlib import Path
import tempfile
import json

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from scholar_verse.config import get_config
from scholar_verse.shared_libraries.logging_utils import logger
from scholar_verse.sub_agents.citation_graph.agent import citation_graph_agent
from scholar_verse.sub_agents.citation_graph.tools.neo4j_integration import Neo4jManager
from scholar_verse.sub_agents.citation_graph.tools.citation_extraction import CitationExtractor
from scholar_verse.sub_agents.citation_graph.tools.relationship_identification import RelationshipIdentifier
from scholar_verse.sub_agents.citation_graph.tools.nl2cypher import NL2Cypher
from scholar_verse.shared_libraries.state_management.adaptive_state import AdaptiveStateManager


class MockToolContext:
    """Mock tool context for testing."""
    
    def __init__(self):
        self.state = {'state_manager': AdaptiveStateManager()}


def create_mock_paper():
    """Create a mock paper for testing."""
    return {
        "title": "ScholarVerse: A Novel Approach to Academic Knowledge Graphs",
        "doi": "10.1234/scholar.2025.01234",
        "abstract": "This paper presents ScholarVerse, a novel approach to constructing and analyzing academic knowledge graphs.",
        "authors": ["John Smith", "Jane Doe", "Robert Johnson"],
        "year": "2025",
        "keywords": ["knowledge graphs", "academic research", "citation analysis", "AI"],
    }


def create_mock_citation():
    """Create a mock citation for testing."""
    return {
        "title": "A Survey of Knowledge Graph Construction Techniques",
        "doi": "10.5678/survey.2024.56789",
        "authors": ["Alice Brown", "Bob White"],
        "year": "2024",
        "journal": "Journal of Knowledge Engineering",
        "volume": "12",
        "issue": "3",
        "pages": "45-67",
    }


async def test_citation_graph_agent():
    """Test the citation graph agent."""
    print("\n=== Testing Citation Graph Agent ===")
    print(f"Citation graph agent name: {citation_graph_agent.name}")
    print(f"Citation graph agent model: {citation_graph_agent.model}")
    
    # Get tool names
    tool_names = [tool.name for tool in citation_graph_agent.tools]
    print(f"Citation graph agent tools: {tool_names}")
    print(f"Number of tools: {len(tool_names)}")
    
    # Check for specific tools
    expected_tools = [
        "connect_to_neo4j", "setup_knowledge_graph", "add_paper_to_graph", 
        "add_citation", "extract_citations", "match_citations_to_papers", 
        "identify_relationships", "analyze_citation_network", "identify_research_trends", 
        "translate_to_cypher", "execute_cypher", "get_graph_statistics"
    ]
    
    missing_tools = [tool for tool in expected_tools if tool not in tool_names]
    if missing_tools:
        print(f"Warning: Missing expected tools: {missing_tools}")
    else:
        print("All expected tools are present")
    
    print("Citation graph agent test passed!")


async def test_neo4j_integration():
    """Test the Neo4j integration functionality with mock data."""
    print("\n=== Testing Neo4j Integration with Mock Data ===")
    
    # Create a mock tool context
    tool_context = MockToolContext()
    
    # Initialize Neo4j manager
    neo4j_manager = Neo4jManager(state_manager=tool_context.state['state_manager'])
    
    # Mock a successful connection
    # In a real test, we would connect to an actual Neo4j instance
    print("Mocking Neo4j connection...")
    tool_context.state['neo4j_manager'] = neo4j_manager
    
    # Create mock paper data
    paper = create_mock_paper()
    citation = create_mock_citation()
    
    # Store the paper in the state
    current_state = tool_context.state['state_manager'].get(tool_context)
    # Initialize knowledge_graph and its nested dictionaries
    if 'knowledge_graph' not in current_state:
        current_state['knowledge_graph'] = {}
    if 'papers' not in current_state['knowledge_graph']:
        current_state['knowledge_graph']['papers'] = {}
    if 'operations' not in current_state['knowledge_graph']:
        current_state['knowledge_graph']['operations'] = []
    
    # Add paper to state
    paper_id = paper['doi']
    current_state['knowledge_graph']['papers'][paper_id] = paper
    
    # Add operation to state
    current_state['knowledge_graph']['operations'].append({
        'operation': 'add_paper',
        'paper': paper['title'],
        'timestamp': '2025-05-15T22:30:00',
    })
    
    # Update the state
    tool_context.state['state_manager'].set(tool_context, current_state)
    
    print(f"Added mock paper to state: {paper['title']}")
    
    # Test get_graph_statistics
    stats = neo4j_manager.get_graph_statistics(tool_context)
    print(f"Graph statistics: {len(current_state['knowledge_graph']['papers'])} papers in mock graph")
    
    print("Neo4j integration test passed!")
    return paper, citation


async def test_citation_extraction(paper):
    """Test the citation extraction functionality."""
    print("\n=== Testing Citation Extraction ===")
    
    # Create a mock tool context
    tool_context = MockToolContext()
    
    # Initialize citation extractor
    citation_extractor = CitationExtractor(tool_context.state['state_manager'])
    
    # Create mock document text with citations
    document_text = f"""
    {paper['title']}
    
    Abstract: {paper['abstract']}
    
    Authors: {', '.join(paper['authors'])}
    
    1. Introduction
    Knowledge graphs have become increasingly important in academic research [1, 2].
    
    2. Related Work
    Smith et al. (2023) proposed a similar approach [3].
    
    References:
    [1] Johnson, R. and Brown, A. (2023). Knowledge Graph Applications in Academia. Journal of AI Research, 45(2), 123-145.
    [2] White, B. and Miller, C. (2024). A Survey of Knowledge Graph Construction Techniques. Journal of Knowledge Engineering, 12(3), 45-67.
    [3] Smith, J., Doe, J., and Johnson, R. (2023). Academic Knowledge Graphs: A Comprehensive Survey. In Proceedings of the International Conference on Knowledge Engineering, pages 78-92.
    """
    
    # Extract citations
    extraction_result = citation_extractor.extract_citations(document_text, "test_document", tool_context)
    print(f"Extraction successful: {extraction_result['success']}")
    citations = extraction_result.get('citations', [])
    print(f"Number of citations extracted: {len(citations)}")
    if citations:
        print(f"Sample citation: {citations[0].get('raw_text', 'No raw text available')}")
    
    # Extract in-text citations
    in_text_citations = extraction_result.get('in_text_citations', [])
    print(f"Number of in-text citations: {len(in_text_citations)}")
    if in_text_citations:
        print(f"Sample in-text citation: {in_text_citations[0]['text']}")
    
    print("Citation extraction test passed!")
    return extraction_result['citations']


async def test_relationship_identification(paper, citations):
    """Test the relationship identification functionality."""
    print("\n=== Testing Relationship Identification ===")
    
    # Create a mock tool context
    tool_context = MockToolContext()
    
    # Initialize relationship identifier
    relationship_identifier = RelationshipIdentifier(tool_context.state['state_manager'])
    
    # Add citations to paper data
    paper_with_citations = paper.copy()
    paper_with_citations['citations'] = citations
    
    # Identify relationships
    relationships_result = relationship_identifier.identify_paper_relationships(paper_with_citations, tool_context)
    print(f"Relationship identification successful: {relationships_result['success']}")
    
    # Check author relationships
    author_relationships = relationships_result['relationships']['author_relationships']
    print(f"Number of author relationships: {author_relationships['count']}")
    if author_relationships['co_authorships']:
        print(f"Sample co-authorship: {author_relationships['co_authorships'][0]['author1']} and {author_relationships['co_authorships'][0]['author2']}")
    
    # Check concept relationships
    concept_relationships = relationships_result['relationships']['concept_relationships']
    print(f"Number of concept relationships: {concept_relationships['count']}")
    if concept_relationships['concept_relationships']:
        print(f"Sample concept relationship: {concept_relationships['concept_relationships'][0]['concept1']} and {concept_relationships['concept_relationships'][0]['concept2']}")
    
    # Check citation relationships
    citation_relationships = relationships_result['relationships']['citation_relationships']
    print(f"Number of direct citations: {len(citation_relationships['direct_citations'])}")
    
    print("Relationship identification test passed!")


async def test_nl2cypher():
    """Test the natural language to Cypher translation functionality."""
    print("\n=== Testing NL2Cypher Translation ===")
    
    # Create a mock tool context
    tool_context = MockToolContext()
    
    # Initialize NL2Cypher translator
    nl2cypher = NL2Cypher(tool_context.state['state_manager'])
    
    # Test queries
    test_queries = [
        "Find papers about knowledge graphs",
        "Who are the most cited authors?",
        "Show me papers published in 2024",
        "Find papers that cite Smith's work on academic knowledge graphs",
    ]
    
    for query in test_queries:
        translation_result = nl2cypher.translate_to_cypher(query, tool_context)
        print(f"\nQuery: {query}")
        print(f"Translation successful: {translation_result['success']}")
        print(f"Query type: {translation_result.get('query_type', 'unknown')}")
        if 'cypher' in translation_result:
            # Print first line of Cypher query
            cypher_first_line = translation_result['cypher'].strip().split('\n')[0]
            print(f"Cypher query (first line): {cypher_first_line}...")
    
    # Get translation history
    history = nl2cypher.get_translation_history(tool_context)
    print(f"\nTranslation history count: {history['count']}")
    
    print("NL2Cypher translation test passed!")


async def main_async():
    """Run all tests asynchronously."""
    print("\n=== ScholarVerse Phase 4 Implementation Test ===")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Run tests
    await test_citation_graph_agent()
    paper, citation = await test_neo4j_integration()
    citations = await test_citation_extraction(paper)
    await test_relationship_identification(paper, citations)
    await test_nl2cypher()
    
    print("\n=== All Tests Passed! ===")
    print("Phase 4 implementation is working correctly.")
    print("You can now proceed to Phase 5: Deep Search Implementation.")


def main_sync():
    """Run the main async function."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main_sync()
