"""Citation Graph Agent for ScholarVerse.

This module defines the Citation Graph Agent that constructs and maintains the knowledge graph in Neo4j.
"""

from typing import Dict, Any, List, Optional, Any
import os

from google.adk import Agent
from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.function_tool import FunctionTool

from scholar_verse.config import DEFAULT_MODEL
from scholar_verse.prompt import CITATION_GRAPH_AGENT_INSTRUCTIONS
from scholar_verse.shared_libraries.logging_utils import logger
from scholar_verse.shared_libraries.state_management.adaptive_state import AdaptiveStateManager

# Import citation graph tools
from scholar_verse.sub_agents.citation_graph.tools.neo4j_integration import Neo4jManager
from scholar_verse.sub_agents.citation_graph.tools.citation_extraction import CitationExtractor
from scholar_verse.sub_agents.citation_graph.tools.relationship_identification import RelationshipIdentifier
from scholar_verse.sub_agents.citation_graph.tools.nl2cypher import NL2Cypher


# Initialize tool components
neo4j_manager = Neo4jManager()
citation_extractor = CitationExtractor()
relationship_identifier = RelationshipIdentifier()
nl2cypher = NL2Cypher()


# Define tool functions
def connect_to_neo4j(uri: str, username: str, password: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Connect to a Neo4j database.
    
    Args:
        uri: The Neo4j connection URI.
        username: The Neo4j username.
        password: The Neo4j password.
        tool_context: The tool context.
        
    Returns:
        A dictionary indicating whether the connection was successful.
    """
    logger.info(f"Connecting to Neo4j at {uri}")
    
    # Initialize state manager if not already in context
    if 'state_manager' not in tool_context.state:
        tool_context.state['state_manager'] = AdaptiveStateManager()
    
    # Initialize Neo4j manager
    neo4j_manager = Neo4jManager(
        uri=uri,
        username=username,
        password=password,
        state_manager=tool_context.state['state_manager']
    )
    
    # Connect to Neo4j
    success = neo4j_manager.connect()
    
    if success:
        # Store Neo4j manager in the state
        tool_context.state['neo4j_manager'] = neo4j_manager
        return {"success": True, "message": f"Successfully connected to Neo4j at {uri}"}
    else:
        return {"success": False, "error": f"Failed to connect to Neo4j at {uri}"}


def setup_knowledge_graph(tool_context: ToolContext) -> Dict[str, Any]:
    """Set up the knowledge graph by creating constraints and indexes.
    
    Args:
        tool_context: The tool context.
        
    Returns:
        A dictionary with the results of the operation.
    """
    logger.info("Setting up knowledge graph")
    
    # Initialize state manager if not already in context
    if 'state_manager' not in tool_context.state:
        tool_context.state['state_manager'] = AdaptiveStateManager()
    
    # Get Neo4j manager from the state or create a new one
    neo4j_manager = tool_context.state.get('neo4j_manager')
    if not neo4j_manager:
        neo4j_manager = Neo4jManager(state_manager=tool_context.state['state_manager'])
        tool_context.state['neo4j_manager'] = neo4j_manager
    
    # Create constraints and indexes
    result = neo4j_manager.create_constraints_and_indexes()
    
    return result


def add_paper_to_graph(paper_data: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
    """Add a paper to the knowledge graph.
    
    Args:
        paper_data: The paper data to add.
        tool_context: The tool context.
        
    Returns:
        A dictionary with the results of the operation.
    """
    logger.info(f"Adding paper to knowledge graph: {paper_data.get('title', 'Unknown Title')}")
    
    # Initialize state manager if not already in context
    if 'state_manager' not in tool_context.state:
        tool_context.state['state_manager'] = AdaptiveStateManager()
    
    # Get Neo4j manager from the state or create a new one
    neo4j_manager = tool_context.state.get('neo4j_manager')
    if not neo4j_manager:
        neo4j_manager = Neo4jManager(state_manager=tool_context.state['state_manager'])
        tool_context.state['neo4j_manager'] = neo4j_manager
    
    # Add paper to the graph
    result = neo4j_manager.add_paper(paper_data, tool_context)
    
    return result


def add_citation(citing_paper: Dict[str, Any], cited_paper: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
    """Add a citation relationship between two papers.
    
    Args:
        citing_paper: The paper that cites another paper.
        cited_paper: The paper that is cited.
        tool_context: The tool context.
        
    Returns:
        A dictionary with the results of the operation.
    """
    logger.info(f"Adding citation relationship: {citing_paper.get('title', 'Unknown')} -> {cited_paper.get('title', 'Unknown')}")
    
    # Initialize state manager if not already in context
    if 'state_manager' not in tool_context.state:
        tool_context.state['state_manager'] = AdaptiveStateManager()
    
    # Get Neo4j manager from the state or create a new one
    neo4j_manager = tool_context.state.get('neo4j_manager')
    if not neo4j_manager:
        neo4j_manager = Neo4jManager(state_manager=tool_context.state['state_manager'])
        tool_context.state['neo4j_manager'] = neo4j_manager
    
    # Add citation relationship
    result = neo4j_manager.add_citation(citing_paper, cited_paper, tool_context)
    
    return result


def extract_citations(document_text: str, document_id: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Extract citations from document text.
    
    Args:
        document_text: The full text of the document.
        document_id: The ID of the document.
        tool_context: The tool context.
        
    Returns:
        A dictionary containing the extracted citations.
    """
    logger.info(f"Extracting citations from document: {document_id}")
    
    # Initialize state manager if not already in context
    if 'state_manager' not in tool_context.state:
        tool_context.state['state_manager'] = AdaptiveStateManager()
    
    # Extract citations
    result = citation_extractor.extract_citations(document_text, document_id, tool_context)
    
    return result


def match_citations_to_papers(citations: List[Dict[str, Any]], tool_context: ToolContext) -> Dict[str, Any]:
    """Match extracted citations to papers in the knowledge graph.
    
    Args:
        citations: The list of extracted citations.
        tool_context: The tool context.
        
    Returns:
        A dictionary with the results of the matching operation.
    """
    logger.info(f"Matching {len(citations)} citations to papers in the knowledge graph")
    
    # Initialize state manager if not already in context
    if 'state_manager' not in tool_context.state:
        tool_context.state['state_manager'] = AdaptiveStateManager()
    
    # Match citations to papers
    result = citation_extractor.match_citations_to_papers(citations, tool_context)
    
    return result


def identify_relationships(paper_data: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
    """Identify relationships between a paper and other entities.
    
    Args:
        paper_data: The paper data to analyze.
        tool_context: The tool context.
        
    Returns:
        A dictionary containing the identified relationships.
    """
    logger.info(f"Identifying relationships for paper: {paper_data.get('title', 'Unknown Title')}")
    
    # Initialize state manager if not already in context
    if 'state_manager' not in tool_context.state:
        tool_context.state['state_manager'] = AdaptiveStateManager()
    
    # Identify relationships
    result = relationship_identifier.identify_paper_relationships(paper_data, tool_context)
    
    return result


def analyze_citation_network(tool_context: ToolContext) -> Dict[str, Any]:
    """Analyze the citation network to identify important papers and authors.
    
    Args:
        tool_context: The tool context.
        
    Returns:
        A dictionary containing the citation network analysis results.
    """
    logger.info("Analyzing citation network")
    
    # Initialize state manager if not already in context
    if 'state_manager' not in tool_context.state:
        tool_context.state['state_manager'] = AdaptiveStateManager()
    
    # Analyze citation network
    result = relationship_identifier.analyze_citation_network(tool_context)
    
    return result


def identify_research_trends(tool_context: ToolContext) -> Dict[str, Any]:
    """Identify research trends based on temporal analysis of the citation network.
    
    Args:
        tool_context: The tool context.
        
    Returns:
        A dictionary containing the identified research trends.
    """
    logger.info("Identifying research trends")
    
    # Initialize state manager if not already in context
    if 'state_manager' not in tool_context.state:
        tool_context.state['state_manager'] = AdaptiveStateManager()
    
    # Identify research trends
    result = relationship_identifier.identify_research_trends(tool_context)
    
    return result


def translate_to_cypher(query: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Translate a natural language query to a Cypher query.
    
    Args:
        query: The natural language query.
        tool_context: The tool context.
        
    Returns:
        A dictionary containing the translated Cypher query and parameters.
    """
    logger.info(f"Translating query to Cypher: {query}")
    
    # Initialize state manager if not already in context
    if 'state_manager' not in tool_context.state:
        tool_context.state['state_manager'] = AdaptiveStateManager()
    
    # Translate query to Cypher
    result = nl2cypher.translate_to_cypher(query, tool_context)
    
    return result


def execute_cypher(cypher: str, parameters: Optional[Dict[str, Any]] = None, tool_context: ToolContext = None) -> Dict[str, Any]:
    """Execute a Cypher query.
    
    Args:
        cypher: The Cypher query to execute.
        parameters: The query parameters.
        tool_context: The tool context.
        
    Returns:
        A dictionary with the results of the operation.
    """
    logger.info(f"Executing Cypher query: {cypher}")
    
    # Initialize state manager if not already in context
    if 'state_manager' not in tool_context.state:
        tool_context.state['state_manager'] = AdaptiveStateManager()
    
    # Get Neo4j manager from the state or create a new one
    neo4j_manager = tool_context.state.get('neo4j_manager')
    if not neo4j_manager:
        neo4j_manager = Neo4jManager(state_manager=tool_context.state['state_manager'])
        tool_context.state['neo4j_manager'] = neo4j_manager
    
    # Execute Cypher query
    result = neo4j_manager.execute_cypher(cypher, parameters, tool_context)
    
    return result


def get_graph_statistics(tool_context: ToolContext) -> Dict[str, Any]:
    """Get statistics about the knowledge graph.
    
    Args:
        tool_context: The tool context.
        
    Returns:
        A dictionary with graph statistics.
    """
    logger.info("Getting knowledge graph statistics")
    
    # Initialize state manager if not already in context
    if 'state_manager' not in tool_context.state:
        tool_context.state['state_manager'] = AdaptiveStateManager()
    
    # Get Neo4j manager from the state or create a new one
    neo4j_manager = tool_context.state.get('neo4j_manager')
    if not neo4j_manager:
        neo4j_manager = Neo4jManager(state_manager=tool_context.state['state_manager'])
        tool_context.state['neo4j_manager'] = neo4j_manager
    
    # Get graph statistics
    result = neo4j_manager.get_graph_statistics(tool_context)
    
    return result


# Create function tools
connect_to_neo4j_tool = FunctionTool(connect_to_neo4j)
setup_knowledge_graph_tool = FunctionTool(setup_knowledge_graph)
add_paper_to_graph_tool = FunctionTool(add_paper_to_graph)
add_citation_tool = FunctionTool(add_citation)
extract_citations_tool = FunctionTool(extract_citations)
match_citations_to_papers_tool = FunctionTool(match_citations_to_papers)
identify_relationships_tool = FunctionTool(identify_relationships)
analyze_citation_network_tool = FunctionTool(analyze_citation_network)
identify_research_trends_tool = FunctionTool(identify_research_trends)
translate_to_cypher_tool = FunctionTool(translate_to_cypher)
execute_cypher_tool = FunctionTool(execute_cypher)
get_graph_statistics_tool = FunctionTool(get_graph_statistics)


class CitationGraphAgent(Agent):
    """Citation Graph Agent for managing and analyzing citation networks.
    
    This agent handles the construction, maintenance, and analysis of citation graphs
    using Neo4j as the underlying graph database.
    """
    
    # Define class attributes with type hints for Pydantic
    neo4j_manager: Any = None
    citation_extractor: Any = None
    relationship_identifier: Any = None
    nl2cypher: Any = None
    
    def __init__(self, **data):
        """Initialize the CitationGraphAgent with its tools and configuration."""
        # Initialize the base class first
        super().__init__(
            name="citation_graph_agent",
            instruction=CITATION_GRAPH_AGENT_INSTRUCTIONS,
            model=DEFAULT_MODEL,
            tools=[
                connect_to_neo4j_tool,
                setup_knowledge_graph_tool,
                add_paper_to_graph_tool,
                add_citation_tool,
                extract_citations_tool,
                match_citations_to_papers_tool,
                identify_relationships_tool,
                analyze_citation_network_tool,
                identify_research_trends_tool,
                translate_to_cypher_tool,
                execute_cypher_tool,
                get_graph_statistics_tool,
            ],
            **data
        )
        
        # Initialize component instances
        self.neo4j_manager = neo4j_manager
        self.citation_extractor = citation_extractor
        self.relationship_identifier = relationship_identifier
        self.nl2cypher = nl2cypher


# Create an instance of the CitationGraphAgent
citation_graph_agent = CitationGraphAgent()

# Create an AgentTool from the CitationGraphAgent
citation_graph_agent_tool = AgentTool(agent=citation_graph_agent)

# Export the CitationGraphAgent class and the agent instance
__all__ = ['CitationGraphAgent', 'citation_graph_agent', 'citation_graph_agent_tool']
