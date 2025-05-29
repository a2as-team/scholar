"""Natural Language to Cypher translation for the ScholarVerse Citation Graph Agent.

This module provides tools for translating natural language queries to Cypher queries.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import re
from datetime import datetime

from google.adk.tools import ToolContext

from scholar_verse.shared_libraries.logging_utils import logger
from scholar_verse.shared_libraries.state_management.adaptive_state import AdaptiveStateManager


class NL2Cypher:
    """Natural Language to Cypher translator for ScholarVerse.
    
    This class provides methods for translating natural language queries to Cypher queries
    for querying the Neo4j knowledge graph.
    """
    
    def __init__(self, state_manager: Optional[AdaptiveStateManager] = None):
        """Initialize the NL2Cypher translator.
        
        Args:
            state_manager: The state manager to use for storing translation history.
        """
        self.state_manager = state_manager or AdaptiveStateManager()
        
        # Define query templates for common query types
        self.query_templates = {
            "find_paper": """
            MATCH (p:Paper)
            WHERE p.title CONTAINS $title OR p.doi = $doi
            RETURN p.title as title, p.doi as doi, p.year as year, p.abstract as abstract
            LIMIT 10
            """,
            
            "find_author": """
            MATCH (a:Author)-[:AUTHORED]->(p:Paper)
            WHERE a.name CONTAINS $name
            RETURN a.name as author, collect(p.title) as papers, count(p) as paper_count
            LIMIT 10
            """,
            
            "find_citations": """
            MATCH (p1:Paper)-[:CITES]->(p2:Paper)
            WHERE p1.title CONTAINS $title OR p1.doi = $doi
            RETURN p1.title as citing_paper, p2.title as cited_paper, p2.doi as cited_doi
            LIMIT 20
            """,
            
            "find_references": """
            MATCH (p1:Paper)-[:CITES]->(p2:Paper)
            WHERE p2.title CONTAINS $title OR p2.doi = $doi
            RETURN p2.title as cited_paper, p1.title as citing_paper, p1.doi as citing_doi
            LIMIT 20
            """,
            
            "find_collaborators": """
            MATCH (a1:Author)-[:AUTHORED]->(p:Paper)<-[:AUTHORED]-(a2:Author)
            WHERE a1.name CONTAINS $name AND a1 <> a2
            RETURN a1.name as author, a2.name as collaborator, count(p) as collaboration_count
            ORDER BY collaboration_count DESC
            LIMIT 20
            """,
            
            "find_concepts": """
            MATCH (p:Paper)-[:HAS_CONCEPT]->(c:Concept)
            WHERE p.title CONTAINS $title OR p.doi = $doi
            RETURN p.title as paper, collect(c.name) as concepts
            LIMIT 10
            """,
            
            "papers_by_concept": """
            MATCH (p:Paper)-[:HAS_CONCEPT]->(c:Concept)
            WHERE c.name CONTAINS $concept
            RETURN c.name as concept, collect(p.title) as papers, count(p) as paper_count
            LIMIT 20
            """,
            
            "papers_by_year": """
            MATCH (p:Paper)
            WHERE p.year = $year
            RETURN p.year as year, collect(p.title) as papers, count(p) as paper_count
            LIMIT 100
            """,
            
            "citation_count": """
            MATCH (p:Paper)<-[c:CITES]-()
            WHERE p.title CONTAINS $title OR p.doi = $doi
            RETURN p.title as paper, count(c) as citation_count
            LIMIT 10
            """,
            
            "most_cited_papers": """
            MATCH (p:Paper)<-[c:CITES]-()
            RETURN p.title as paper, p.doi as doi, count(c) as citation_count
            ORDER BY citation_count DESC
            LIMIT $limit
            """,
            
            "most_cited_authors": """
            MATCH (a:Author)-[:AUTHORED]->(p:Paper)<-[c:CITES]-()
            RETURN a.name as author, count(c) as citation_count, collect(DISTINCT p.title) as papers
            ORDER BY citation_count DESC
            LIMIT $limit
            """,
            
            "citation_graph": """
            MATCH path = (p1:Paper)-[:CITES*1..2]->(p2:Paper)
            WHERE p1.title CONTAINS $title OR p1.doi = $doi
            RETURN p1.title as source, p2.title as target
            LIMIT 50
            """,
        }
    
    def translate_to_cypher(self, query: str, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """Translate a natural language query to a Cypher query.
        
        Args:
            query: The natural language query.
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A dictionary containing the translated Cypher query and parameters.
        """
        logger.info(f"Translating query to Cypher: {query}")
        
        # Get state manager from context if available
        if tool_context:
            state_manager = tool_context.state.get('state_manager', self.state_manager)
        else:
            state_manager = self.state_manager
        
        try:
            # Identify query type and extract parameters
            query_type, parameters = self._identify_query_type(query)
            
            if query_type:
                # Use template for the identified query type
                cypher_query = self.query_templates.get(query_type)
                
                # Store the translation in the state if tool_context is provided
                if tool_context:
                    # Get the current state
                    current_state = state_manager.get(tool_context)
                    
                    # Initialize nl2cypher in state if not present
                    if 'nl2cypher' not in current_state:
                        current_state['nl2cypher'] = {
                            'history': [],
                        }
                    
                    # Add translation to history
                    current_state['nl2cypher']['history'].append({
                        'natural_language': query,
                        'query_type': query_type,
                        'cypher': cypher_query,
                        'parameters': parameters,
                        'timestamp': datetime.now().isoformat(),
                    })
                    
                    # Limit history to last 20 translations
                    if len(current_state['nl2cypher']['history']) > 20:
                        current_state['nl2cypher']['history'] = current_state['nl2cypher']['history'][-20:]
                    
                    # Update the state
                    state_manager.set(tool_context, current_state)
                
                return {
                    "success": True,
                    "query_type": query_type,
                    "cypher": cypher_query,
                    "parameters": parameters,
                }
            else:
                # If no template matches, generate a custom query
                cypher_query, parameters = self._generate_custom_query(query)
                
                # Store the translation in the state if tool_context is provided
                if tool_context:
                    # Get the current state
                    current_state = state_manager.get(tool_context)
                    
                    # Initialize nl2cypher in state if not present
                    if 'nl2cypher' not in current_state:
                        current_state['nl2cypher'] = {
                            'history': [],
                        }
                    
                    # Add translation to history
                    current_state['nl2cypher']['history'].append({
                        'natural_language': query,
                        'query_type': 'custom',
                        'cypher': cypher_query,
                        'parameters': parameters,
                        'timestamp': datetime.now().isoformat(),
                    })
                    
                    # Limit history to last 20 translations
                    if len(current_state['nl2cypher']['history']) > 20:
                        current_state['nl2cypher']['history'] = current_state['nl2cypher']['history'][-20:]
                    
                    # Update the state
                    state_manager.set(tool_context, current_state)
                
                return {
                    "success": True,
                    "query_type": "custom",
                    "cypher": cypher_query,
                    "parameters": parameters,
                }
        except Exception as e:
            error_msg = f"Error translating query to Cypher: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
            }
    
    def _identify_query_type(self, query: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """Identify the type of query and extract parameters.
        
        Args:
            query: The natural language query.
            
        Returns:
            A tuple of (query_type, parameters).
        """
        query = query.lower()
        parameters = {}
        
        # Find papers by title or DOI
        if re.search(r'find paper|paper titled|paper with title|paper with doi|paper doi', query):
            # Extract title
            title_match = re.search(r'titled[\s"\']*([^"\']+)[\s"\']*', query) or \
                         re.search(r'title[\s:]*["\']*([^"\']+)[\s"\']*', query) or \
                         re.search(r'paper[\s:]*["\']*([^"\']+)[\s"\']*', query)
            
            # Extract DOI
            doi_match = re.search(r'doi[\s:]*["\']*([^"\']+)[\s"\']*', query)
            
            if title_match:
                parameters['title'] = title_match.group(1).strip()
            else:
                parameters['title'] = ""
                
            if doi_match:
                parameters['doi'] = doi_match.group(1).strip()
            else:
                parameters['doi'] = ""
            
            return "find_paper", parameters
        
        # Find author
        elif re.search(r'find author|author named|papers by author|publications by', query):
            # Extract author name
            author_match = re.search(r'author[\s:]*["\']*([^"\']+)[\s"\']*', query) or \
                          re.search(r'by[\s:]*["\']*([^"\']+)[\s"\']*', query) or \
                          re.search(r'named[\s:]*["\']*([^"\']+)[\s"\']*', query)
            
            if author_match:
                parameters['name'] = author_match.group(1).strip()
                return "find_author", parameters
        
        # Find citations (papers that cite a given paper)
        elif re.search(r'papers that cite|citations of|who cites|cited by', query):
            # Extract paper title or DOI
            title_match = re.search(r'paper[\s:]*["\']*([^"\']+)[\s"\']*', query) or \
                         re.search(r'titled[\s:]*["\']*([^"\']+)[\s"\']*', query) or \
                         re.search(r'cite[\s:]*["\']*([^"\']+)[\s"\']*', query)
            
            doi_match = re.search(r'doi[\s:]*["\']*([^"\']+)[\s"\']*', query)
            
            if title_match:
                parameters['title'] = title_match.group(1).strip()
            else:
                parameters['title'] = ""
                
            if doi_match:
                parameters['doi'] = doi_match.group(1).strip()
            else:
                parameters['doi'] = ""
            
            return "find_references", parameters
        
        # Find references (papers cited by a given paper)
        elif re.search(r'papers cited by|references of|references in|cites which papers', query):
            # Extract paper title or DOI
            title_match = re.search(r'paper[\s:]*["\']*([^"\']+)[\s"\']*', query) or \
                         re.search(r'titled[\s:]*["\']*([^"\']+)[\s"\']*', query) or \
                         re.search(r'by[\s:]*["\']*([^"\']+)[\s"\']*', query)
            
            doi_match = re.search(r'doi[\s:]*["\']*([^"\']+)[\s"\']*', query)
            
            if title_match:
                parameters['title'] = title_match.group(1).strip()
            else:
                parameters['title'] = ""
                
            if doi_match:
                parameters['doi'] = doi_match.group(1).strip()
            else:
                parameters['doi'] = ""
            
            return "find_citations", parameters
        
        # Find collaborators
        elif re.search(r'collaborators of|who collaborated with|co-authors of', query):
            # Extract author name
            author_match = re.search(r'of[\s:]*["\']*([^"\']+)[\s"\']*', query) or \
                          re.search(r'with[\s:]*["\']*([^"\']+)[\s"\']*', query) or \
                          re.search(r'author[\s:]*["\']*([^"\']+)[\s"\']*', query)
            
            if author_match:
                parameters['name'] = author_match.group(1).strip()
                return "find_collaborators", parameters
        
        # Find concepts in a paper
        elif re.search(r'concepts in|topics in|keywords in', query):
            # Extract paper title or DOI
            title_match = re.search(r'paper[\s:]*["\']*([^"\']+)[\s"\']*', query) or \
                         re.search(r'titled[\s:]*["\']*([^"\']+)[\s"\']*', query) or \
                         re.search(r'in[\s:]*["\']*([^"\']+)[\s"\']*', query)
            
            doi_match = re.search(r'doi[\s:]*["\']*([^"\']+)[\s"\']*', query)
            
            if title_match:
                parameters['title'] = title_match.group(1).strip()
            else:
                parameters['title'] = ""
                
            if doi_match:
                parameters['doi'] = doi_match.group(1).strip()
            else:
                parameters['doi'] = ""
            
            return "find_concepts", parameters
        
        # Find papers by concept
        elif re.search(r'papers about|papers on topic|papers on concept|papers related to', query):
            # Extract concept
            concept_match = re.search(r'about[\s:]*["\']*([^"\']+)[\s"\']*', query) or \
                           re.search(r'on[\s:]*["\']*([^"\']+)[\s"\']*', query) or \
                           re.search(r'to[\s:]*["\']*([^"\']+)[\s"\']*', query)
            
            if concept_match:
                parameters['concept'] = concept_match.group(1).strip()
                return "papers_by_concept", parameters
        
        # Find papers by year
        elif re.search(r'papers from year|papers published in|papers in year', query):
            # Extract year
            year_match = re.search(r'year[\s:]*[\"\']*(\d{4})[\s\"\']*', query) or \
                        re.search(r'in[\s:]*[\"\']*(\d{4})[\s\"\']*', query) or \
                        re.search(r'from[\s:]*[\"\']*(\d{4})[\s\"\']*', query)
            
            if year_match:
                parameters['year'] = year_match.group(1).strip()
                return "papers_by_year", parameters
        
        # Get citation count
        elif re.search(r'citation count|how many citations|number of citations', query):
            # Extract paper title or DOI
            title_match = re.search(r'for[\s:]*["\']*([^"\']+)[\s"\']*', query) or \
                         re.search(r'paper[\s:]*["\']*([^"\']+)[\s"\']*', query) or \
                         re.search(r'titled[\s:]*["\']*([^"\']+)[\s"\']*', query)
            
            doi_match = re.search(r'doi[\s:]*["\']*([^"\']+)[\s"\']*', query)
            
            if title_match:
                parameters['title'] = title_match.group(1).strip()
            else:
                parameters['title'] = ""
                
            if doi_match:
                parameters['doi'] = doi_match.group(1).strip()
            else:
                parameters['doi'] = ""
            
            return "citation_count", parameters
        
        # Get most cited papers
        elif re.search(r'most cited papers|top cited papers|papers with most citations', query):
            # Extract limit
            limit_match = re.search(r'top[\s:]*(\d+)', query) or \
                         re.search(r'(\d+)[\s:]*most', query)
            
            if limit_match:
                parameters['limit'] = int(limit_match.group(1).strip())
            else:
                parameters['limit'] = 10
            
            return "most_cited_papers", parameters
        
        # Get most cited authors
        elif re.search(r'most cited authors|top cited authors|authors with most citations', query):
            # Extract limit
            limit_match = re.search(r'top[\s:]*(\d+)', query) or \
                         re.search(r'(\d+)[\s:]*most', query)
            
            if limit_match:
                parameters['limit'] = int(limit_match.group(1).strip())
            else:
                parameters['limit'] = 10
            
            return "most_cited_authors", parameters
        
        # Get citation graph
        elif re.search(r'citation graph|citation network|citation relationships', query):
            # Extract paper title or DOI
            title_match = re.search(r'for[\s:]*["\']*([^"\']+)[\s"\']*', query) or \
                         re.search(r'paper[\s:]*["\']*([^"\']+)[\s"\']*', query) or \
                         re.search(r'titled[\s:]*["\']*([^"\']+)[\s"\']*', query)
            
            doi_match = re.search(r'doi[\s:]*["\']*([^"\']+)[\s"\']*', query)
            
            if title_match:
                parameters['title'] = title_match.group(1).strip()
            else:
                parameters['title'] = ""
                
            if doi_match:
                parameters['doi'] = doi_match.group(1).strip()
            else:
                parameters['doi'] = ""
            
            return "citation_graph", parameters
        
        # No matching template found
        return None, {}
    
    def _generate_custom_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Generate a custom Cypher query for a natural language query.
        
        Args:
            query: The natural language query.
            
        Returns:
            A tuple of (cypher_query, parameters).
        """
        # This is a simplified implementation that handles a few common cases
        # In a production system, this would use a more sophisticated NLP approach
        
        query = query.lower()
        parameters = {}
        
        # Check for paper-related queries
        if 'paper' in query:
            if 'author' in query:
                # Find papers by a specific author
                author_match = re.search(r'author[\s:]*["\']*([^"\']+)[\s"\']*', query) or \
                              re.search(r'by[\s:]*["\']*([^"\']+)[\s"\']*', query)
                
                if author_match:
                    parameters['author_name'] = author_match.group(1).strip()
                    cypher = """
                    MATCH (a:Author)-[:AUTHORED]->(p:Paper)
                    WHERE a.name CONTAINS $author_name
                    RETURN p.title as title, p.year as year, p.doi as doi, a.name as author
                    LIMIT 20
                    """
                    return cypher, parameters
            
            if 'year' in query:
                # Find papers from a specific year
                year_match = re.search(r'year[\s:]*[\"\']*(\d{4})[\s\"\']*', query) or \
                            re.search(r'in[\s:]*[\"\']*(\d{4})[\s\"\']*', query) or \
                            re.search(r'from[\s:]*[\"\']*(\d{4})[\s\"\']*', query)
                
                if year_match:
                    parameters['year'] = year_match.group(1).strip()
                    cypher = """
                    MATCH (p:Paper)
                    WHERE p.year = $year
                    RETURN p.title as title, p.year as year, p.doi as doi
                    LIMIT 50
                    """
                    return cypher, parameters
            
            if 'title' in query:
                # Find papers with a specific title
                title_match = re.search(r'title[\s:]*["\']*([^"\']+)[\s"\']*', query)
                
                if title_match:
                    parameters['title'] = title_match.group(1).strip()
                    cypher = """
                    MATCH (p:Paper)
                    WHERE p.title CONTAINS $title
                    RETURN p.title as title, p.year as year, p.doi as doi, p.abstract as abstract
                    LIMIT 10
                    """
                    return cypher, parameters
        
        # Check for author-related queries
        if 'author' in query:
            if 'most papers' in query or 'most publications' in query:
                # Find authors with the most papers
                limit_match = re.search(r'top[\s:]*(\d+)', query) or \
                             re.search(r'(\d+)[\s:]*most', query)
                
                if limit_match:
                    parameters['limit'] = int(limit_match.group(1).strip())
                else:
                    parameters['limit'] = 10
                
                cypher = """
                MATCH (a:Author)-[:AUTHORED]->(p:Paper)
                RETURN a.name as author, count(p) as paper_count, collect(p.title)[0..5] as sample_papers
                ORDER BY paper_count DESC
                LIMIT $limit
                """
                return cypher, parameters
        
        # Check for concept-related queries
        if 'concept' in query or 'topic' in query or 'keyword' in query:
            if 'most papers' in query or 'most common' in query:
                # Find most common concepts/topics
                limit_match = re.search(r'top[\s:]*(\d+)', query) or \
                             re.search(r'(\d+)[\s:]*most', query)
                
                if limit_match:
                    parameters['limit'] = int(limit_match.group(1).strip())
                else:
                    parameters['limit'] = 10
                
                cypher = """
                MATCH (p:Paper)-[:HAS_CONCEPT]->(c:Concept)
                RETURN c.name as concept, count(p) as paper_count
                ORDER BY paper_count DESC
                LIMIT $limit
                """
                return cypher, parameters
        
        # Default query - return some papers
        cypher = """
        MATCH (p:Paper)
        RETURN p.title as title, p.year as year, p.doi as doi
        LIMIT 10
        """
        return cypher, parameters
    
    def get_translation_history(self, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """Get the history of NL2Cypher translations.
        
        Args:
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A dictionary containing the translation history.
        """
        logger.info("Getting NL2Cypher translation history")
        
        # Get state manager from context if available
        if tool_context:
            state_manager = tool_context.state.get('state_manager', self.state_manager)
        else:
            state_manager = self.state_manager
        
        # Get the current state
        current_state = state_manager.get(tool_context) if tool_context else {}
        
        # Get translation history from state
        history = current_state.get('nl2cypher', {}).get('history', [])
        
        return {
            "success": True,
            "history": history,
            "count": len(history),
        }
