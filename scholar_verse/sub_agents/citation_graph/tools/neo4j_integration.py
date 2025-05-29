"""Neo4j integration tools for the ScholarVerse Citation Graph Agent.

This module provides tools for integrating with Neo4j and constructing the knowledge graph.
"""

from typing import Dict, Any, List, Optional, Union
import os
from datetime import datetime
import json

from neo4j import GraphDatabase, Driver, Session
from google.adk.tools import ToolContext

from scholar_verse.shared_libraries.logging_utils import logger
from scholar_verse.shared_libraries.state_management.adaptive_state import AdaptiveStateManager


class Neo4jManager:
    """Neo4j database manager for ScholarVerse.
    
    This class provides methods for connecting to Neo4j, executing queries,
    and managing the knowledge graph database.
    """
    
    def __init__(self, 
                 uri: Optional[str] = None, 
                 username: Optional[str] = None, 
                 password: Optional[str] = None,
                 state_manager: Optional[AdaptiveStateManager] = None):
        """Initialize the Neo4j manager.
        
        Args:
            uri: The Neo4j connection URI.
            username: The Neo4j username.
            password: The Neo4j password.
            state_manager: The state manager to use for storing database information.
        """
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.environ.get("NEO4J_USERNAME", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "password")
        self.state_manager = state_manager or AdaptiveStateManager()
        self._driver = None
    
    def connect(self) -> bool:
        """Connect to the Neo4j database.
        
        Returns:
            True if the connection was successful, False otherwise.
        """
        try:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # Verify connection by running a simple query
            with self._driver.session() as session:
                result = session.run("RETURN 1 AS test")
                record = result.single()
                if record and record["test"] == 1:
                    logger.info(f"Successfully connected to Neo4j at {self.uri}")
                    return True
                else:
                    logger.error("Failed to verify Neo4j connection")
                    return False
        except Exception as e:
            logger.error(f"Error connecting to Neo4j: {str(e)}")
            return False
    
    def close(self) -> None:
        """Close the Neo4j connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Closed Neo4j connection")
    
    def get_driver(self) -> Optional[Driver]:
        """Get the Neo4j driver.
        
        Returns:
            The Neo4j driver if connected, None otherwise.
        """
        if not self._driver:
            self.connect()
        return self._driver
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query.
        
        Args:
            query: The Cypher query to execute.
            parameters: The query parameters.
            
        Returns:
            A list of records as dictionaries.
        """
        logger.info(f"Executing Cypher query: {query}")
        
        if not self._driver:
            if not self.connect():
                return [{"error": "Failed to connect to Neo4j"}]
        
        try:
            with self._driver.session() as session:
                result = session.run(query, parameters or {})
                records = [dict(record) for record in result]
                logger.info(f"Query returned {len(records)} records")
                return records
        except Exception as e:
            error_msg = f"Error executing Cypher query: {str(e)}"
            logger.error(error_msg)
            return [{"error": error_msg}]
    
    def create_constraints_and_indexes(self) -> Dict[str, Any]:
        """Create constraints and indexes for the knowledge graph.
        
        Returns:
            A dictionary with the results of the operation.
        """
        logger.info("Creating constraints and indexes for the knowledge graph")
        
        if not self._driver:
            if not self.connect():
                return {"success": False, "error": "Failed to connect to Neo4j"}
        
        try:
            constraints = [
                # Unique constraints
                "CREATE CONSTRAINT paper_doi_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.doi IS UNIQUE",
                "CREATE CONSTRAINT author_id_unique IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE",
                "CREATE CONSTRAINT institution_id_unique IF NOT EXISTS FOR (i:Institution) REQUIRE i.id IS UNIQUE",
                "CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
                
                # Node property existence constraints
                "CREATE CONSTRAINT paper_title_exists IF NOT EXISTS FOR (p:Paper) REQUIRE p.title IS NOT NULL",
                "CREATE CONSTRAINT author_name_exists IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS NOT NULL",
            ]
            
            indexes = [
                # Indexes for faster lookups
                "CREATE INDEX paper_title_index IF NOT EXISTS FOR (p:Paper) ON (p.title)",
                "CREATE INDEX paper_year_index IF NOT EXISTS FOR (p:Paper) ON (p.year)",
                "CREATE INDEX author_name_index IF NOT EXISTS FOR (a:Author) ON (a.name)",
                "CREATE INDEX concept_name_index IF NOT EXISTS FOR (c:Concept) ON (c.name)",
            ]
            
            with self._driver.session() as session:
                for constraint in constraints:
                    session.run(constraint)
                
                for index in indexes:
                    session.run(index)
            
            return {
                "success": True,
                "constraints": len(constraints),
                "indexes": len(indexes),
                "message": "Successfully created constraints and indexes",
            }
        except Exception as e:
            error_msg = f"Error creating constraints and indexes: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def add_paper(self, paper_data: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """Add a paper to the knowledge graph.
        
        Args:
            paper_data: The paper data to add.
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A dictionary with the results of the operation.
        """
        logger.info(f"Adding paper to knowledge graph: {paper_data.get('title', 'Unknown Title')}")
        
        if not self._driver:
            if not self.connect():
                return {"success": False, "error": "Failed to connect to Neo4j"}
        
        try:
            # Extract paper properties
            title = paper_data.get('title', 'Unknown Title')
            doi = paper_data.get('doi', '')
            abstract = paper_data.get('abstract', '')
            year = paper_data.get('year') or paper_data.get('publication_date', '')
            if isinstance(year, str) and len(year) >= 4:
                year = year[:4]  # Extract year from date string
            
            # Extract authors
            authors = paper_data.get('authors', [])
            if isinstance(authors, str):
                # Split author string if it's not already a list
                authors = [author.strip() for author in authors.split(',') if author.strip()]
            
            # Extract keywords/concepts
            keywords = paper_data.get('keywords', [])
            if isinstance(keywords, str):
                # Split keywords string if it's not already a list
                keywords = [keyword.strip() for keyword in keywords.split(',') if keyword.strip()]
            
            # Create paper node
            query = """
            MERGE (p:Paper {doi: $doi})
            ON CREATE SET 
                p.title = $title,
                p.abstract = $abstract,
                p.year = $year,
                p.added = datetime(),
                p.updated = datetime()
            ON MATCH SET 
                p.title = $title,
                p.abstract = $abstract,
                p.year = $year,
                p.updated = datetime()
            RETURN p.doi as doi, p.title as title, id(p) as node_id
            """
            
            with self._driver.session() as session:
                # Create paper node
                result = session.run(query, {
                    "doi": doi,
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                })
                paper_record = result.single()
                
                if not paper_record:
                    return {"success": False, "error": "Failed to create paper node"}
                
                paper_node_id = paper_record["node_id"]
                
                # Create author nodes and relationships
                for author_name in authors:
                    author_query = """
                    MERGE (a:Author {name: $name})
                    ON CREATE SET a.added = datetime()
                    WITH a
                    MATCH (p:Paper) WHERE id(p) = $paper_id
                    MERGE (a)-[:AUTHORED]->(p)
                    RETURN a.name as name, id(a) as node_id
                    """
                    
                    author_result = session.run(author_query, {
                        "name": author_name,
                        "paper_id": paper_node_id,
                    })
                
                # Create concept nodes and relationships
                for keyword in keywords:
                    concept_query = """
                    MERGE (c:Concept {name: $name})
                    ON CREATE SET c.added = datetime()
                    WITH c
                    MATCH (p:Paper) WHERE id(p) = $paper_id
                    MERGE (p)-[:HAS_CONCEPT]->(c)
                    RETURN c.name as name, id(c) as node_id
                    """
                    
                    concept_result = session.run(concept_query, {
                        "name": keyword,
                        "paper_id": paper_node_id,
                    })
            
            # Store the operation in the state if tool_context is provided
            if tool_context:
                # Get state manager from context if available
                state_manager = tool_context.state.get('state_manager', self.state_manager)
                
                # Get the current state
                current_state = state_manager.get(tool_context)
                
                # Initialize knowledge_graph in state if not present
                if 'knowledge_graph' not in current_state:
                    current_state['knowledge_graph'] = {
                        'papers': {},
                        'operations': [],
                    }
                
                # Add paper to state
                current_state['knowledge_graph']['papers'][doi or title] = {
                    'title': title,
                    'doi': doi,
                    'authors': authors,
                    'year': year,
                    'added': datetime.now().isoformat(),
                }
                
                # Add operation to state
                current_state['knowledge_graph']['operations'].append({
                    'operation': 'add_paper',
                    'paper': title,
                    'timestamp': datetime.now().isoformat(),
                })
                
                # Update the state
                state_manager.set(tool_context, current_state)
            
            return {
                "success": True,
                "paper": {
                    "doi": doi,
                    "title": title,
                    "node_id": paper_node_id,
                },
                "authors": len(authors),
                "concepts": len(keywords),
                "message": "Successfully added paper to knowledge graph",
            }
        except Exception as e:
            error_msg = f"Error adding paper to knowledge graph: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def add_citation(self, citing_paper: Dict[str, Any], cited_paper: Dict[str, Any], 
                    tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """Add a citation relationship between two papers.
        
        Args:
            citing_paper: The paper that cites another paper.
            cited_paper: The paper that is cited.
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A dictionary with the results of the operation.
        """
        logger.info(f"Adding citation relationship: {citing_paper.get('title', 'Unknown')} -> {cited_paper.get('title', 'Unknown')}")
        
        if not self._driver:
            if not self.connect():
                return {"success": False, "error": "Failed to connect to Neo4j"}
        
        try:
            # Extract paper identifiers
            citing_doi = citing_paper.get('doi', '')
            citing_title = citing_paper.get('title', 'Unknown Title')
            cited_doi = cited_paper.get('doi', '')
            cited_title = cited_paper.get('title', 'Unknown Title')
            
            # Create citation relationship
            query = """
            MATCH (citing:Paper), (cited:Paper)
            WHERE 
                (citing.doi = $citing_doi OR citing.title = $citing_title) AND
                (cited.doi = $cited_doi OR cited.title = $cited_title)
            MERGE (citing)-[r:CITES]->(cited)
            ON CREATE SET r.added = datetime()
            RETURN 
                citing.title as citing_title, 
                cited.title as cited_title, 
                id(r) as relationship_id
            """
            
            with self._driver.session() as session:
                result = session.run(query, {
                    "citing_doi": citing_doi,
                    "citing_title": citing_title,
                    "cited_doi": cited_doi,
                    "cited_title": cited_title,
                })
                
                record = result.single()
                if not record:
                    # Papers might not exist, try to create them first
                    self.add_paper(citing_paper, tool_context)
                    self.add_paper(cited_paper, tool_context)
                    
                    # Try again to create the citation relationship
                    result = session.run(query, {
                        "citing_doi": citing_doi,
                        "citing_title": citing_title,
                        "cited_doi": cited_doi,
                        "cited_title": cited_title,
                    })
                    
                    record = result.single()
                    if not record:
                        return {"success": False, "error": "Failed to create citation relationship"}
            
            # Store the operation in the state if tool_context is provided
            if tool_context:
                # Get state manager from context if available
                state_manager = tool_context.state.get('state_manager', self.state_manager)
                
                # Get the current state
                current_state = state_manager.get(tool_context)
                
                # Initialize knowledge_graph in state if not present
                if 'knowledge_graph' not in current_state:
                    current_state['knowledge_graph'] = {
                        'papers': {},
                        'operations': [],
                    }
                
                # Add operation to state
                current_state['knowledge_graph']['operations'].append({
                    'operation': 'add_citation',
                    'citing_paper': citing_title,
                    'cited_paper': cited_title,
                    'timestamp': datetime.now().isoformat(),
                })
                
                # Update the state
                state_manager.set(tool_context, current_state)
            
            return {
                "success": True,
                "citing_paper": record["citing_title"],
                "cited_paper": record["cited_title"],
                "relationship_id": record["relationship_id"],
                "message": "Successfully added citation relationship",
            }
        except Exception as e:
            error_msg = f"Error adding citation relationship: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def execute_cypher(self, cypher: str, parameters: Optional[Dict[str, Any]] = None,
                      tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """Execute a custom Cypher query.
        
        Args:
            cypher: The Cypher query to execute.
            parameters: The query parameters.
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A dictionary with the results of the operation.
        """
        logger.info(f"Executing custom Cypher query: {cypher}")
        
        if not self._driver:
            if not self.connect():
                return {"success": False, "error": "Failed to connect to Neo4j"}
        
        try:
            records = self.execute_query(cypher, parameters)
            
            # Store the operation in the state if tool_context is provided
            if tool_context:
                # Get state manager from context if available
                state_manager = tool_context.state.get('state_manager', self.state_manager)
                
                # Get the current state
                current_state = state_manager.get(tool_context)
                
                # Initialize knowledge_graph in state if not present
                if 'knowledge_graph' not in current_state:
                    current_state['knowledge_graph'] = {
                        'papers': {},
                        'operations': [],
                    }
                
                # Add operation to state
                current_state['knowledge_graph']['operations'].append({
                    'operation': 'execute_cypher',
                    'cypher': cypher,
                    'parameters': parameters,
                    'timestamp': datetime.now().isoformat(),
                })
                
                # Update the state
                state_manager.set(tool_context, current_state)
            
            return {
                "success": True,
                "records": records,
                "count": len(records),
                "message": f"Successfully executed Cypher query with {len(records)} results",
            }
        except Exception as e:
            error_msg = f"Error executing Cypher query: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def get_graph_statistics(self, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """Get statistics about the knowledge graph.
        
        Args:
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A dictionary with graph statistics.
        """
        logger.info("Getting knowledge graph statistics")
        
        if not self._driver:
            if not self.connect():
                return {"success": False, "error": "Failed to connect to Neo4j"}
        
        try:
            # Query for node counts by label
            node_query = """
            MATCH (n)
            RETURN labels(n) as label, count(n) as count
            """
            
            # Query for relationship counts by type
            rel_query = """
            MATCH ()-[r]->() 
            RETURN type(r) as type, count(r) as count
            """
            
            with self._driver.session() as session:
                # Get node counts
                node_result = session.run(node_query)
                node_counts = {}
                for record in node_result:
                    label = record["label"][0] if record["label"] else "Unknown"
                    node_counts[label] = record["count"]
                
                # Get relationship counts
                rel_result = session.run(rel_query)
                rel_counts = {}
                for record in rel_result:
                    rel_counts[record["type"]] = record["count"]
            
            # Calculate total counts
            total_nodes = sum(node_counts.values())
            total_relationships = sum(rel_counts.values())
            
            # Get recent papers
            recent_query = """
            MATCH (p:Paper)
            RETURN p.title as title, p.doi as doi, p.year as year
            ORDER BY p.added DESC
            LIMIT 5
            """
            
            with self._driver.session() as session:
                recent_result = session.run(recent_query)
                recent_papers = [dict(record) for record in recent_result]
            
            statistics = {
                "success": True,
                "node_counts": node_counts,
                "relationship_counts": rel_counts,
                "total_nodes": total_nodes,
                "total_relationships": total_relationships,
                "recent_papers": recent_papers,
                "timestamp": datetime.now().isoformat(),
            }
            
            # Store the statistics in the state if tool_context is provided
            if tool_context:
                # Get state manager from context if available
                state_manager = tool_context.state.get('state_manager', self.state_manager)
                
                # Get the current state
                current_state = state_manager.get(tool_context)
                
                # Initialize knowledge_graph in state if not present
                if 'knowledge_graph' not in current_state:
                    current_state['knowledge_graph'] = {}
                
                # Add statistics to state
                current_state['knowledge_graph']['statistics'] = statistics
                
                # Update the state
                state_manager.set(tool_context, current_state)
            
            return statistics
        except Exception as e:
            error_msg = f"Error getting graph statistics: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
