"""Neo4j database connector for ScholarVerse."""

from typing import Dict, Any, List, Optional
from neo4j import GraphDatabase

from scholar_verse.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from scholar_verse.shared_libraries.logging_utils import logger


class Neo4jConnector:
    """Neo4j database connector for ScholarVerse.
    
    This class provides a connection to the Neo4j graph database and methods for
    querying and manipulating the knowledge graph. It will be expanded in Phase 4
    to include more sophisticated graph operations.
    """
    
    def __init__(self, uri: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None):
        """Initialize the Neo4j connector.
        
        Args:
            uri: The Neo4j URI, or None to use the value from config.
            username: The Neo4j username, or None to use the value from config.
            password: The Neo4j password, or None to use the value from config.
        """
        self.uri = uri or NEO4J_URI
        self.username = username or NEO4J_USERNAME
        self.password = password or NEO4J_PASSWORD
        self.driver = None
        logger.info("Initialized Neo4j connector")
    
    def connect(self) -> bool:
        """Connect to the Neo4j database.
        
        Returns:
            True if the connection was successful, False otherwise.
        """
        try:
            logger.info(f"Connecting to Neo4j database at {self.uri}")
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # Verify the connection
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j database successfully")
            return True
        except Exception as e:
            logger.error(f"Error connecting to Neo4j database: {e}")
            return False
    
    def close(self) -> None:
        """Close the connection to the Neo4j database."""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j database connection")
    
    def run_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run a Cypher query against the Neo4j database.
        
        Args:
            query: The Cypher query to run.
            parameters: The query parameters, or None if there are no parameters.
            
        Returns:
            A list of records returned by the query, or an empty list if there was an error.
        """
        if not self.driver:
            if not self.connect():
                return []
        
        try:
            logger.debug(f"Running Neo4j query: {query}")
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Error running Neo4j query: {e}")
            return []
    
    def create_paper_node(self, paper_data: Dict[str, Any]) -> Optional[str]:
        """Create a paper node in the knowledge graph.
        
        Args:
            paper_data: The paper data to create the node with.
            
        Returns:
            The ID of the created node, or None if there was an error.
        """
        query = """
        CREATE (p:Paper {
            title: $title,
            authors: $authors,
            year: $year,
            abstract: $abstract,
            doi: $doi,
            url: $url
        })
        RETURN id(p) AS node_id
        """
        
        parameters = {
            "title": paper_data.get("title", ""),
            "authors": paper_data.get("authors", []),
            "year": paper_data.get("year"),
            "abstract": paper_data.get("abstract", ""),
            "doi": paper_data.get("doi", ""),
            "url": paper_data.get("url", ""),
        }
        
        result = self.run_query(query, parameters)
        if result and "node_id" in result[0]:
            node_id = str(result[0]["node_id"])
            logger.info(f"Created paper node with ID: {node_id}")
            return node_id
        
        return None
    
    def create_citation_relationship(self, source_id: str, target_id: str, citation_data: Optional[Dict[str, Any]] = None) -> bool:
        """Create a citation relationship between two paper nodes.
        
        Args:
            source_id: The ID of the source paper node.
            target_id: The ID of the target paper node.
            citation_data: Additional data about the citation, or None.
            
        Returns:
            True if the relationship was created successfully, False otherwise.
        """
        query = """
        MATCH (source:Paper), (target:Paper)
        WHERE id(source) = $source_id AND id(target) = $target_id
        CREATE (source)-[r:CITES $citation_data]->(target)
        RETURN id(r) AS relationship_id
        """
        
        parameters = {
            "source_id": int(source_id),
            "target_id": int(target_id),
            "citation_data": citation_data or {},
        }
        
        result = self.run_query(query, parameters)
        if result and "relationship_id" in result[0]:
            relationship_id = str(result[0]["relationship_id"])
            logger.info(f"Created citation relationship with ID: {relationship_id}")
            return True
        
        return False


# Create a default Neo4j connector
neo4j_connector = Neo4jConnector()
