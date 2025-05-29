"""Qdrant vector database connector for ScholarVerse."""

from typing import Dict, Any, List, Optional, Union
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

from scholar_verse.config import QDRANT_HOST, QDRANT_PORT
from scholar_verse.shared_libraries.logging_utils import logger


class QdrantConnector:
    """Qdrant vector database connector for ScholarVerse.
    
    This class provides a connection to the Qdrant vector database and methods for
    storing and retrieving vector embeddings. It will be expanded in later phases
    to include more sophisticated vector search operations.
    """
    
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        """Initialize the Qdrant connector.
        
        Args:
            host: The Qdrant host, or None to use the value from config.
            port: The Qdrant port, or None to use the value from config.
        """
        self.host = host or QDRANT_HOST
        self.port = port or QDRANT_PORT
        self.client = None
        logger.info("Initialized Qdrant connector")
    
    def connect(self) -> bool:
        """Connect to the Qdrant database.
        
        Returns:
            True if the connection was successful, False otherwise.
        """
        try:
            logger.info(f"Connecting to Qdrant database at {self.host}:{self.port}")
            self.client = QdrantClient(host=self.host, port=self.port)
            # Verify the connection by getting the list of collections
            self.client.get_collections()
            logger.info("Connected to Qdrant database successfully")
            return True
        except Exception as e:
            logger.error(f"Error connecting to Qdrant database: {e}")
            return False
    
    def close(self) -> None:
        """Close the connection to the Qdrant database."""
        if self.client:
            # Qdrant client doesn't have a close method, but we'll set it to None
            self.client = None
            logger.info("Closed Qdrant database connection")
    
    def create_collection(self, collection_name: str, vector_size: int = 768) -> bool:
        """Create a collection in the Qdrant database.
        
        Args:
            collection_name: The name of the collection to create.
            vector_size: The size of the vectors to store in the collection.
            
        Returns:
            True if the collection was created successfully, False otherwise.
        """
        if not self.client:
            if not self.connect():
                return False
        
        try:
            logger.info(f"Creating Qdrant collection: {collection_name}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error creating Qdrant collection: {e}")
            return False
    
    def insert_vectors(self, collection_name: str, vectors: List[List[float]], metadata: List[Dict[str, Any]], ids: Optional[List[Union[str, int]]] = None) -> bool:
        """Insert vectors into a Qdrant collection.
        
        Args:
            collection_name: The name of the collection to insert into.
            vectors: The vectors to insert.
            metadata: The metadata to associate with each vector.
            ids: The IDs to assign to each vector, or None to use auto-generated IDs.
            
        Returns:
            True if the vectors were inserted successfully, False otherwise.
        """
        if not self.client:
            if not self.connect():
                return False
        
        try:
            logger.info(f"Inserting {len(vectors)} vectors into Qdrant collection: {collection_name}")
            
            # Convert vectors to numpy arrays if they aren't already
            vectors_np = [np.array(v, dtype=np.float32) for v in vectors]
            
            # Create points
            points = [
                models.PointStruct(
                    id=ids[i] if ids else i,
                    vector=vectors_np[i].tolist(),
                    payload=metadata[i],
                )
                for i in range(len(vectors))
            ]
            
            # Insert points
            self.client.upsert(
                collection_name=collection_name,
                points=points,
            )
            
            logger.info(f"Inserted {len(vectors)} vectors into Qdrant collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error inserting vectors into Qdrant collection: {e}")
            return False
    
    def search_vectors(self, collection_name: str, query_vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors in a Qdrant collection.
        
        Args:
            collection_name: The name of the collection to search in.
            query_vector: The query vector to search for.
            limit: The maximum number of results to return.
            
        Returns:
            A list of search results, each containing the vector ID, score, and metadata.
        """
        if not self.client:
            if not self.connect():
                return []
        
        try:
            logger.info(f"Searching for similar vectors in Qdrant collection: {collection_name}")
            
            # Convert query vector to numpy array if it isn't already
            query_vector_np = np.array(query_vector, dtype=np.float32)
            
            # Search for similar vectors
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector_np.tolist(),
                limit=limit,
            )
            
            # Format the results
            results = [
                {
                    "id": str(result.id),
                    "score": float(result.score),
                    "metadata": result.payload,
                }
                for result in search_result
            ]
            
            logger.info(f"Found {len(results)} similar vectors in Qdrant collection: {collection_name}")
            return results
        except Exception as e:
            logger.error(f"Error searching for vectors in Qdrant collection: {e}")
            return []


# Create a default Qdrant connector
qdrant_connector = QdrantConnector()
