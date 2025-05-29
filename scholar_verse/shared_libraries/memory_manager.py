"""Memory management for ScholarVerse using Google ADK's RAG capabilities."""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import json
import logging
from datetime import datetime
from google.api_core.exceptions import GoogleAPIError

from scholar_verse.config import (
    GOOGLE_CLOUD_PROJECT,
    GOOGLE_CLOUD_LOCATION,
    RAG_CORPUS,
    logger
)

# Try to import Vertex AI RAG service
try:
    from google.adk.memory.vertex_ai_rag import VertexAiRagMemoryService
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    logger.warning("Vertex AI RAG service not available. Using in-memory fallback.")

# Fallback in-memory memory service
from google.adk.memory import InMemoryMemoryService

@dataclass
class MemoryRecord:
    """Represents a memory record in the knowledge base."""
    id: str
    content: str
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str

class MemoryManager:
    """Manages long-term memory using Google ADK's RAG capabilities."""
    
    def __init__(self, session_manager=None):
        """Initialize the memory manager.
        
        Args:
            session_manager: Optional session manager for session-related operations.
        """
        self.session_manager = session_manager
        self.memory_service = self._init_memory_service()
    
    def _init_memory_service(self):
        """Initialize the appropriate memory service based on availability."""
        if VERTEX_AI_AVAILABLE and RAG_CORPUS:
            try:
                return VertexAiRagMemoryService(
                    project=GOOGLE_CLOUD_PROJECT,
                    location=GOOGLE_CLOUD_LOCATION,
                    rag_corpus=RAG_CORPUS,
                    similarity_top_k=5,
                    vector_distance_threshold=0.7
                )
            except GoogleAPIError as e:
                logger.error(f"Failed to initialize Vertex AI RAG service: {e}")
                logger.warning("Falling back to in-memory memory service")
        
        # Fallback to in-memory service
        logger.warning("Using in-memory memory service. Data will not persist between restarts.")
        return InMemoryMemoryService()
    
    async def add_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> str:
        """Add a memory to the knowledge base.
        
        Args:
            content: The content to store in memory.
            metadata: Optional metadata associated with the content.
            session_id: Optional session ID associated with this memory.
            user_id: Optional user ID associated with this memory.
            
        Returns:
            The ID of the created memory.
        """
        try:
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            # Add system metadata
            metadata.update({
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            })
            
            if session_id:
                metadata["session_id"] = session_id
            if user_id:
                metadata["user_id"] = user_id
            
            # Add to memory service
            memory_id = await self.memory_service.add_session_to_memory(
                content=content,
                metadata=metadata
            )
            
            logger.debug(f"Added memory with ID: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise
    
    async def search_memories(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        top_k: int = 5,
        min_relevance: float = 0.5
    ) -> List[MemoryRecord]:
        """Search for relevant memories.
        
        Args:
            query: The search query.
            user_id: Optional user ID to filter results.
            session_id: Optional session ID to filter results.
            top_k: Maximum number of results to return.
            min_relevance: Minimum relevance score (0-1) for results.
            
        Returns:
            List of relevant memory records sorted by relevance.
        """
        try:
            # Prepare filters
            filters = {}
            if user_id:
                filters["user_id"] = user_id
            if session_id:
                filters["session_id"] = session_id
            
            # Execute search
            results = await self.memory_service.search_memory(
                query=query,
                top_k=top_k,
                filters=filters,
                min_relevance=min_relevance
            )
            
            # Convert to MemoryRecord objects
            memories = []
            for result in results.get("results", []):
                try:
                    metadata = result.get("metadata", {})
                    memory = MemoryRecord(
                        id=result.get("id", ""),
                        content=result.get("content", ""),
                        metadata=metadata,
                        created_at=metadata.get("created_at", ""),
                        updated_at=metadata.get("updated_at", "")
                    )
                    memories.append(memory)
                except Exception as e:
                    logger.warning(f"Failed to parse memory result: {e}")
            
            return memories
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []
    
    async def get_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        """Retrieve a specific memory by ID.
        
        Args:
            memory_id: The ID of the memory to retrieve.
            
        Returns:
            The memory record if found, None otherwise.
        """
        try:
            # Note: This assumes the memory service implements get_document
            if hasattr(self.memory_service, 'get_document'):
                doc = await self.memory_service.get_document(memory_id)
                if doc:
                    metadata = doc.get("metadata", {})
                    return MemoryRecord(
                        id=memory_id,
                        content=doc.get("content", ""),
                        metadata=metadata,
                        created_at=metadata.get("created_at", ""),
                        updated_at=metadata.get("updated_at", "")
                    )
            return None
        except Exception as e:
            logger.error(f"Failed to get memory {memory_id}: {e}")
            return None
    
    async def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata_updates: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing memory.
        
        Args:
            memory_id: The ID of the memory to update.
            content: New content for the memory (optional).
            metadata_updates: Metadata fields to update (optional).
            
        Returns:
            True if the update was successful, False otherwise.
        """
        try:
            # Note: This is a simplified implementation. In a real RAG system,
            # you would need to handle updates according to the specific API.
            # This is a placeholder that would need to be adapted.
            
            # For in-memory service, we can't update directly, so we'll remove and re-add
            if isinstance(self.memory_service, InMemoryMemoryService):
                # This is a simplified approach - in reality, you'd need to handle the update
                # according to the specific memory service implementation
                logger.warning("Update not fully supported for in-memory service")
                return False
                
            # For Vertex AI RAG, we'd need to implement the update logic here
            # This is a placeholder that would need to be implemented
            logger.warning("Update operation needs to be implemented for the current memory service")
            return False
            
        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID.
        
        Args:
            memory_id: The ID of the memory to delete.
            
        Returns:
            True if the memory was deleted, False otherwise.
        """
        try:
            # Note: This assumes the memory service implements delete_document
            if hasattr(self.memory_service, 'delete_document'):
                return await self.memory_service.delete_document(memory_id)
            return False
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    async def list_memories(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[MemoryRecord]:
        """List memories with optional filtering.
        
        Args:
            user_id: Optional user ID to filter by.
            session_id: Optional session ID to filter by.
            limit: Maximum number of results to return.
            offset: Offset for pagination.
            
        Returns:
            List of memory records matching the criteria.
        """
        try:
            # Note: This is a simplified implementation. In a real RAG system,
            # you would need to implement proper listing with filters.
            # This is a placeholder that would need to be adapted.
            
            # For in-memory service, we can list all memories
            if isinstance(self.memory_service, InMemoryMemoryService):
                # This is a simplified approach - in reality, you'd need to handle the listing
                # according to the specific memory service implementation
                logger.warning("List operation not fully supported for in-memory service")
                return []
                
            # For Vertex AI RAG, we'd need to implement the list logic here
            # This is a placeholder that would need to be implemented
            logger.warning("List operation needs to be implemented for the current memory service")
            return []
            
        except Exception as e:
            logger.error(f"Failed to list memories: {e}")
            return []
