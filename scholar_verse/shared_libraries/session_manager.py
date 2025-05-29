"""Session management for ScholarVerse using Google ADK."""

from typing import Dict, Any, Optional
import logging
from google.adk.sessions import VertexAiSessionService, Session
from google.adk.memory import VertexAiRagMemoryService
from google.api_core.exceptions import GoogleAPIError

from scholar_verse.config import (
    GOOGLE_CLOUD_PROJECT,
    GOOGLE_CLOUD_LOCATION,
    RAG_CORPUS
)

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages session and memory services for ScholarVerse."""
    
    def __init__(self):
        """Initialize session and memory services."""
        self.session_service = self._init_session_service()
        self.memory_service = self._init_memory_service()
    
    def _init_session_service(self) -> VertexAiSessionService:
        """Initialize the Vertex AI session service."""
        try:
            return VertexAiSessionService(
                project=GOOGLE_CLOUD_PROJECT,
                location=GOOGLE_CLOUD_LOCATION
            )
        except GoogleAPIError as e:
            logger.error(f"Failed to initialize Vertex AI session service: {e}")
            raise
    
    def _init_memory_service(self) -> VertexAiRagMemoryService:
        """Initialize the Vertex AI RAG memory service."""
        try:
            return VertexAiRagMemoryService(
                rag_corpus=RAG_CORPUS,
                similarity_top_k=5,
                vector_distance_threshold=0.7
            )
        except GoogleAPIError as e:
            logger.error(f"Failed to initialize Vertex AI RAG memory service: {e}")
            raise
    
    async def create_session(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        initial_state: Optional[Dict[str, Any]] = None
    ) -> Session:
        """Create a new session with the given user and initial state.
        
        Args:
            user_id: The ID of the user.
            session_id: Optional custom session ID. If not provided, one will be generated.
            initial_state: Optional initial state for the session.
            
        Returns:
            The created Session object.
        """
        try:
            return await self.session_service.create_session(
                app_name="scholarverse",
                user_id=user_id,
                session_id=session_id,
                state=initial_state or {}
            )
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    async def get_session(self, user_id: str, session_id: str) -> Optional[Session]:
        """Retrieve an existing session.
        
        Args:
            user_id: The ID of the user.
            session_id: The ID of the session to retrieve.
            
        Returns:
            The Session object if found, None otherwise.
        """
        try:
            return await self.session_service.get_session(
                app_name="scholarverse",
                user_id=user_id,
                session_id=session_id
            )
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    async def update_session_state(
        self,
        session: Session,
        state_updates: Dict[str, Any],
        merge: bool = True
    ) -> Session:
        """Update the state of a session.
        
        Args:
            session: The session to update.
            state_updates: Dictionary of state updates.
            merge: If True, merge updates with existing state. If False, replace state.
            
        Returns:
            The updated Session object.
        """
        try:
            if merge:
                new_state = {**session.state, **state_updates}
            else:
                new_state = state_updates
                
            session.state = new_state
            return await self.session_service.update_session(session)
        except Exception as e:
            logger.error(f"Failed to update session state: {e}")
            raise
    
    async def add_to_memory(self, session: Session, content: str, metadata: Optional[Dict] = None) -> str:
        """Add content to the long-term memory.
        
        Args:
            session: The session associated with the content.
            content: The content to store.
            metadata: Optional metadata for the content.
            
        Returns:
            The ID of the stored memory.
        """
        try:
            return await self.memory_service.add_session_to_memory(
                session=session,
                content=content,
                metadata=metadata or {}
            )
        except Exception as e:
            logger.error(f"Failed to add to memory: {e}")
            raise
    
    async def search_memory(
        self,
        query: str,
        user_id: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Search the long-term memory.
        
        Args:
            query: The search query.
            user_id: The ID of the user.
            top_k: Maximum number of results to return.
            
        Returns:
            Dictionary containing search results.
        """
        try:
            return await self.memory_service.search_memory(
                app_name="scholarverse",
                user_id=user_id,
                query=query,
                top_k=top_k
            )
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            raise
