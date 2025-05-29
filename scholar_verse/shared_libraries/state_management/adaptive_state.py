"""Adaptive state management for ScholarVerse agents with ADK integration."""

from typing import Dict, Any, Optional, List, TypeVar, Generic, Type, cast
from datetime import datetime, timezone
import json
from dataclasses import dataclass, asdict, field
from enum import Enum
from google.adk.tools import ToolContext
from scholar_verse.shared_libraries.logging_utils import logger
from .redis_state_store import (
    RedisStateStore,
    VersionedState,
    StateVersionMismatchError
)

# Type variable for state data
T = TypeVar('T')

# Global state store instance
state_store = RedisStateStore(prefix="scholarverse:state:")

class StateKey(str, Enum):
    """Keys for different parts of the application state."""
    WORKFLOW = "workflow"
    DOCUMENTS = "documents"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    ANALYSIS = "analysis"
    USER_PREFERENCES = "user_preferences"
    FEEDBACK = "feedback"

class WorkflowStage(str, Enum):
    """Enumeration of possible workflow stages."""
    DOCUMENT_INGESTION = "document_ingestion"
    CITATION_GRAPH = "citation_graph"
    DEEP_SEARCH = "deep_search"
    CROSS_PAPER_ANALYSIS = "cross_paper_analysis"
    INSIGHT_GENERATION = "insight_generation"
    VISUALIZATION = "visualization"

class StateMetadata(VersionedState):
    """Metadata for the state with versioning support."""
    
    def __init__(self, **data):
        # Initialize with default values
        data.setdefault('version', '0.3.0')
        data.setdefault('created_at', datetime.now(timezone.utc))
        data.setdefault('updated_at', datetime.now(timezone.utc))
        data.setdefault('data', {})
        
        # Call parent's __init__ with the prepared data
        super().__init__(**data)
    
    @property
    def last_updated(self) -> str:
        """Alias for updated_at to maintain backward compatibility."""
        return self.updated_at.isoformat()
    
    @last_updated.setter
    def last_updated(self, value: str) -> None:
        """Update the last_updated timestamp."""
        self.updated_at = datetime.fromisoformat(value) if isinstance(value, str) else value

@dataclass
class WorkflowState:
    """State related to workflow management."""
    current_stage: Optional[WorkflowStage] = None
    stages_completed: List[WorkflowStage] = field(default_factory=list)
    next_possible_stages: List[WorkflowStage] = field(default_factory=list)

@dataclass
class DocumentState:
    """State related to document processing."""
    current_document: Optional[Dict[str, Any]] = None
    processed_documents: List[Dict[str, Any]] = field(default_factory=list)
    document_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KnowledgeGraphState:
    """State related to knowledge graph."""
    nodes: int = 0
    relationships: int = 0
    last_updated: Optional[str] = None

@dataclass
class AnalysisState:
    """State related to analysis results."""
    insights: List[Dict[str, Any]] = field(default_factory=list)
    trends: List[Dict[str, Any]] = field(default_factory=list)
    comparisons: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class UserPreferences:
    """User preferences for the system."""
    visualization_type: str = "3d"
    analysis_depth: str = "standard"
    citation_style: str = "apa"

@dataclass
class FeedbackState:
    """State related to feedback and performance."""
    agent_performance: Dict[str, Any] = field(default_factory=dict)
    user_feedback: List[Dict[str, Any]] = field(default_factory=list)

class AdaptiveStateManager:
    """Enhanced state manager for ScholarVerse agents with Redis persistence.
    
    This state manager provides type-safe access to application state with
    Redis-based persistence and versioning support.
    """
    
    def __init__(self, state_store_instance: Optional[RedisStateStore] = None):
        """Initialize the adaptive state manager with Redis persistence.
        
        Args:
            state_store_instance: Optional RedisStateStore instance to use
        """
        self.state_store = state_store_instance or state_store
        self._default_state = {
            "metadata": StateMetadata().dict(),
            "workflow": asdict(WorkflowState()),
            "documents": asdict(DocumentState()),
            "knowledge_graph": asdict(KnowledgeGraphState()),
            "analysis": asdict(AnalysisState()),
            "user_preferences": asdict(UserPreferences()),
            "feedback": asdict(FeedbackState()),
        }
        logger.info("Initialized Redis-backed adaptive state manager")
    
    async def get_state(self, context: ToolContext) -> Dict[str, Any]:
        """Get the current state from Redis or initialize if not present.
        
        Args:
            context: The tool context containing the state ID
            
        Returns:
            The current state dictionary
        """
        if not hasattr(context, 'state_id') or not context.state_id:
            # Generate a new state ID if none exists
            context.state_id = f"state_{datetime.now(timezone.utc).timestamp()}"
            await self.state_store.save_state(context.state_id, StateMetadata())
            return self._default_state
        
        # Try to load state from Redis
        state_metadata = await self.state_store.get_state(context.state_id, StateMetadata)
        if not state_metadata:
            # Initialize new state if not found
            state_metadata = StateMetadata()
            await self.state_store.save_state(context.state_id, state_metadata)
            return self._default_state
            
        # Load all state components
        state = {}
        for key in StateKey:
            state_data = await self.state_store.get_state(
                f"{context.state_id}:{key.value}",
                VersionedState
            )
            if state_data:
                state[key.value] = state_data.data
            else:
                # Initialize missing components with defaults
                state[key.value] = self._default_state[key.value]
                
        return state
        
    def _to_dict(self) -> Dict[str, Any]:
        """Convert the state to a dictionary.
        
        Returns:
            Dictionary representation of the state
        """
        return {
            'metadata': asdict(self._state['metadata']),
            'workflow': asdict(self._state['workflow']),
            'documents': asdict(self._state['documents']),
            'knowledge_graph': asdict(self._state['knowledge_graph']),
            'analysis': asdict(self._state['analysis']),
            'user_preferences': asdict(self._state['user_preferences']),
            'feedback': asdict(self._state['feedback'])
        }
    
    async def update_state(
        self,
        context: ToolContext,
        updates: Dict[str, Any]
    ) -> bool:
        """Update the state with the given values using atomic operations.
        
        Args:
            context: The tool context with state_id
            updates: Dictionary of state updates to apply
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        if not hasattr(context, 'state_id') or not context.state_id:
            logger.error("Cannot update state: No state_id in context")
            return False
            
        try:
            # Update each component separately for fine-grained control
            for key, value in updates.items():
                if key not in [e.value for e in StateKey]:
                    logger.warning(f"Unknown state component: {key}")
                    continue
                    
                # Update the component state
                success = await self.state_store.update_state(
                    f"{context.state_id}:{key}",
                    VersionedState,
                    lambda state, v: self._update_component(state, v),
                    value
                )
                
                if not success:
                    logger.error(f"Failed to update state component: {key}")
                    return False
                    
            # Update metadata timestamp
            await self.state_store.update_state(
                context.state_id,
                StateMetadata,
                lambda state, _: self._update_metadata(state)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating state: {e}", exc_info=True)
            return False
            
    def _update_component(
        self,
        state: VersionedState,
        updates: Dict[str, Any]
    ) -> VersionedState:
        """Update a single state component."""
        if not state.data:
            state.data = {}
            
        # Handle nested updates
        for key, value in updates.items():
            if isinstance(value, dict) and key in state.data and isinstance(state.data[key], dict):
                state.data[key].update(value)
            else:
                state.data[key] = value
                
        return state
        
    def _update_metadata(self, metadata: StateMetadata) -> StateMetadata:
        """Update metadata with current timestamp."""
        metadata.last_updated = datetime.now(timezone.utc).isoformat()
        return metadata
        
    async def update_workflow_stage(
        self,
        context: ToolContext,
        stage: WorkflowStage,
        is_complete: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update the current workflow stage with atomic operations.
        
        Args:
            context: The tool context with state_id
            stage: The new workflow stage
            is_complete: Whether the previous stage is complete
            metadata: Additional metadata to include
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            # Get current workflow state
            workflow_key = f"{context.state_id}:{StateKey.WORKFLOW.value}"
            workflow_state = await self.state_store.get_state(workflow_key, VersionedState)
            
            if not workflow_state:
                workflow_state = VersionedState(data=asdict(WorkflowState()))
            
            # Update workflow state
            workflow_data = workflow_state.data or {}
            current_stage = workflow_data.get('current_stage')
            completed_stages = set(workflow_data.get('stages_completed', []))
            
            # Update current stage
            workflow_data['current_stage'] = stage.value if isinstance(stage, WorkflowStage) else stage
            
            # Update completed stages if needed
            if is_complete and current_stage and current_stage not in completed_stages:
                completed_stages.add(current_stage)
                workflow_data['stages_completed'] = list(completed_stages)
            
            # Update next possible stages
            next_stages = self._get_next_possible_stages(stage)
            workflow_data['next_possible_stages'] = [s.value for s in next_stages]
            
            # Update metadata if provided
            if metadata:
                workflow_data.update(metadata)
            
            # Save the updated state
            workflow_state.data = workflow_data
            success = await self.state_store.save_state(workflow_key, workflow_state)
            
            if not success:
                logger.error("Failed to save workflow state")
                return False
                
            # Update metadata timestamp
            await self.state_store.update_state(
                context.state_id,
                StateMetadata,
                lambda state, _: self._update_metadata(state)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating workflow stage: {e}", exc_info=True)
            return False
        
    def _get_next_possible_stages(self, stage: WorkflowStage) -> List[WorkflowStage]:
        """Determine the next possible stages based on the current stage.
        
        Args:
            stage: The current workflow stage
            
        Returns:
            List of possible next stages
        """
        transitions = {
            WorkflowStage.DOCUMENT_INGESTION: [
                WorkflowStage.CITATION_GRAPH,
                WorkflowStage.DEEP_SEARCH
            ],
            WorkflowStage.CITATION_GRAPH: [
                WorkflowStage.CROSS_PAPER_ANALYSIS
            ],
            WorkflowStage.DEEP_SEARCH: [
                WorkflowStage.CROSS_PAPER_ANALYSIS
            ],
            WorkflowStage.CROSS_PAPER_ANALYSIS: [
                WorkflowStage.INSIGHT_GENERATION
            ],
            WorkflowStage.INSIGHT_GENERATION: [
                WorkflowStage.VISUALIZATION
            ],
            WorkflowStage.VISUALIZATION: []
        }
        return transitions.get(stage, [])
    
    async def update_document_state(
        self,
        context: ToolContext,
        updates: Dict[str, Any]
    ) -> bool:
        """Update the document state with atomic operations.
        
        Args:
            context: The tool context with state_id
            updates: Dictionary of document state updates
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            doc_key = f"{context.state_id}:{StateKey.DOCUMENTS.value}"
            
            # Define update function
            def update_docs(state: VersionedState, updates: Dict[str, Any]) -> VersionedState:
                if not state.data:
                    state.data = {}
                state.data.update(updates)
                return state
            
            # Update document state
            success = await self.state_store.update_state(
                doc_key,
                VersionedState,
                update_docs,
                updates
            )
            
            if not success:
                logger.error("Failed to update document state")
                return False
                
            # Update metadata timestamp
            await self.state_store.update_state(
                context.state_id,
                StateMetadata,
                lambda state, _: self._update_metadata(state)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating document state: {e}", exc_info=True)
            return False
    
    def get_document_state(self, context: ToolContext) -> DocumentState:
        """Get the current document state.
        
        Args:
            context: The tool context
            
        Returns:
            The current document state
        """
        state = self.get_state(context)
        return DocumentState(**state.get('documents', {}))
