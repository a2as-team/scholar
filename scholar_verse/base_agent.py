"""Base agent class for all ScholarVerse agents.

This module defines the base agent class that all sub-agents should inherit from.
It provides common functionality and ensures consistent behavior across all agents.
"""

from typing import Dict, Any, Optional, AsyncGenerator, Type, List, TypeVar, Generic, Union
import logging
from dataclasses import dataclass
from enum import Enum

from google.adk.agents.llm_agent import Agent as LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.tools import BaseTool, ToolContext
from google.adk.sessions import Session

from scholar_verse.config import DEFAULT_MODEL, logger
from scholar_verse.shared_libraries.session_manager import SessionManager
from scholar_verse.shared_libraries.state_manager import StateManager, StateScope
from scholar_verse.shared_libraries.memory_manager import MemoryManager, MemoryRecord
from datetime import UTC
# Type variable for state models
T = TypeVar('T')

class AgentRole(str, Enum):
    """Defines the role of an agent in the system."""
    ROUTER = "router"
    INGESTION = "ingestion"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    ANALYSIS = "analysis"
    SEARCH = "search"
    VISUALIZATION = "visualization"

@dataclass
class AgentContext:
    """Context object passed to agents during execution."""
    user_id: str
    session_id: str
    session: Optional[Session] = None
    state: Optional[StateManager] = None
    memory: Optional[MemoryManager] = None
    input_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}
        
        # Add timestamps if not present
        if "created_at" not in self.metadata:
            from datetime import datetime
            self.metadata["created_at"] = datetime.now(UTC).isoformat()
        
        # Ensure updated_at is set
        self.metadata["updated_at"] = datetime.now(UTC).isoformat()


class BaseAgent(LlmAgent, Generic[T]):
    """Base class for all ScholarVerse agents.
    
    This class provides common functionality and ensures consistent behavior
    across all agents in the ScholarVerse system. It integrates with the
    session, state, and memory management systems.
    """
    
    # Class-level configuration
    role: AgentRole = None  # Should be set by subclasses
    state_model: Type[T] = dict  # Default state model (can be overridden)
    
    def __init__(
        self,
        name: str,
        description: str,
        model: str = DEFAULT_MODEL,
        instruction: str = "",
        session_manager: Optional[SessionManager] = None,
        memory_manager: Optional[MemoryManager] = None,
        **kwargs
    ):
        """Initialize the base agent.
        
        Args:
            name: The name of the agent (must be unique).
            description: A brief description of the agent's purpose.
            model: The name of the language model to use.
            instruction: Instructions for the agent's behavior.
            session_manager: Optional session manager instance.
            memory_manager: Optional memory manager instance.
            **kwargs: Additional keyword arguments for the parent class.
        """
        super().__init__(
            name=name,
            description=description,
            model=model,
            instruction=instruction,
            **kwargs
        )
        
        self._tools_initialized = False
        self._required_tools: List[Type[BaseTool]] = []
        self.session_manager = session_manager
        self.memory_manager = memory_manager
        
        # Initialize logger with agent name
        self.logger = logger.getChild(f"agent.{self.name.lower()}")
    
    @property
    def role_name(self) -> str:
        """Get the role name of the agent."""
        return self.role.value if self.role else "unknown"
    
    async def initialize(self):
        """Initialize the agent and its tools asynchronously.
        
        This method should be called before the agent is used. It ensures
        that all required tools are properly initialized and the agent is
        ready to process requests.
        """
        if not self._tools_initialized:
            self.logger.info(f"Initializing agent: {self.name}")
            await self._initialize_tools()
            self._tools_initialized = True
            self.logger.info(f"Agent initialized: {self.name}")
    
    def create_agent_context(
        self,
        user_id: str,
        session_id: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentContext:
        """Create a context object for agent execution.
        
        Args:
            user_id: The ID of the user.
            session_id: The ID of the session.
            input_data: Optional input data for the agent.
            metadata: Optional metadata for the context.
            
        Returns:
            An initialized AgentContext object.
        """
        return AgentContext(
            user_id=user_id,
            session_id=session_id,
            input_data=input_data or {},
            metadata=metadata or {}
        )
    
    async def process(
        self,
        context: AgentContext,
        *args,
        **kwargs
    ) -> Any:
        """Process a request with the given context.
        
        This is the main entry point for agent processing. It handles:
        - Session management
        - State management
        - Memory management
        - Error handling
        - Logging
        
        Args:
            context: The agent context containing user/session info.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
            
        Returns:
            The result of the agent's processing.
            
        Raises:
            Exception: If processing fails.
        """
        try:
            # Initialize session and state if not provided
            if context.session is None and self.session_manager:
                context.session = await self.session_manager.get_session(
                    user_id=context.user_id,
                    session_id=context.session_id
                )
                
                if context.session is None:
                    self.logger.info(f"Creating new session: {context.session_id}")
                    context.session = await self.session_manager.create_session(
                        user_id=context.user_id,
                        session_id=context.session_id
                    )
            
            # Initialize state manager if not provided
            if context.state is None and context.session is not None:
                context.state = StateManager(
                    session_manager=self.session_manager,
                    user_id=context.user_id,
                    session_id=context.session_id
                )
                await context.state.initialize()
            
            # Initialize memory manager if not provided
            if context.memory is None and self.memory_manager is not None:
                context.memory = self.memory_manager
            
            # Update context with agent info
            context.metadata.update({
                "agent": self.name,
                "agent_role": self.role_name,
                "model": self.model,
            })
            
            # Call the agent-specific processing
            self.logger.info(f"Processing request for session: {context.session_id}")
            result = await self._process(context, *args, **kwargs)
            
            # Log successful processing
            self.logger.info(f"Successfully processed request for session: {context.session_id}")
            return result
            
        except Exception as e:
            self.logger.error(
                f"Error processing request for session {context.session_id}: {str(e)}",
                exc_info=True
            )
            raise
    
    async def _process(
        self,
        context: AgentContext,
        *args,
        **kwargs
    ) -> Any:
        """Agent-specific processing logic to be implemented by subclasses.
        
        Args:
            context: The agent context.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
            
        Returns:
            The result of the agent's processing.
            
        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement _process method")
    
    async def _initialize_tools(self):
        """Initialize tools required by this agent.
        
        This method should be overridden by subclasses to initialize
        any tools they require. The default implementation does nothing.
        """
        pass
    
    async def finalize(self):
        """Clean up resources used by the agent.
        
        This method should be overridden by subclasses to perform any
        necessary cleanup when the agent is no longer needed.
        """
        self.logger.info(f"Finalizing agent: {self.name}")
        self._tools_initialized = False
        if self._tools_initialized:
            return
            
        try:
            # Initialize required tools
            self._register_tools()
            self._tools_initialized = True
            logger.info(f"Initialized agent: {self.name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent {self.name}: {str(e)}", exc_info=True)
            raise
    
    def _register_tools(self):
        """Register all tools with the agent.
        
        Subclasses should override this method to register their specific tools.
        """
        pass
    
    async def _run_async_impl(
        self, 
        context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Core implementation of the agent's async execution.
        
        Args:
            context: The invocation context containing the request and state.
            
        Yields:
            Events generated during execution.
        """
        try:
            # Ensure agent is initialized
            await self.initialize()
            
            # Get user input from context
            user_input = getattr(context, 'user_input', '')
            if not user_input:
                yield Event(content=Content.from_text("No input provided"))
                return
                
            logger.info(f"Processing request in {self.name}: {user_input}")
            
            # Process the request (to be implemented by subclasses)
            response = await self._process_request(user_input, context)
            
            yield Event(content=Content.from_text(response))
            
        except Exception as e:
            logger.error(f"Error in {self.name}: {str(e)}", exc_info=True)
            yield Event(content=Content.from_text(
                f"An error occurred while processing your request: {str(e)}"
            ))
    
    async def _process_request(
        self, 
        request: str, 
        context: InvocationContext
    ) -> str:
        """Process a user request.
        
        Subclasses must implement this method to handle specific agent logic.
        
        Args:
            request: The user's request text.
            context: The invocation context.
            
        Returns:
            The agent's response as a string.
        """
        raise NotImplementedError("Subclasses must implement _process_request")
    
    def _get_tool_context(self, context: InvocationContext) -> ToolContext:
        """Get or create a tool context from an invocation context.
        
        Args:
            context: The invocation context.
            
        Returns:
            A ToolContext instance with the state from the invocation context.
        """
        tool_context = ToolContext()
        tool_context.state = getattr(context, 'state', {})
        return tool_context
