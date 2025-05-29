"""Main router agent for ScholarVerse.

This module defines the main router agent that orchestrates the workflow between sub-agents
using the ScholarVerse BaseAgent as the base class.
"""
import asyncio
import json
import logging
import random
import time
from datetime import datetime, timezone, UTC
from enum import Enum, auto
from functools import wraps
from typing import Any, Dict, List, Optional, Type, TypeVar, Callable, Awaitable, Tuple

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.tools import BaseTool, ToolContext

from scholar_verse.base_agent import BaseAgent
from scholar_verse.config import DEFAULT_MODEL, get_config
from scholar_verse.shared_libraries.logging_utils import logger
from scholar_verse.shared_libraries.state_management.adaptive_state import (
    AdaptiveStateManager,
    WorkflowStage,
    StateKey
)

# Import sub-agents
from scholar_verse.sub_agents.ingestion.agent import IngestionAgent
from scholar_verse.sub_agents.citation_graph.agent import CitationGraphAgent
from scholar_verse.sub_agents.cross_paper_analysis.agent import CrossPaperAnalysisAgent
from scholar_verse.sub_agents.deep_search.agent import DeepSearchAgent
from scholar_verse.sub_agents.insight.agent import InsightAgent
from scholar_verse.sub_agents.visualization.agent import VisualizationAgent

# Import router agent instructions
from scholar_verse.prompt import ROUTER_AGENT_INSTRUCTIONS

# Type variable for generic function typing
T = TypeVar('T')

class CircuitState(Enum):
    """Possible states of the circuit breaker."""
    CLOSED = auto()    # Normal operation, all calls pass through
    OPEN = auto()      # Circuit is open, all calls fail fast
    HALF_OPEN = auto() # Test if the service has recovered

class CircuitBreakerError(Exception):
    """Exception raised when the circuit is open."""
    def __init__(self, circuit_name: str, state: CircuitState):
        self.circuit_name = circuit_name
        self.state = state
        super().__init__(f"Circuit '{circuit_name}' is {state.name}")

class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: int = 30,
        excluded_exceptions: Tuple[Type[Exception], ...] = ()
    ):
        """Initialize the circuit breaker.
        
        Args:
            name: Name of the circuit breaker for identification
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Time in seconds to wait before attempting recovery
            excluded_exceptions: Exceptions that should not trigger the circuit breaker
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.excluded_exceptions = excluded_exceptions
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
    
    @property
    def state(self) -> CircuitState:
        """Get the current state of the circuit."""
        # Check if we should transition from OPEN to HALF_OPEN
        if self._state == CircuitState.OPEN:
            if self._last_failure_time and \
               (datetime.now(timezone.utc) - self._last_failure_time).total_seconds() > self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info(f"Circuit '{self.name}' moved to HALF_OPEN state")
        return self._state
    
    def record_success(self) -> None:
        """Record a successful call and reset the circuit if needed."""
        if self._state == CircuitState.HALF_OPEN:
            self._reset()
            logger.info(f"Circuit '{self.name}' reset to CLOSED after successful call")
    
    def record_failure(self, error: Exception) -> None:
        """Record a failed call and update circuit state if needed."""
        # Skip excluded exceptions
        if any(isinstance(error, exc_type) for exc_type in self.excluded_exceptions):
            return
            
        self._failure_count += 1
        self._last_failure_time = datetime.now(timezone.utc)
        
        if self._state == CircuitState.HALF_OPEN:
            self._trip()
        elif self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold:
            self._trip()
    
    def _trip(self) -> None:
        """Trip the circuit to open state."""
        self._state = CircuitState.OPEN
        logger.warning(f"Circuit '{self.name}' tripped to OPEN state")
    
    def _reset(self) -> None:
        """Reset the circuit to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to wrap a function with circuit breaker logic."""
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            # Check circuit state
            if self.state == CircuitState.OPEN:
                raise CircuitBreakerError(self.name, self.state)
                
            try:
                result = await func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            # Check circuit state
            if self.state == CircuitState.OPEN:
                raise CircuitBreakerError(self.name, self.state)
                
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: float = 0.1,
    retry_on: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that retries a function with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Factor to multiply delay by after each retry
        jitter: Random jitter factor (0.0 to 1.0)
        retry_on: Tuple of exception types to retry on
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    if attempt == max_attempts:
                        break
                        
                    # Calculate next delay with jitter
                    delay = min(delay * backoff_factor, max_delay)
                    jitter_amount = delay * jitter * (2 * random.random() - 1)
                    sleep_time = max(0, delay + jitter_amount)
                    
                    logger.warning(
                        f"Attempt {attempt} failed: {str(e)}. "
                        f"Retrying in {sleep_time:.2f}s..."
                    )
                    
                    await asyncio.sleep(sleep_time)
            
            raise last_exception  # type: ignore
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    if attempt == max_attempts:
                        break
                        
                    # Calculate next delay with jitter
                    delay = min(delay * backoff_factor, max_delay)
                    jitter_amount = delay * jitter * (2 * random.random() - 1)
                    sleep_time = max(0, delay + jitter_amount)
                    
                    logger.warning(
                        f"Attempt {attempt} failed: {str(e)}. "
                        f"Retrying in {sleep_time:.2f}s..."
                    )
                    
                    time.sleep(sleep_time)
            
            raise last_exception  # type: ignore
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


class AgentCircuitBreaker:
    """Manages circuit breakers for agent calls."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the agent circuit breaker manager.
        
        Args:
            config: Configuration for circuit breakers
        """
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._config = config or {}
    
    def get_breaker(self, agent_name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Configured CircuitBreaker instance
        """
        if agent_name not in self._breakers:
            agent_config = self._config.get(agent_name, {})
            self._breakers[agent_name] = CircuitBreaker(
                name=f"agent_{agent_name}",
                failure_threshold=agent_config.get('failure_threshold', 3),
                recovery_timeout=agent_config.get('recovery_timeout', 30),
                excluded_exceptions=(ValueError, KeyError)  # Don't trip on input validation
            )
        return self._breakers[agent_name]
    
    def get_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get the status of all circuit breakers."""
        return {
            name: {
                'state': breaker.state.name,
                'failure_count': breaker._failure_count,
                'last_failure': breaker._last_failure_time.isoformat() if breaker._last_failure_time else None
            }
            for name, breaker in self._breakers.items()
        }


class RouterAgent(BaseAgent):
    """Main router agent that orchestrates workflow between sub-agents.
    
    This agent is responsible for:
    - Receiving user requests
    - Determining the appropriate sub-agent to handle each request
    - Coordinating between multiple sub-agents when needed
    - Managing the overall conversation state with circuit breaking and retry
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Router Agent with sub-agents and tools.
        
        Args:
            config: Configuration for the router agent
        """
        super().__init__(
            name="router_agent",
            description="Main router agent that coordinates between specialized sub-agents",
            model=DEFAULT_MODEL,
            instruction=ROUTER_AGENT_INSTRUCTIONS,
        )
        
        # Configuration
        self._config = config or {}
        
        # Initialize Redis-based state manager
        self.state_manager = AdaptiveStateManager()
        
        # Initialize circuit breakers
        self.circuit_breaker = AgentCircuitBreaker(
            self._config.get('circuit_breakers', {})
        )
        
        # Track if state has been loaded
        self._state_loaded = False
        
        # Initialize sub-agents
        self._sub_agents: Dict[str, BaseAgent] = {}
        self._tools_initialized = False
        self._agent_descriptions: Dict[str, str] = {}
        
        # Default retry configuration
        self.retry_config = {
            'max_attempts': self._config.get('max_retry_attempts', 3),
            'initial_delay': self._config.get('initial_retry_delay', 1.0),
            'max_delay': self._config.get('max_retry_delay', 30.0),
            'backoff_factor': self._config.get('retry_backoff_factor', 2.0),
            'jitter': self._config.get('retry_jitter', 0.1)
        }
    
    async def initialize(self):
        """Initialize the router agent and all sub-agents asynchronously."""
        if self._tools_initialized:
            return
            
        try:
            # Initialize sub-agents with circuit breakers
            self._sub_agents = {
                "ingestion": IngestionAgent(),
                "citation_graph": CitationGraphAgent(),
                "cross_paper_analysis": CrossPaperAnalysisAgent(),
                "deep_search": DeepSearchAgent(),
                "insight": InsightAgent(),
                "visualization": VisualizationAgent(),
            }
            
            # Store agent descriptions for routing
            self._agent_descriptions = {
                "ingestion": "Handles document ingestion and processing",
                "citation_graph": "Manages and analyzes citation networks",
                "cross_paper_analysis": "Performs analysis across multiple papers",
                "deep_search": "Conducts in-depth web and academic searches",
                "insight": "Generates insights from processed content",
                "visualization": "Creates visual representations of data",
            }
            
            # Initialize each sub-agent with error handling
            for name, agent in self._sub_agents.items():
                try:
                    if hasattr(agent, 'initialize') and callable(agent.initialize):
                        await agent.initialize()
                    logger.info(f"Initialized sub-agent: {name}")
                except Exception as e:
                    logger.error(f"Failed to initialize sub-agent {name}: {str(e)}", exc_info=True)
                    # Mark the circuit as open for this agent
                    self.circuit_breaker.get_breaker(name).record_failure(e)
            
            # Register tools
            await super().initialize()
            
        except Exception as e:
            logger.error(f"Failed to initialize router agent: {str(e)}", exc_info=True)
            raise
    
    def _register_tools(self):
        """Register all tools with the agent."""
        # Register route_to_agent tool with retry and circuit breaking
        route_to_agent_tool = BaseTool(
            name="route_to_agent",
            description="Route a request to a specific sub-agent with circuit breaking and retry"
        )
        
        # Register get_circuit_status tool
        get_circuit_status_tool = BaseTool(
            name="get_circuit_status",
            description="Get the status of all circuit breakers"
        )
        
        # Register reset_circuit tool
        reset_circuit_tool = BaseTool(
            name="reset_circuit",
            description="Reset a circuit breaker for a specific agent"
        )
        
        # Register the tools with the agent
        self.register_tool(route_to_agent_tool, self._route_to_agent_impl)
        self.register_tool(get_circuit_status_tool, self._get_circuit_status_impl)
        self.register_tool(reset_circuit_tool, self._reset_circuit_impl)
    
    @retry(max_attempts=3, initial_delay=1.0, max_delay=10.0)
    async def _route_to_agent_impl(
        self,
        context: ToolContext,
        agent_name: str,
        input_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Implementation of route_to_agent with circuit breaking and retry.
        
        Args:
            context: The tool context
            agent_name: Name of the agent to route to
            input_data: Input data for the agent
            **kwargs: Additional arguments
            
        Returns:
            Response from the agent
        """
        logger.info(f"Routing to agent: {agent_name}")
        
        # Get or create circuit breaker for this agent
        breaker = self.circuit_breaker.get_breaker(agent_name)
        
        # Check if circuit is open
        if breaker.state == CircuitState.OPEN:
            raise CircuitBreakerError(agent_name, CircuitState.OPEN)
        
        # Get the agent instance
        if agent_name not in self._sub_agents:
            return {
                "success": False,
                "error": f"Unknown agent: {agent_name}",
                "available_agents": list(self._sub_agents.keys())
            }
            
        agent = self._sub_agents[agent_name]
        
        # Define the function to call the agent with retry logic
        @retry(
            max_attempts=self.retry_config['max_attempts'],
            initial_delay=self.retry_config['initial_delay'],
            max_delay=self.retry_config['max_delay'],
            backoff_factor=self.retry_config['backoff_factor'],
            jitter=self.retry_config['jitter']
        )
        async def _call_agent():
            # Ensure state is loaded before processing
            if not self._state_loaded and hasattr(context, 'state_id'):
                await self.state_manager.get_state(context)
                self._state_loaded = True
                
            # Call the agent's process_request method
            response = await agent.process_request(
                input_data.get('input', ''),
                context
            )
            
            # Record success on the circuit breaker
            breaker.record_success()
            return response
        
        try:
            # Execute the agent call with retry logic
            result = await _call_agent()
            
            # Update workflow state if this was a successful agent call
            if isinstance(result, dict) and result.get('success', False):
                await self._update_workflow_after_agent_call(context, agent_name)
            
            return {
                "success": True,
                "agent": agent_name,
                "response": result
            }
            
        except Exception as e:
            # Record failure on the circuit breaker
            if not any(isinstance(e, exc) for exc in breaker.excluded_exceptions):
                breaker.record_failure(e)
                
            logger.error(f"Error in agent '{agent_name}': {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "agent": agent_name,
                "circuit_state": breaker.state.name
            }
        except Exception as e:
            logger.error(f"Error in agent {agent_name}: {str(e)}", exc_info=True)
            raise
    
    async def _update_workflow_after_agent_call(
        self,
        context: ToolContext,
        agent_name: str,
        result: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update the workflow state after a successful agent call.
        
        Args:
            context: The tool context with state_id
            agent_name: Name of the agent that was called
            result: Optional result from the agent call
        """
        try:
            # Map agent names to workflow stages and determine if the stage is complete
            agent_stage_map = {
                'ingestion': (WorkflowStage.DOCUMENT_INGESTION, True),
                'citation_graph': (WorkflowStage.CITATION_GRAPH, True),
                'deep_search': (WorkflowStage.DEEP_SEARCH, True),
                'cross_paper_analysis': (WorkflowStage.CROSS_PAPER_ANALYSIS, True),
                'insight': (WorkflowStage.INSIGHT_GENERATION, True),
                'visualization': (WorkflowStage.VISUALIZATION, True)
            }
            
            if agent_name not in agent_stage_map:
                logger.warning(f"No workflow stage mapping for agent: {agent_name}")
                return
                
            stage, is_complete = agent_stage_map[agent_name]
            
            # Prepare metadata for the workflow update
            metadata = {
                'last_agent': agent_name,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            if result and isinstance(result, dict):
                metadata.update({
                    'success': result.get('success', False),
                    'error': result.get('error'),
                    'agent_response': {k: v for k, v in result.items() 
                                     if k not in ['success', 'error', 'agent']}
                })
            
            # Update the workflow stage
            await self.update_workflow_stage(
                context=context,
                stage=stage,
                is_complete=is_complete,
                metadata=metadata
            )
            
            logger.info(f"Updated workflow stage to {stage} after {agent_name} agent call")
            
        except Exception as e:
            logger.error(f"Error updating workflow after agent call: {str(e)}", exc_info=True)
            # Don't raise the exception to avoid disrupting the agent call
    
    async def _get_circuit_status_impl(self, context: ToolContext) -> Dict[str, Any]:
        """Get the status of all circuit breakers.
        
        Args:
            context: The tool context
            
        Returns:
            Dictionary with circuit breaker statuses
        """
        return {
            'circuit_status': self.circuit_breaker.get_breaker_status(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def _reset_circuit_impl(
        self,
        context: ToolContext,
        agent_name: str
    ) -> Dict[str, Any]:
        """Reset a circuit breaker for a specific agent.
        
        Args:
            context: The tool context
            agent_name: Name of the agent whose circuit to reset
            
        Returns:
            Status of the reset operation
        """
        if agent_name not in self._sub_agents:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        # Get the circuit breaker
        breaker = self.circuit_breaker.get_breaker(agent_name)
        
        # Reset the circuit
        breaker._reset()
        
        return {
            'success': True,
            'agent': agent_name,
            'message': f"Circuit for {agent_name} has been reset"
        }
    
    async def run(self, context: ToolContext, **kwargs) -> Dict[str, Any]:
        """Main entry point for the router agent.
        
        Args:
            context: The tool context
            **kwargs: Additional arguments
            
        Returns:
            The agent's response
        """
        try:
            # Get user input from context
            user_input = kwargs.get('input', '')
            
            # If no input, return available agents
            if not user_input:
                return await self._get_available_agents_impl(context)
            
            # Analyze the request to determine the best agent to handle it
            analysis = await self._analyze_request_impl(context, user_input)
            
            # If we have a recommended agent, route to it
            if 'recommended_agent' in analysis and analysis['recommended_agent']:
                return await self._route_to_agent_impl(
                    context=context,
                    agent_name=analysis['recommended_agent'],
                    input_data={'input': user_input}
                )
            
            # Otherwise, return the analysis
            return {
                'status': 'analysis_complete',
                'analysis': analysis,
                'next_steps': analysis.get('next_steps', [])
            }
            
        except Exception as e:
            logger.error(f"Error in router agent: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': f"Error processing request: {str(e)}"
            }
    
    async def _analyze_request_impl(
        self,
        context: ToolContext,
        user_input: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze a user request to determine the appropriate agent.
        
        Args:
            context: The tool context
            user_input: The user's input text
            **kwargs: Additional arguments
            
        Returns:
            Analysis of the request with recommended actions
        """
        # Get current workflow state
        workflow_state = self.state_manager.get_workflow_state(context)
        current_stage = workflow_state.current_stage
        
        # Simple keyword-based routing (can be enhanced with ML/NLP)
        user_input_lower = user_input.lower()
        
        # Check for specific intents
        if 'ingest' in user_input_lower or 'upload' in user_input_lower:
            return self._create_analysis_response(
                current_stage,
                workflow_state,
                'ingestion',
                'Document ingestion and processing'
            )
        elif 'citation' in user_input_lower or 'reference' in user_input_lower:
            return self._create_analysis_response(
                current_stage,
                workflow_state,
                'citation_graph',
                'Citation network analysis'
            )
        elif 'search' in user_input_lower or 'find' in user_input_lower:
            return self._create_analysis_response(
                current_stage,
                workflow_state,
                'deep_search',
                'Deep search and information retrieval'
            )
        elif 'analyze' in user_input_lower or 'compare' in user_input_lower:
            return self._create_analysis_response(
                current_stage,
                workflow_state,
                'cross_paper_analysis',
                'Cross-paper analysis'
            )
        elif 'insight' in user_input_lower or 'summarize' in user_input_lower:
            return self._create_analysis_response(
                current_stage,
                workflow_state,
                'insight',
                'Insight generation'
            )
        elif 'visualize' in user_input_lower or 'graph' in user_input_lower or 'chart' in user_input_lower:
            return self._create_analysis_response(
                current_stage,
                workflow_state,
                'visualization',
                'Data visualization'
            )
        
        # Default to ingestion if no clear intent
        return self._create_analysis_response(
            current_stage,
            workflow_state,
            'ingestion',
            'Document ingestion and processing (default)'
        )
    
    def _create_analysis_response(
        self,
        current_stage: Optional[WorkflowStage],
        workflow_state: Any,
        recommended_agent: str,
        description: str
    ) -> Dict[str, Any]:
        """Create a standardized analysis response.
        
        Args:
            current_stage: Current workflow stage
            workflow_state: Current workflow state
            recommended_agent: Name of the recommended agent
            description: Description of the recommended action
            
        Returns:
            Formatted analysis response
        """
        return {
            'intent': 'process_document' if not current_stage else 'continue_workflow',
            'confidence': 0.9,
            'recommended_agent': recommended_agent,
            'description': description,
            'current_stage': current_stage.value if current_stage else None,
            'next_steps': [stage.value for stage in workflow_state.next_possible_stages],
            'available_agents': list(self._agent_descriptions.keys())
        }
    
    async def _get_available_agents_impl(self, context: ToolContext) -> Dict[str, Any]:
        """Get information about all available agents.
        
        Args:
            context: The tool context
            
        Returns:
            Dictionary with agent information
        """
        return {
            'available_agents': [
                {
                    'name': name,
                    'description': desc,
                    'circuit_state': self.circuit_breaker.get_breaker(name).state.name
                }
                for name, desc in self._agent_descriptions.items()
            ],
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Mark tools as initialized
        self._tools_initialized = True
    
    async def update_workflow_stage(
        self,
        context: ToolContext,
        stage: WorkflowStage,
        is_complete: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update the current workflow stage with persistence.
        
        Args:
            context: The tool context with state_id
            stage: The new workflow stage
            is_complete: Whether the current stage is complete
            metadata: Additional metadata for the stage
            
        Returns:
            Updated workflow state with version information
        """
        try:
            # Ensure state is loaded
            if not self._state_loaded:
                await self.state_manager.get_state(context)
                self._state_loaded = True
                
            # Update the workflow stage with persistence
            success = await self.state_manager.update_workflow_stage(
                context=context,
                stage=stage,
                is_complete=is_complete,
                metadata=metadata or {}
            )
            
            if not success:
                logger.error(f"Failed to update workflow stage to {stage}")
                return {
                    "success": False,
                    "error": f"Failed to update workflow stage to {stage}",
                    "state_id": getattr(context, 'state_id', None)
                }
                
            # Get the updated state with version info
            state = await self.state_manager.get_state(context)
            
            # Get metadata with version
            metadata = await self.state_manager.state_store.get_state(
                getattr(context, 'state_id', ''),
                StateMetadata
            )
            
            return {
                "success": True,
                "workflow": state.get('workflow', {}),
                "version": metadata.version if metadata else "unknown",
                "last_updated": metadata.last_updated if metadata else None,
                "state_id": getattr(context, 'state_id', None),
                "message": f"Updated workflow stage to {stage}"
            }
            
        except Exception as e:
            logger.error(f"Error updating workflow stage: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "state_id": getattr(context, 'state_id', None)
            }
    
    async def get_agent_status(self, agent_name: str) -> Dict[str, Any]:
        """Get the status of a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Status information for the agent
        """
        if agent_name not in self._sub_agents:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        breaker = self.circuit_breaker.get_breaker(agent_name)
        
        return {
            'name': agent_name,
            'description': self._agent_descriptions.get(agent_name, ''),
            'circuit_state': breaker.state.name,
            'failure_count': breaker._failure_count,
            'last_failure': breaker._last_failure_time.isoformat() if breaker._last_failure_time else None,
            'is_available': breaker.state == CircuitState.CLOSED
        }
    
    async def get_workflow_state(self, context: ToolContext) -> Dict[str, Any]:
        """Get the current workflow state.
        
        Args:
            context: The tool context
            
        Returns:
            Current workflow state information
        """
        workflow_state = self.state_manager.get_workflow_state(context)
        return {
            'current_stage': workflow_state.current_stage.value if workflow_state.current_stage else None,
            'is_complete': workflow_state.is_complete,
            'next_possible_stages': [stage.value for stage in workflow_state.next_possible_stages],
            'metadata': workflow_state.metadata
        }
    
    async def update_workflow_stage(
        self,
        context: ToolContext,
        stage: WorkflowStage,
        is_complete: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update the current workflow stage.
        
        Args:
            context: The tool context
            stage: The new workflow stage
            is_complete: Whether the current stage is complete
            metadata: Additional metadata for the stage
            
        Returns:
            Updated workflow state
        """
        self.state_manager.update_workflow_stage(context, stage, is_complete, metadata)
        return await self.get_workflow_state(context)
    
    async def reset_workflow(self, context: ToolContext) -> Dict[str, Any]:
        """Reset the workflow to its initial state.
        
        Args:
            context: The tool context
            
        Returns:
            Reset workflow state
        """
        self.state_manager.reset_workflow(context)
        return await self.get_workflow_state(context)
    
    async def _process_request(
        self, 
        request: str, 
        context: InvocationContext
    ) -> str:
        """Process a user request by analyzing and routing it to the appropriate agent.
        
        This is a convenience method that wraps the main run() method for backward compatibility.
        
        Args:
            request: The user's request text
            context: The invocation context
            
        Returns:
            The agent's response as a string
        """
        try:
            # Convert InvocationContext to ToolContext if needed
            tool_context = ToolContext(
                user_id=context.user_id if hasattr(context, 'user_id') else None,
                session_id=context.session_id if hasattr(context, 'session_id') else None,
                metadata={
                    'request_id': context.request_id if hasattr(context, 'request_id') else None,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
            
            # Process the request using the main run method
            result = await self.run(tool_context, input=request)
            
            # Format the response
            if result.get('success', False):
                if 'response' in result:
                    return str(result['response'])
                return "Request processed successfully"
            else:
                return f"Error: {result.get('error', 'Unknown error')}"
                
        except Exception as e:
            logger.error(f"Error in _process_request: {str(e)}", exc_info=True)
            return f"An error occurred while processing your request: {str(e)}"
    
    # Additional utility methods
    async def get_agent_status(self, agent_name: str) -> Dict[str, Any]:
        """Get the status of a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Status information for the agent
        """
        if agent_name not in self._sub_agents:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        breaker = self.circuit_breaker.get_breaker(agent_name)
        
        return {
            'name': agent_name,
            'description': self._agent_descriptions.get(agent_name, ''),
            'circuit_state': breaker.state.name,
            'failure_count': breaker._failure_count,
            'last_failure': breaker._last_failure_time.isoformat() if breaker._last_failure_time else None,
            'is_available': breaker.state == CircuitState.CLOSED
        }
    
    async def get_workflow_state(self, context: ToolContext) -> Dict[str, Any]:
        """Get the current workflow state.
        
        Args:
            context: The tool context
            
        Returns:
            Current workflow state information
        """
        workflow_state = self.state_manager.get_workflow_state(context)
        return {
            'current_stage': workflow_state.current_stage.value if workflow_state.current_stage else None,
            'is_complete': workflow_state.is_complete,
            'next_possible_stages': [stage.value for stage in workflow_state.next_possible_stages],
            'metadata': workflow_state.metadata
        }
    
    async def update_workflow_stage(
        self,
        context: ToolContext,
        stage: WorkflowStage,
        is_complete: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update the current workflow stage.
        
        Args:
            context: The tool context
            stage: The new workflow stage
            is_complete: Whether the current stage is complete
            metadata: Additional metadata for the stage
            
        Returns:
            Updated workflow state
        """
        self.state_manager.update_workflow_stage(context, stage, is_complete, metadata)
        return await self.get_workflow_state(context)
    
    async def reset_workflow(self, context: ToolContext) -> Dict[str, Any]:
        """Reset the workflow to its initial state.
        
        Args:
            context: The tool context
            
        Returns:
            Reset workflow state
        """
        self.state_manager.reset_workflow(context)
        return await self.get_workflow_state(context)
    
    async def _process_request(
        self, 
        request: str, 
        context: InvocationContext
    ) -> str:
        """Process a user request by analyzing and routing it to the appropriate agent.
        
        This is a convenience method that wraps the main run() method for backward compatibility.
        
        Args:
            request: The user's request text
            context: The invocation context
            
        Returns:
            The agent's response as a string
        """
        try:
            # Convert InvocationContext to ToolContext if needed
            tool_context = ToolContext(
                user_id=context.user_id if hasattr(context, 'user_id') else None,
                session_id=context.session_id if hasattr(context, 'session_id') else None,
                metadata={
                    'request_id': context.request_id if hasattr(context, 'request_id') else None,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
            
            # Process the request using the main run method
            result = await self.run(tool_context, input=request)
            
            # Format the response
            if result.get('success', False):
                if 'response' in result:
                    return str(result['response'])
                return "Request processed successfully"
            else:
                return f"Error: {result.get('error', 'Unknown error')}"
                
        except Exception as e:
            logger.error(f"Error in _process_request: {str(e)}", exc_info=True)
            return f"An error occurred while processing your request: {str(e)}"
    
    # Clean up complete - all redundant methods have been removed


# Create a global instance of the router agent
router_agent = RouterAgent()


async def main():
    """Main entry point for the ScholarVerse agent."""
    try:
        # Initialize the router agent
        await router_agent.initialize()
        
        # Example usage
        context = InvocationContext()
        context.user_input = "Find recent papers about quantum computing"
        
        # Process the request
        response = await router_agent.process_request(
            context.user_input,
            context
        )
        
        print(f"Response: {response}")
                
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
