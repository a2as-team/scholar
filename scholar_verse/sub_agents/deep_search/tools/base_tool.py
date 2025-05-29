"""Base tool class for Deep Search Agent tools."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, AsyncGenerator

from google.adk.tools.base_tool import BaseTool as ADKBaseTool
from google.adk.tools.tool_context import ToolContext
from google.adk.events.event import Event


class BaseTool(ADKBaseTool, ABC):
    """Base class for all tools used by the DeepSearchAgent.
    
    This class extends the ADK BaseTool with common functionality
    and enforces a consistent interface for all tools.
    """
    
    def __init__(self, name: str, description: str):
        """Initialize the base tool.
        
        Args:
            name: The name of the tool.
            description: A description of what the tool does.
        """
        super().__init__(name=name, description=description)
    
    @abstractmethod
    async def _call(
        self, 
        args: Dict[str, Any], 
        context: Optional[ToolContext] = None
    ) -> Dict[str, Any]:
        """Execute the tool's main functionality.
        
        This method must be implemented by all concrete tool classes.
        
        Args:
            args: A dictionary of arguments for the tool.
            context: Optional context for the tool execution.
            
        Returns:
            A dictionary containing the tool's results.
        """
        pass
    
    async def __call__(
        self, 
        args: Dict[str, Any], 
        context: Optional[ToolContext] = None
    ) -> Dict[str, Any]:
        """Execute the tool with error handling and logging.
        
        Args:
            args: A dictionary of arguments for the tool.
            context: Optional context for the tool execution.
            
        Returns:
            A dictionary containing the tool's results or error information.
        """
        try:
            return await self._call(args, context or ToolContext())
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'tool': self.name
            }
    
    async def _run_async(
        self, 
        args: Dict[str, Any], 
        context: Optional[ToolContext] = None
    ) -> AsyncGenerator[Event, None]:
        """Run the tool asynchronously, yielding events.
        
        This provides a default implementation that wraps the tool's _call
        method in an event generator. Tools can override this if they need
        more control over event generation.
        
        Args:
            args: A dictionary of arguments for the tool.
            context: Optional context for the tool execution.
            
        Yields:
            Events generated during tool execution.
        """
        result = await self(args, context)
        yield Event(content=Content.from_text(json.dumps(result)))
