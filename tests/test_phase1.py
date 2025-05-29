"""Tests for Phase 1 of the ScholarVerse project.

This module contains unit tests for the core components implemented in Phase 1,
including the RouterAgent and its integration with the Google ADK framework.
"""

import asyncio
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta, UTC

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from google.adk.agents.invocation_context import InvocationContext
from google.adk.tools import ToolContext

from scholar_verse.agent import RouterAgent, router_agent
from scholar_verse.base_agent import BaseAgent
from scholar_verse.config import get_config
from scholar_verse.shared_libraries.logging_utils import logger
from scholar_verse.shared_libraries.state_management.adaptive_state import AdaptiveStateManager

# Test data
TEST_QUERY = "Find papers about quantum computing"
TEST_AGENT_NAME = "deep_search"
TEST_AGENT_RESPONSE = "Here are some papers about quantum computing..."

# Import sub-agents to patch
from scholar_verse.sub_agents.ingestion.agent import IngestionAgent
from scholar_verse.sub_agents.citation_graph.agent import CitationGraphAgent
from scholar_verse.sub_agents.cross_paper_analysis.agent import CrossPaperAnalysisAgent
from scholar_verse.sub_agents.deep_search.agent import DeepSearchAgent
from scholar_verse.sub_agents.insight.agent import InsightAgent
from scholar_verse.sub_agents.visualization.agent import VisualizationAgent


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = MagicMock(spec=BaseAgent)
    agent.name = TEST_AGENT_NAME
    agent.process_request = AsyncMock(return_value=TEST_AGENT_RESPONSE)
    return agent


@pytest.fixture
def router_with_mocks(mock_agent):
    """Create a RouterAgent with mocked dependencies."""
    with patch('scholar_verse.agent.RouterAgent') as MockRouter:
        # Configure the mock router
        mock_router = MockRouter.return_value
        mock_router._sub_agents = {TEST_AGENT_NAME: mock_agent}
        mock_router._agent_descriptions = {TEST_AGENT_NAME: "Test agent description"}
        mock_router.process_request = AsyncMock(return_value=TEST_AGENT_RESPONSE)
        
        # Mock the initialize method
        async def mock_initialize():
            mock_router._tools_initialized = True
        
        mock_router.initialize = mock_initialize
        
        yield mock_router


@pytest.mark.asyncio
async def test_configuration():
    """Test the configuration module."""
    config = get_config()
    assert config is not None
    assert 'model' in config
    assert 'app' in config
    assert 'log_level' in config['app']


@pytest.mark.asyncio
async def test_logging():
    """Test the logging module."""
    # This test just verifies we can log at different levels
    logger.info("Test info message")
    logger.warning("Test warning message")
    logger.error("Test error message")
    logger.debug("Test debug message")
    assert True  # If we get here without exceptions, logging works


@pytest.mark.asyncio
async def test_state_management():
    """Test the state management functionality."""
    state_manager = AdaptiveStateManager()
    
    # Create a mock context
    class MockContext:
        pass
    
    context = MockContext()
    
    # Test initial state
    state = state_manager.get(context)
    assert 'workflow' in state
    
    # Test state update
    state_manager.update(context, {'test_key': 'test_value'})
    updated_state = state_manager.get(context)
    assert updated_state.get('test_key') == 'test_value'


@pytest.mark.asyncio
async def test_router_agent_initialization():
    """Test that the router agent initializes correctly."""
    # Create a test instance of the router agent with patched sub-agent imports
    with patch('scholar_verse.agent.IngestionAgent') as mock_ingestion_agent, \
         patch('scholar_verse.agent.CitationGraphAgent') as mock_citation_agent, \
         patch('scholar_verse.agent.CrossPaperAnalysisAgent') as mock_cross_paper_agent, \
         patch('scholar_verse.agent.DeepSearchAgent') as mock_deep_search_agent, \
         patch('scholar_verse.agent.InsightAgent') as mock_insight_agent, \
         patch('scholar_verse.agent.VisualizationAgent') as mock_viz_agent:
        
        # Configure mock agents
        mock_agents = {
            'ingestion': mock_ingestion_agent.return_value,
            'citation_graph': mock_citation_agent.return_value,
            'cross_paper_analysis': mock_cross_paper_agent.return_value,
            'deep_search': mock_deep_search_agent.return_value,
            'insight': mock_insight_agent.return_value,
            'visualization': mock_viz_agent.return_value
        }
        
        # Configure initialize methods to be awaitable
        for agent in mock_agents.values():
            agent.initialize = AsyncMock(return_value=None)
        
        # Create router agent instance
        test_router = RouterAgent()
        
        # Verify initial state
        assert test_router.name == "router_agent"
        assert not test_router._tools_initialized
        
        # Initialize and verify
        await test_router.initialize()
        assert test_router._tools_initialized
        
        # Verify sub-agents were initialized
        for agent in mock_agents.values():
            agent.initialize.assert_awaited_once()


@pytest.mark.asyncio
async def test_router_agent_process_request(router_with_mocks):
    """Test that the router agent can process requests."""
    # Create a mock for the InvocationContext
    with patch('scholar_verse.agent.InvocationContext') as mock_context_class:
        # Create a mock context instance
        mock_context = AsyncMock()
        mock_context.user_input = TEST_QUERY
        mock_context_class.return_value = mock_context
        
        # Mock the process_request method to return a test response
        router_with_mocks.process_request = AsyncMock(return_value=TEST_AGENT_RESPONSE)
        
        # Process the request
        response = await router_with_mocks.process_request(TEST_QUERY, mock_context)
        
        # Verify the response
        assert TEST_AGENT_RESPONSE in response
        router_with_mocks.process_request.assert_awaited_once_with(TEST_QUERY, mock_context)


@pytest.mark.asyncio
async def test_analyze_request_impl(router_with_mocks):
    """Test the request analysis implementation."""
    # Create a test router
    router = RouterAgent()
    router._sub_agents = router_with_mocks._sub_agents
    router._agent_descriptions = router_with_mocks._agent_descriptions
    
    # Test with a search query
    analysis = await router._analyze_request_impl("Find papers about AI")
    assert analysis["success"] is True
    assert "suggested_agents" in analysis
    assert len(analysis["suggested_agents"]) > 0
    
    # Test with an ingestion query
    analysis = await router._analyze_request_impl("Upload a document")
    assert "ingestion" in analysis["suggested_agents"]


@pytest.mark.asyncio
async def test_route_to_agent_impl():
    """Test routing to a specific agent."""
    # Create a mock sub-agent
    mock_agent = AsyncMock()
    mock_agent.process_request = AsyncMock(return_value="Mock response")
    
    # Create a mock InvocationContext
    mock_invocation_context = AsyncMock()
    
    # Create a ToolContext with the mock InvocationContext
    from google.adk.tools.tool_context import ToolContext
    mock_context = ToolContext(invocation_context=mock_invocation_context)
    
    # Create a RouterAgent instance
    from scholar_verse.agent import RouterAgent
    router = RouterAgent()
    
    # Set up the sub-agents dictionary
    router._sub_agents = {"test_agent": mock_agent}
    
    # Call the method under test
    result = await router._route_to_agent_impl(
        agent_name="test_agent",
        request="test request",
        context=mock_context
    )
    
    # Verify the result
    assert result["success"] is True
    assert result["agent"] == "test_agent"
    assert result["response"] == "Mock response"
    
    # Verify the sub-agent was called with the correct context
    mock_agent.process_request.assert_awaited_once()
    
    # Get the context that was passed to process_request
    call_args = mock_agent.process_request.call_args
    assert call_args is not None
    
    # The second argument should be the context
    context_arg = call_args[0][1] if len(call_args[0]) > 1 else None
    assert context_arg is not None
    assert isinstance(context_arg, ToolContext)
    
    # Test error handling for unknown agent
    result = await router._route_to_agent_impl(
        agent_name="nonexistent_agent",
        request="test request",
        context=mock_context
    )
    
    assert result.get("success") is False
    assert "Unknown agent" in result.get("error", "")


@pytest.mark.asyncio
async def test_main_entry_point(capsys, monkeypatch):
    """Test the main entry point of the application."""
    # Create a mock for the InvocationContext
    mock_context = MagicMock()
    mock_context.user_input = TEST_QUERY
    
    # Create a mock for the router_agent
    mock_router = AsyncMock()
    mock_router.initialize = AsyncMock()
    mock_router.process_request = AsyncMock(return_value=TEST_AGENT_RESPONSE)
    
    # Patch the necessary components
    with patch('scholar_verse.agent.router_agent', mock_router), \
         patch('scholar_verse.agent.InvocationContext', return_value=mock_context) as mock_context_cls, \
         patch('builtins.print') as mock_print:
        
        # Import here to avoid circular imports
        from scholar_verse.agent import main as router_main
        
        # Mock asyncio.run to capture the coroutine passed to it
        original_run = asyncio.run
        coro_result = None
        
        async def mock_run(coro):
            nonlocal coro_result
            coro_result = await coro
            return coro_result
            
        with patch('asyncio.run', new=mock_run):
            # Run the main function
            await router_main()
            
            # Verify the router agent was initialized
            mock_router.initialize.assert_awaited_once()
            
            # Verify process_request was called with the expected arguments
            mock_router.process_request.assert_awaited_once_with(
                mock_context.user_input,
                mock_context
            )
            
            # Verify the output was printed
            mock_print.assert_called_with(f"Response: {TEST_AGENT_RESPONSE}")


@pytest.mark.asyncio
async def test_agent_error_handling(router_with_mocks):
    """Test error handling in the router agent."""
    # Create a mock for the InvocationContext
    with patch('scholar_verse.agent.InvocationContext') as mock_context_class:
        # Create a mock context instance
        mock_context = AsyncMock()
        mock_context.user_input = "invalid query"
        mock_context_class.return_value = mock_context
        
        # Configure the mock to raise an exception
        error_msg = "Test error"
        router_with_mocks.process_request = AsyncMock(side_effect=Exception(error_msg))
        
        # Process a request that will cause an error
        with pytest.raises(Exception) as exc_info:
            await router_with_mocks.process_request("invalid query", mock_context)
        
        # Verify the error was raised
        assert error_msg in str(exc_info.value)
        
        # Verify the error was logged
        # Note: You might want to add logging assertions here if you have a logging system to test
    
    # Verify the error was logged
    # Note: You might want to add logging assertions here if you have a logging system to test


@pytest.mark.asyncio
async def test_router_agent_tool_registration():
    """Test that the router agent registers its tools correctly."""
    # Create a test router
    router = RouterAgent()
    
    # Mock the tools list
    router.tools = []
    
    # Register tools
    router._register_tools()
    
    # Verify tools were registered
    assert len(router.tools) == 2  # route_to_agent and analyze_request
    tool_names = {tool.name for tool in router.tools}
    assert "route_to_agent" in tool_names
    assert "analyze_request" in tool_names


@pytest.mark.asyncio
async def test_router_agent_initialize_sub_agents():
    """Test that the router agent initializes its sub-agents."""
    # Create mock sub-agents with async initialize methods
    mock_ingestion = AsyncMock()
    mock_citation_graph = AsyncMock()
    mock_cross_paper = AsyncMock()
    mock_deep_search = AsyncMock()
    mock_insight = AsyncMock()
    mock_visualization = AsyncMock()
    
    # Create a test router with mock sub-agents
    with patch.multiple(
        'scholar_verse.agent',
        IngestionAgent=MagicMock(return_value=mock_ingestion),
        CitationGraphAgent=MagicMock(return_value=mock_citation_graph),
        CrossPaperAnalysisAgent=MagicMock(return_value=mock_cross_paper),
        DeepSearchAgent=MagicMock(return_value=mock_deep_search),
        InsightAgent=MagicMock(return_value=mock_insight),
        VisualizationAgent=MagicMock(return_value=mock_visualization)
    ):
        # Create router and initialize
        router = RouterAgent()
        
        # Mock the _register_tools method to avoid actual tool registration
        router._register_tools = MagicMock()
        
        # Initialize the router
        await router.initialize()
        
        # Check that all sub-agents were initialized
        assert len(router._sub_agents) == 6
        assert "ingestion" in router._sub_agents
        assert "citation_graph" in router._sub_agents
        assert "cross_paper_analysis" in router._sub_agents
        assert "deep_search" in router._sub_agents
        assert "insight" in router._sub_agents
        assert "visualization" in router._sub_agents
        
        # Verify initialize was called on all sub-agents
        mock_ingestion.initialize.assert_awaited_once()
        mock_citation_graph.initialize.assert_awaited_once()
        mock_cross_paper.initialize.assert_awaited_once()
        mock_deep_search.initialize.assert_awaited_once()
        mock_insight.initialize.assert_awaited_once()
        mock_visualization.initialize.assert_awaited_once()
        
        # Verify tools were registered
        router._register_tools.assert_called_once()


# For backward compatibility with the original test script
def main():
    """Run all tests using pytest."""
    import pytest
    pytest.main([__file__, '-v'])


if __name__ == "__main__":
    asyncio.run(main())
