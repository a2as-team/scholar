"""Comprehensive test suite for Phase 2 implementation of ScholarVerse.

This test suite verifies the functionality of the Adaptive Router Agent,
state management, and inter-agent communication components.
"""

import os
import sys
import asyncio
import pytest
import json
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, Optional

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from scholar_verse.config import get_config, DEFAULT_MODEL
from scholar_verse.shared_libraries.logging_utils import logger
from scholar_verse.agent import RouterAgent, router_agent, main
from scholar_verse.shared_libraries.state_management.adaptive_state import (
    AdaptiveStateManager, WorkflowStage, StateKey
)
from scholar_verse.tools.dynamic_routing import setup_dynamic_routing
from scholar_verse.tools.feedback_processing import FeedbackProcessor
from scholar_verse.tools.agent_evaluation import AgentEvaluator
from scholar_verse.shared_libraries.state_management.redis_state_store import RedisStateStore


class MockToolContext:
    """Mock tool context for testing with enhanced state management."""
    
    def __init__(self, state_id: Optional[str] = None):
        self.state_id = state_id or f"test_{datetime.now(timezone.utc).timestamp()}"
        self.state_manager = AdaptiveStateManager()
        self.state = {'state_manager': self.state_manager}
        self.metadata = {}
        
    async def get_state(self):
        """Get the current state."""
        return await self.state_manager.get_state(self)
        
    async def update_state(self, state_data: Dict[str, Any]):
        """Update the current state."""
        return await self.state_manager.update_state(self, state_data)


@pytest.mark.asyncio
async def test_workflow_state_management():
    """Test the workflow state management and transitions."""
    # Create a test context
    context = MockToolContext()
    
    # Initialize state
    await context.get_state()
    
    # Test initial state
    initial_state = await context.state_manager.get_workflow_state(context)
    assert initial_state.current_stage == WorkflowStage.INITIAL
    assert not initial_state.is_complete
    
    # Test state transitions
    for stage in WorkflowStage:
        await context.state_manager.update_workflow_stage(
            context, 
            stage=stage,
            is_complete=True,
            metadata={"test_data": f"test_{stage.value}"}
        )
        
        # Verify state was updated
        updated_state = await context.state_manager.get_workflow_state(context)
        assert updated_state.current_stage == stage
        assert updated_state.is_complete
        assert updated_state.metadata.get("test_data") == f"test_{stage.value}"
    
    # Test state reset
    await context.state_manager.reset_workflow(context)
    reset_state = await context.state_manager.get_workflow_state(context)
    assert reset_state.current_stage == WorkflowStage.INITIAL
    assert not reset_state.is_complete

@pytest.mark.asyncio
async def test_router_agent_initialization():
    """Test the router agent initialization and configuration."""
    # Create a test config
    test_config = {
        "model": "test-model",
        "max_retry_attempts": 3,
        "circuit_breakers": {
            "ingestion": {"failure_threshold": 2},
            "citation_graph": {"failure_threshold": 2}
        }
    }
    
    # Initialize router agent
    agent = RouterAgent(config=test_config)
    
    # Verify configuration
    assert agent.model == "test-model"
    assert agent.retry_config["max_attempts"] == 3
    
    # Verify circuit breakers
    assert agent.circuit_breaker.get_breaker("ingestion").failure_threshold == 2
    assert agent.circuit_breaker.get_breaker("citation_graph").failure_threshold == 2

@pytest.mark.asyncio
async def test_agent_routing():
    """Test the agent routing functionality."""
    # Initialize router agent with mock sub-agents
    router = RouterAgent()
    
    # Create mock sub-agents
    mock_agents = {
        "ingestion": AsyncMock(),
        "citation_graph": AsyncMock(),
        "deep_search": AsyncMock()
    }
    
    # Configure mock responses
    mock_agents["ingestion"].process_request.return_value = {"status": "success"}
    mock_agents["citation_graph"].process_request.return_value = {"status": "success"}
    mock_agents["deep_search"].process_request.return_value = {"status": "success"}
    
    # Patch the sub-agents
    router._sub_agents = mock_agents
    
    # Test routing to each agent
    for agent_name in mock_agents.keys():
        context = MockToolContext()
        result = await router._route_to_agent_impl(
            context=context,
            agent_name=agent_name,
            input_data={"input": f"Test {agent_name} input"}
        )
        
        # Verify the agent was called
        mock_agents[agent_name].process_request.assert_awaited_once()
        assert result["success"] is True
        assert result["agent"] == agent_name
        
        # Verify state was updated
        state = await context.state_manager.get_workflow_state(context)
        assert state.current_stage.name.lower() in agent_name or \
               any(agent_name in stage.name.lower() for stage in WorkflowStage)

@pytest.mark.asyncio
async def test_circuit_breaker_behavior():
    """Test the circuit breaker behavior with failing agents."""
    # Initialize router agent with a failing mock agent
    router = RouterAgent()
    
    # Create a failing mock agent
    mock_agent = AsyncMock()
    mock_agent.process_request.side_effect = Exception("Test failure")
    router._sub_agents = {"test_agent": mock_agent}
    
    # Get the circuit breaker
    breaker = router.circuit_breaker.get_breaker("test_agent")
    
    # First call should fail but not open the circuit
    context = MockToolContext()
    result = await router._route_to_agent_impl(
        context=context,
        agent_name="test_agent",
        input_data={"input": "Test input"}
    )
    
    assert result["success"] is False
    assert breaker.failure_count == 1
    
    # After max failures, circuit should open
    for _ in range(breaker.failure_threshold):
        await router._route_to_agent_impl(
            context=context,
            agent_name="test_agent",
            input_data={"input": "Test input"}
        )
    
    # Next call should be rejected by circuit breaker
    with pytest.raises(Exception) as exc_info:
        await router._route_to_agent_impl(
            context=context,
            agent_name="test_agent",
            input_data={"input": "Test input"}
        )
    assert "Circuit 'test_agent' is OPEN" in str(exc_info.value)
    
    # Reset the circuit
    breaker.record_success()
    assert breaker.state.name == "CLOSED"


@pytest.mark.asyncio
async def test_feedback_processing():
    """Test the feedback processing module with comprehensive test cases."""
    # Create test context
    context = MockToolContext()
    
    # Initialize feedback processor with mock state manager
    feedback_processor = FeedbackProcessor(context.state_manager)
    
    # Test user feedback processing
    test_feedback = {
        "feedback": "The citation graph visualization is excellent!",
        "rating": 5,
        "feedback_type": "visualization_quality",
        "agent_name": "visualization"
    }
    
    # Process user feedback
    feedback_result = await feedback_processor.process_user_feedback(
        feedback=test_feedback["feedback"],
        rating=test_feedback["rating"],
        feedback_type=test_feedback["feedback_type"],
        agent_name=test_feedback["agent_name"],
        tool_context=context
    )
    
    # Verify feedback was processed
    assert feedback_result["type"] == test_feedback["feedback_type"]
    assert feedback_result["rating"] == test_feedback["rating"]
    assert feedback_result["agent"] == test_feedback["agent_name"]
    
    # Test agent feedback processing
    agent_feedback = await feedback_processor.process_agent_feedback(
        source_agent="router",
        target_agent="ingestion",
        feedback="Successfully processed document",
        success=True,
        metrics={"processing_time": 1.5, "documents_processed": 3},
        tool_context=context
    )
    
    # Verify agent feedback was processed
    assert agent_feedback["source_agent"] == "router"
    assert agent_feedback["target_agent"] == "ingestion"
    assert agent_feedback["success"] is True
    
    # Test getting performance metrics
    performance = await feedback_processor.get_agent_performance("ingestion", context)
    assert performance["total_feedbacks"] >= 1
    assert "success_rate" in performance
    assert "average_processing_time" in performance
    
    # Test feedback aggregation
    for i in range(3):
        await feedback_processor.process_agent_feedback(
            source_agent="router",
            target_agent="ingestion",
            feedback=f"Processing batch {i}",
            success=i < 2,  # 2 success, 1 failure
            metrics={"processing_time": 1.0 + i * 0.5},
            tool_context=context
        )
    
    # Verify aggregated metrics
    updated_perf = await feedback_processor.get_agent_performance("ingestion", context)
    assert updated_perf["total_feedbacks"] == 4  # 1 initial + 3 new
    assert 0.6 < updated_perf["success_rate"] < 0.8  # ~75% success
    assert 1.0 <= updated_perf["average_processing_time"] <= 2.0

@pytest.mark.asyncio
async def test_agent_evaluation():
    """Test the agent evaluation module with comprehensive test cases."""
    # Create test context
    context = MockToolContext()
    
    # Initialize feedback processor and evaluator
    feedback_processor = FeedbackProcessor(context.state_manager)
    evaluator = AgentEvaluator(context.state_manager, feedback_processor)
    
    # Add test feedback data
    test_data = [
        {"success": True, "metrics": {"processing_time": 1.0, "accuracy": 0.9}, "agent": "ingestion"},
        {"success": True, "metrics": {"processing_time": 1.2, "accuracy": 0.85}, "agent": "ingestion"},
        {"success": False, "metrics": {"processing_time": 2.5, "error": "timeout"}, "agent": "ingestion"},
        {"success": True, "metrics": {"processing_time": 0.8, "accuracy": 0.95}, "agent": "visualization"},
    ]
    
    # Process test data
    for i, data in enumerate(test_data):
        await feedback_processor.process_agent_feedback(
            source_agent="test_runner",
            target_agent=data["agent"],
            feedback=f"Test feedback {i}",
            success=data["success"],
            metrics=data["metrics"],
            tool_context=context
        )
    
    # Test agent evaluation
    eval_results = await evaluator.evaluate_agent_performance("ingestion", context)
    
    # Verify evaluation results
    assert "score" in eval_results
    assert "metrics" in eval_results
    assert "success_rate" in eval_results["metrics"]
    assert "average_processing_time" in eval_results["metrics"]
    
    # Test agent recommendation
    recommended_agent = await evaluator.recommend_agent(
        request="I need to process some documents",
        candidate_agents=["ingestion", "visualization"],
        tool_context=context
    )
    
    # Verify recommendation is one of the candidates
    assert recommended_agent in ["ingestion", "visualization"]
    
    # Test ranking agents by performance
    rankings = await evaluator.rank_agents(
        context=context,
        metric="success_rate",
        limit=2
    )
    
    # Verify rankings
    assert len(rankings) <= 2
    if len(rankings) > 1:
        # Success rates should be in descending order
        assert rankings[0]["metrics"]["success_rate"] >= rankings[1]["metrics"]["success_rate"]

@pytest.mark.asyncio
async def test_integration_workflow():
    """Test the complete workflow from request to response with state management."""
    # Initialize router agent with mock sub-agents
    router = RouterAgent()
    
    # Create mock sub-agents with realistic responses
    mock_agents = {
        "ingestion": AsyncMock(),
        "citation_graph": AsyncMock(),
        "deep_search": AsyncMock()
    }
    
    # Configure mock responses
    mock_agents["ingestion"].process_request.return_value = {
        "status": "success",
        "documents_processed": 3,
        "metadata_extracted": {"title": "Test Document"}
    }
    mock_agents["citation_graph"].process_request.return_value = {
        "status": "success",
        "citations_processed": 5,
        "relationships_found": 8
    }
    mock_agents["deep_search"].process_request.return_value = {
        "status": "success",
        "results": [
            {"title": "Related Paper 1", "relevance": 0.92},
            {"title": "Related Paper 2", "relevance": 0.87}
        ]
    }
    
    # Patch the sub-agents
    router._sub_agents = mock_agents
    
    # Test a complete workflow
    context = MockToolContext()
    
    # 1. Document ingestion
    ingestion_result = await router._route_to_agent_impl(
        context=context,
        agent_name="ingestion",
        input_data={
            "action": "process_documents",
            "documents": ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
        }
    )
    
    # Verify ingestion results
    assert ingestion_result["success"] is True
    assert ingestion_result["agent"] == "ingestion"
    assert ingestion_result["response"]["documents_processed"] == 3
    
    # Verify state was updated
    state = await context.state_manager.get_workflow_state(context)
    assert state.current_stage == WorkflowStage.DOCUMENT_INGESTION
    assert state.is_complete is True
    
    # 2. Citation graph construction
    citation_result = await router._route_to_agent_impl(
        context=context,
        agent_name="citation_graph",
        input_data={"action": "build_graph"}
    )
    
    # Verify citation graph results
    assert citation_result["success"] is True
    assert citation_result["agent"] == "citation_graph"
    assert citation_result["response"]["citations_processed"] == 5
    
    # 3. Deep search
    search_result = await router._route_to_agent_impl(
        context=context,
        agent_name="deep_search",
        input_data={"query": "related research papers"}
    )
    
    # Verify search results
    assert search_result["success"] is True
    assert search_result["agent"] == "deep_search"
    assert len(search_result["response"]["results"]) == 2
    
    # Verify final state
    final_state = await context.state_manager.get_workflow_state(context)
    assert final_state.current_stage == WorkflowStage.DEEP_SEARCH
    assert "last_agent" in final_state.metadata
    assert "timestamp" in final_state.metadata
    
    # Verify all agents were called with the correct parameters
    mock_agents["ingestion"].process_request.assert_awaited_once()
    mock_agents["citation_graph"].process_request.assert_awaited_once()
    mock_agents["deep_search"].process_request.assert_awaited_once()


@pytest.mark.asyncio
async def test_router_agent_initialization_and_tools():
    """Test the router agent initialization and tool registration."""
    # Initialize router agent with test config
    test_config = {
        "model": "test-model",
        "max_retry_attempts": 5,
        "circuit_breakers": {
            "ingestion": {"failure_threshold": 3},
            "visualization": {"failure_threshold": 2}
        }
    }
    
    agent = RouterAgent(config=test_config)
    
    # Test initialization
    assert agent.model == "test-model"
    assert agent.retry_config["max_attempts"] == 5
    
    # Test tool registration
    tool_names = [tool.name for tool in agent.tools]
    expected_tools = ["route_to_agent", "get_circuit_status", "reset_circuit"]
    
    for tool in expected_tools:
        assert tool in tool_names, f"Expected tool {tool} not found"
    
    # Test circuit breaker configuration
    assert agent.circuit_breaker.get_breaker("ingestion").failure_threshold == 3
    assert agent.circuit_breaker.get_breaker("visualization").failure_threshold == 2

@pytest.mark.asyncio
async def test_main_entry_point():
    """Test the main entry point of the application."""
    # Mock the router agent's run method
    with patch('scholar_verse.agent.router_agent.run') as mock_run:
        # Configure the mock to return a successful response
        mock_run.return_value = {"status": "success", "message": "Test response"}
        
        # Call the main function
        result = await main()
        
        # Verify the result
        assert result == {"status": "success", "message": "Test response"}
        mock_run.assert_awaited_once()

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in the router agent."""
    # Initialize router agent with a mock sub-agent that raises an exception
    router = RouterAgent()
    
    # Create a failing mock agent
    mock_agent = AsyncMock()
    mock_agent.process_request.side_effect = Exception("Test error")
    router._sub_agents = {"test_agent": mock_agent}
    
    # Test error handling
    with pytest.raises(Exception) as exc_info:
        await router._route_to_agent_impl(
            context=MockToolContext(),
            agent_name="test_agent",
            input_data={"input": "Test input"}
        )
    
    assert "Test error" in str(exc_info.value)

if __name__ == "__main__":
    # Run all tests
    import pytest
    import sys
    
    # Exit with the status code of the test run
    sys.exit(pytest.main(["-v", "-s", __file__]))
