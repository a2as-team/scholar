"""Test cases for the ScholarVerse agent system."""

import os
import sys
import unittest
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timezone

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from google.genai import types
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.agents.invocation_context import InvocationContext

from scholar_verse.base_agent import BaseAgent, AgentRole, AgentContext
from scholar_verse.shared_libraries.session_manager import SessionManager
from scholar_verse.shared_libraries.state_manager import StateManager, StateScope
from scholar_verse.shared_libraries.memory_manager import MemoryManager

# Test data
TEST_USER_ID = "test_user_123"
TEST_SESSION_ID = "test_session_456"
TEST_AGENT_NAME = "test_agent"
TEST_AGENT_DESCRIPTION = "A test agent"
TEST_MODEL = "test-model"
TEST_INSTRUCTION = "Test instruction"
TEST_APP_NAME = "test_app"

# Initialize services
session_service = InMemorySessionService()
artifact_service = InMemoryArtifactService()


class TestScholarVerseAgents(unittest.TestCase):
    """Test cases for the ScholarVerse agent system."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a test session
        self.session = session_service.create_session(
            app_name=TEST_APP_NAME,
            user_id=TEST_USER_ID,
        )
        self.user_id = TEST_USER_ID
        self.session_id = self.session.id

        # Create a test runner
        self.runner = Runner(
            app_name=TEST_APP_NAME,
            agent=None,
            artifact_service=artifact_service,
            session_service=session_service,
        )

        # Create a test agent class that properly initializes required fields
        class TestAgent(BaseAgent):
            """Test agent implementation for testing."""
            role: AgentRole = AgentRole.ANALYSIS
            
            def __init__(self, *args, **kwargs):
                # Initialize required services if not provided
                session_manager = kwargs.pop('session_manager', SessionManager())
                memory_manager = kwargs.pop('memory_manager', MemoryManager())
                
                # Call parent with all required fields
                super().__init__(
                    name=TEST_AGENT_NAME,
                    description=TEST_AGENT_DESCRIPTION,
                    model=TEST_MODEL,
                    instruction=TEST_INSTRUCTION,
                    session_manager=session_manager,
                    memory_manager=memory_manager,
                    **kwargs
                )
            
            async def _process(self, context, *args, **kwargs):
                """Test implementation of the process method."""
                return {"result": "success"}
        
        # Store the test agent class for use in tests
        self.TestAgent = TestAgent
        
        # Create a default instance for tests that don't need customization
        self.test_agent = TestAgent(
            session_manager=SessionManager(),
            memory_manager=MemoryManager()
        )

    def _run_agent(self, agent, query):
        """Helper method to run an agent and get the final response."""
        self.runner.agent = agent
        content = types.Content(role="user", parts=[types.Part(text=query)])
        
        # Create a test invocation context
        invocation_context = InvocationContext(
            user_id=self.user_id,
            session_id=self.session_id,
            message=content,
            runner=self.runner,
        )
        
        # Run the agent
        events = list(self.runner._run_agent(agent, invocation_context))
        
        # Get the final response
        if events:
            last_event = events[-1]
            if hasattr(last_event, 'content') and hasattr(last_event.content, 'parts'):
                return "".join([part.text for part in last_event.content.parts if part.text])
        return None

    @pytest.mark.unit
    def test_agent_initialization(self):
        """Test that the test agent initializes correctly."""
        # Create a fresh instance for this test
        agent = self.TestAgent(
            session_manager=SessionManager(),
            memory_manager=MemoryManager()
        )
        self.assertEqual(agent.name, TEST_AGENT_NAME)
        self.assertEqual(agent.description, TEST_AGENT_DESCRIPTION)
        self.assertEqual(agent.model, TEST_MODEL)
        self.assertEqual(agent.role, AgentRole.ANALYSIS)
        self.assertIsNotNone(agent.session_manager)
        self.assertIsNotNone(agent.memory_manager)

    @pytest.mark.integration
    def test_agent_execution(self):
        """Test that the agent can process a query."""
        # Create a fresh instance for this test
        agent = self.TestAgent(
            session_manager=SessionManager(),
            memory_manager=MemoryManager()
        )
        query = "Test query"
        response = self._run_agent(agent, query)
        self.assertIsNotNone(response)
        # The actual response might be wrapped in ADK's response format
        # So we'll just check that we got some response
        self.assertTrue(isinstance(response, str))

    @pytest.mark.integration
    def test_agent_with_session(self):
        """Test that the agent can work with session state."""
        # Create a session manager and state manager with proper initialization
        session_manager = SessionManager()
        
        # Create a test session
        test_session = session_manager.session_service.create_session(
            app_name=TEST_APP_NAME,
            user_id=self.user_id
        )
        
        # Create state manager with the test session
        state_manager = StateManager(session_manager, self.user_id, test_session.id)
        
        # Create agent with the session and state managers
        agent = self.TestAgent(
            session_manager=session_manager,
            memory_manager=MemoryManager()
        )
        
        # Initialize state manager
        import asyncio
        asyncio.run(state_manager.initialize())
        
        # Test basic state operations
        test_key = "test_state"
        test_value = "test_value"
        
        # Set state
        asyncio.run(state_manager.set(test_key, test_value))
        
        # Get state and verify
        retrieved_value = asyncio.run(state_manager.get(test_key))
        self.assertEqual(retrieved_value, test_value)
        
        # Test agent can access state through context
        query = f"Get state for {test_key}"
        response = self._run_agent(agent, query)
        self.assertIsNotNone(response)
        self.assertIn(test_value.lower(), response.lower())


if __name__ == "__main__":
    unittest.main()
