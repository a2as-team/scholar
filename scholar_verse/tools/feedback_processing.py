"""Feedback processing module for ScholarVerse.

This module provides feedback processing capabilities for the ScholarVerse router agent,
allowing it to collect and process feedback from users and other agents to improve
performance and adapt to changing requirements.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from google.adk.tools import ToolContext

from scholar_verse.shared_libraries.logging_utils import logger
from scholar_verse.shared_libraries.state_management.adaptive_state import AdaptiveStateManager


class FeedbackProcessor:
    """Feedback processor for ScholarVerse agents.
    
    This class provides methods for collecting and processing feedback from users
    and other agents to improve performance and adapt to changing requirements.
    """
    
    def __init__(self, state_manager: Optional[AdaptiveStateManager] = None):
        """Initialize the feedback processor.
        
        Args:
            state_manager: The state manager to use for storing feedback.
        """
        self.state_manager = state_manager or AdaptiveStateManager()
    
    def process_user_feedback(
        self,
        feedback: str,
        rating: int,
        feedback_type: str,
        agent_name: Optional[str] = None,
        tool_context: Optional[ToolContext] = None,
    ) -> Dict[str, Any]:
        """Process user feedback.
        
        Args:
            feedback: The user feedback text.
            rating: The user rating (1-5).
            feedback_type: The type of feedback (e.g., 'response_quality', 'accuracy', 'usefulness').
            agent_name: The name of the agent the feedback is for (optional).
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A dictionary containing the processed feedback.
        """
        logger.info(f"Processing user feedback: {feedback_type} - {rating}/5")
        
        # Validate rating
        if not 1 <= rating <= 5:
            logger.warning(f"Invalid rating: {rating}, must be between 1 and 5")
            rating = max(1, min(rating, 5))  # Clamp to 1-5 range
        
        # Create feedback object
        feedback_obj = {
            'text': feedback,
            'rating': rating,
            'type': feedback_type,
            'agent': agent_name,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Add feedback to state
        if tool_context:
            # Get state manager from context if available
            state_manager = tool_context.state.get('state_manager', self.state_manager)
            current_state = state_manager.get(tool_context)
            
            # Add feedback to state
            state_manager.add_user_feedback(feedback_obj)
            
            # Update agent performance metrics if applicable
            if agent_name:
                # Get current metrics or initialize new ones
                agent_metrics = current_state['feedback']['agent_performance'].get(agent_name, {})
                
                # Update metrics
                if 'ratings' not in agent_metrics:
                    agent_metrics['ratings'] = []
                agent_metrics['ratings'].append(rating)
                
                # Calculate average rating
                agent_metrics['average_rating'] = sum(agent_metrics['ratings']) / len(agent_metrics['ratings'])
                
                # Update performance metrics
                state_manager.update_agent_performance(agent_name, agent_metrics)
        else:
            # Add feedback directly to state manager
            self.state_manager.add_user_feedback(feedback_obj)
        
        return feedback_obj
    
    def process_agent_feedback(
        self,
        source_agent: str,
        target_agent: str,
        feedback: str,
        success: bool,
        metrics: Optional[Dict[str, Any]] = None,
        tool_context: Optional[ToolContext] = None,
    ) -> Dict[str, Any]:
        """Process feedback from one agent about another.
        
        Args:
            source_agent: The name of the agent providing the feedback.
            target_agent: The name of the agent the feedback is about.
            feedback: The feedback text.
            success: Whether the interaction was successful.
            metrics: Additional metrics about the interaction (optional).
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A dictionary containing the processed feedback.
        """
        logger.info(f"Processing agent feedback from {source_agent} about {target_agent}")
        
        # Create feedback object
        feedback_obj = {
            'source_agent': source_agent,
            'target_agent': target_agent,
            'text': feedback,
            'success': success,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat(),
        }
        
        # Add feedback to state
        if tool_context:
            # Get state manager from context if available
            state_manager = tool_context.state.get('state_manager', self.state_manager)
            current_state = state_manager.get(tool_context)
            
            # Add feedback to agent_feedback if it doesn't exist
            if 'agent_feedback' not in current_state['feedback']:
                current_state['feedback']['agent_feedback'] = []
            
            # Add feedback to state
            current_state['feedback']['agent_feedback'].append(feedback_obj)
            state_manager.set(tool_context, current_state)
            
            # Update agent performance metrics
            agent_metrics = current_state['feedback']['agent_performance'].get(target_agent, {})
            
            # Update success rate
            if 'success_count' not in agent_metrics:
                agent_metrics['success_count'] = 0
                agent_metrics['total_count'] = 0
            
            if success:
                agent_metrics['success_count'] += 1
            agent_metrics['total_count'] += 1
            
            # Calculate success rate
            agent_metrics['success_rate'] = agent_metrics['success_count'] / agent_metrics['total_count']
            
            # Add any additional metrics
            if metrics:
                for key, value in metrics.items():
                    agent_metrics[key] = value
            
            # Update performance metrics
            state_manager.update_agent_performance(target_agent, agent_metrics)
        else:
            # Add feedback directly to state manager
            current_state = self.state_manager.get(None)
            
            # Add feedback to agent_feedback if it doesn't exist
            if 'agent_feedback' not in current_state['feedback']:
                current_state['feedback']['agent_feedback'] = []
            
            # Add feedback to state
            current_state['feedback']['agent_feedback'].append(feedback_obj)
            self.state_manager.set(None, current_state)
        
        return feedback_obj
    
    def get_agent_performance(
        self,
        agent_name: str,
        tool_context: Optional[ToolContext] = None,
    ) -> Dict[str, Any]:
        """Get performance metrics for an agent.
        
        Args:
            agent_name: The name of the agent to get metrics for.
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A dictionary containing the agent's performance metrics.
        """
        logger.info(f"Getting performance metrics for agent: {agent_name}")
        
        # Get state manager and current state
        if tool_context:
            state_manager = tool_context.state.get('state_manager', self.state_manager)
            current_state = state_manager.get(tool_context)
        else:
            current_state = self.state_manager.get(None)
        
        # Get agent performance metrics
        agent_metrics = current_state['feedback']['agent_performance'].get(agent_name, {})
        
        return agent_metrics
    
    def get_user_feedback(
        self,
        feedback_type: Optional[str] = None,
        agent_name: Optional[str] = None,
        tool_context: Optional[ToolContext] = None,
    ) -> List[Dict[str, Any]]:
        """Get user feedback.
        
        Args:
            feedback_type: The type of feedback to filter by (optional).
            agent_name: The name of the agent to filter by (optional).
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A list of feedback objects matching the filters.
        """
        logger.info("Getting user feedback")
        
        # Get state manager and current state
        if tool_context:
            state_manager = tool_context.state.get('state_manager', self.state_manager)
            current_state = state_manager.get(tool_context)
        else:
            current_state = self.state_manager.get(None)
        
        # Get all user feedback
        all_feedback = current_state['feedback']['user_feedback']
        
        # Filter by type and agent if specified
        filtered_feedback = []
        for feedback in all_feedback:
            if feedback_type and feedback['type'] != feedback_type:
                continue
            if agent_name and feedback.get('agent') != agent_name:
                continue
            filtered_feedback.append(feedback)
        
        return filtered_feedback
