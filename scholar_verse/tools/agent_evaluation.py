"""Agent evaluation module for ScholarVerse.

This module provides agent evaluation capabilities for the ScholarVerse router agent,
allowing it to evaluate the performance of sub-agents and make routing decisions
based on historical performance.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from google.adk.tools import ToolContext

from scholar_verse.shared_libraries.logging_utils import logger
from scholar_verse.shared_libraries.state_management.adaptive_state import AdaptiveStateManager
from scholar_verse.tools.feedback_processing import FeedbackProcessor


class AgentEvaluator:
    """Agent evaluator for ScholarVerse agents.
    
    This class provides methods for evaluating the performance of sub-agents
    and making routing decisions based on historical performance.
    """
    
    def __init__(
        self, 
        state_manager: Optional[AdaptiveStateManager] = None,
        feedback_processor: Optional[FeedbackProcessor] = None,
    ):
        """Initialize the agent evaluator.
        
        Args:
            state_manager: The state manager to use for storing evaluation results.
            feedback_processor: The feedback processor to use for processing feedback.
        """
        self.state_manager = state_manager or AdaptiveStateManager()
        self.feedback_processor = feedback_processor or FeedbackProcessor(self.state_manager)
    
    def evaluate_agent_performance(
        self,
        agent_name: str,
        tool_context: Optional[ToolContext] = None,
    ) -> Dict[str, Any]:
        """Evaluate the performance of an agent.
        
        Args:
            agent_name: The name of the agent to evaluate.
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A dictionary containing the evaluation results.
        """
        logger.info(f"Evaluating performance of agent: {agent_name}")
        
        # Get agent performance metrics
        agent_metrics = self.feedback_processor.get_agent_performance(
            agent_name, tool_context)
        
        # Calculate evaluation score (simple average of available metrics)
        score = 0.0
        score_components = []
        
        # Consider success rate if available
        if 'success_rate' in agent_metrics:
            score_components.append(agent_metrics['success_rate'])
        
        # Consider average rating if available
        if 'average_rating' in agent_metrics:
            # Normalize rating from 1-5 to 0-1
            normalized_rating = (agent_metrics['average_rating'] - 1) / 4
            score_components.append(normalized_rating)
        
        # Calculate overall score
        if score_components:
            score = sum(score_components) / len(score_components)
        
        # Create evaluation result
        evaluation = {
            'agent': agent_name,
            'score': score,
            'metrics': agent_metrics,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Store evaluation in state
        if tool_context:
            state_manager = tool_context.state.get('state_manager', self.state_manager)
            current_state = state_manager.get(tool_context)
            
            # Add evaluations to state if it doesn't exist
            if 'evaluations' not in current_state:
                current_state['evaluations'] = {}
            
            # Add agent evaluations if it doesn't exist
            if 'agent_evaluations' not in current_state['evaluations']:
                current_state['evaluations']['agent_evaluations'] = {}
            
            # Add evaluation to state
            current_state['evaluations']['agent_evaluations'][agent_name] = evaluation
            state_manager.set(tool_context, current_state)
        
        return evaluation
    
    def evaluate_all_agents(
        self,
        tool_context: Optional[ToolContext] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate the performance of all agents.
        
        Args:
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A dictionary mapping agent names to evaluation results.
        """
        logger.info("Evaluating performance of all agents")
        
        # Get state manager and current state
        if tool_context:
            state_manager = tool_context.state.get('state_manager', self.state_manager)
            current_state = state_manager.get(tool_context)
        else:
            current_state = self.state_manager.get(None)
        
        # Get all agent names from performance metrics
        agent_names = list(current_state['feedback']['agent_performance'].keys())
        
        # Evaluate each agent
        evaluations = {}
        for agent_name in agent_names:
            evaluation = self.evaluate_agent_performance(agent_name, tool_context)
            evaluations[agent_name] = evaluation
        
        return evaluations
    
    def recommend_agent(
        self,
        request: str,
        candidate_agents: List[str],
        tool_context: Optional[ToolContext] = None,
    ) -> str:
        """Recommend an agent for a given request.
        
        Args:
            request: The user request.
            candidate_agents: A list of candidate agent names.
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            The name of the recommended agent.
        """
        logger.info(f"Recommending agent for request: {request}")
        
        # If only one candidate, return it
        if len(candidate_agents) == 1:
            return candidate_agents[0]
        
        # Evaluate all candidate agents
        evaluations = {}
        for agent_name in candidate_agents:
            evaluation = self.evaluate_agent_performance(agent_name, tool_context)
            evaluations[agent_name] = evaluation
        
        # Sort agents by score (descending)
        sorted_agents = sorted(
            evaluations.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        # Return the highest-scoring agent
        if sorted_agents:
            return sorted_agents[0][0]
        
        # If no evaluations, return the first candidate
        return candidate_agents[0]
