"""Feedback handling for ScholarVerse agents."""

from datetime import datetime

from typing import Dict, Any, Callable, Optional

from scholar_verse.shared_libraries.logging_utils import logger


class FeedbackHandler:
    """Feedback handler for ScholarVerse agents.
    
    This class provides mechanisms for collecting, processing, and responding to
    feedback from users and other agents. It will be expanded in later phases to
    include more sophisticated feedback handling capabilities.
    """
    
    def __init__(self):
        """Initialize the feedback handler."""
        self.feedback_store = []
        logger.info("Initialized feedback handler")
    
    def record_feedback(self, feedback: Dict[str, Any]) -> None:
        """Record feedback from a user or agent.
        
        Args:
            feedback: The feedback to record.
        """
        # Add timestamp to feedback
        feedback["timestamp"] = feedback.get("timestamp", None) or datetime.now().isoformat()
        
        # Store the feedback
        self.feedback_store.append(feedback)
        
        # Log the feedback
        logger.info(f"Recorded feedback: {feedback}")
    
    def get_feedback(self, agent_name: Optional[str] = None) -> list:
        """Get feedback for a specific agent or all feedback.
        
        Args:
            agent_name: The name of the agent to get feedback for, or None for all feedback.
            
        Returns:
            A list of feedback items.
        """
        if agent_name is None:
            return self.feedback_store
        
        return [f for f in self.feedback_store if f.get("agent_name") == agent_name]
    
    def process_feedback(self, callback: Optional[Callable] = None) -> None:
        """Process feedback using an optional callback function.
        
        Args:
            callback: A callback function to process the feedback.
        """
        if callback is None:
            # Default processing - just log the feedback
            for feedback in self.feedback_store:
                logger.info(f"Processing feedback: {feedback}")
        else:
            # Use the provided callback to process the feedback
            for feedback in self.feedback_store:
                callback(feedback)
        
        logger.info(f"Processed {len(self.feedback_store)} feedback items")


# Create a default feedback handler
feedback_handler = FeedbackHandler()
