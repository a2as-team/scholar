"""Extraction improvement tools for the ScholarVerse Ingestion Agent.

This module provides tools for improving extraction capabilities based on feedback.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from google.adk.tools import ToolContext

from scholar_verse.shared_libraries.logging_utils import logger
from scholar_verse.shared_libraries.state_management.adaptive_state import AdaptiveStateManager


class ExtractionImprover:
    """Extraction improver for ScholarVerse.
    
    This class provides methods for improving extraction capabilities based on feedback
    and historical performance.
    """
    
    def __init__(self, state_manager: Optional[AdaptiveStateManager] = None):
        """Initialize the extraction improver.
        
        Args:
            state_manager: The state manager to use for storing improvement data.
        """
        self.state_manager = state_manager or AdaptiveStateManager()
    
    def process_extraction_feedback(self, document_id: str, feedback: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """Process feedback on extraction quality to improve future extractions.
        
        Args:
            document_id: The ID of the document the feedback is for.
            feedback: The feedback data, including ratings and comments.
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A dictionary containing the processed feedback and improvement actions.
        """
        logger.info(f"Processing extraction feedback for document: {document_id}")
        
        # Validate feedback
        if not feedback.get('ratings'):
            return {"success": False, "error": "No ratings provided in feedback"}
        
        # Get state manager from context if available
        if tool_context:
            state_manager = tool_context.state.get('state_manager', self.state_manager)
        else:
            state_manager = self.state_manager
        
        # Get the current state
        current_state = state_manager.get(tool_context) if tool_context else {}
        
        # Initialize extraction_feedback in state if not present
        if 'extraction_feedback' not in current_state:
            current_state['extraction_feedback'] = {}
        
        # Add feedback to state
        feedback_entry = {
            "document_id": document_id,
            "ratings": feedback.get('ratings', {}),
            "comments": feedback.get('comments', ""),
            "timestamp": datetime.now().isoformat(),
        }
        
        if document_id not in current_state['extraction_feedback']:
            current_state['extraction_feedback'][document_id] = []
        
        current_state['extraction_feedback'][document_id].append(feedback_entry)
        
        # Update extraction parameters based on feedback
        improvement_actions = self._generate_improvement_actions(feedback, current_state)
        
        # Add improvement actions to feedback entry
        feedback_entry["improvement_actions"] = improvement_actions
        
        # Update the state
        if tool_context:
            state_manager.set(tool_context, current_state)
        
        return {
            "success": True,
            "document_id": document_id,
            "feedback_processed": feedback_entry,
            "improvement_actions": improvement_actions,
        }
    
    def _generate_improvement_actions(self, feedback: Dict[str, Any], current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate improvement actions based on feedback.
        
        Args:
            feedback: The feedback data.
            current_state: The current state.
            
        Returns:
            A list of improvement actions.
        """
        actions = []
        ratings = feedback.get('ratings', {})
        
        # Check text extraction rating
        if 'text_extraction' in ratings and ratings['text_extraction'] < 3:
            actions.append({
                "action": "improve_text_extraction",
                "description": "Enhance text extraction for low-quality PDFs",
                "parameters": {
                    "use_ocr": True,
                    "enhance_resolution": True,
                }
            })
        
        # Check metadata extraction rating
        if 'metadata_extraction' in ratings and ratings['metadata_extraction'] < 3:
            actions.append({
                "action": "improve_metadata_extraction",
                "description": "Enhance metadata extraction algorithms",
                "parameters": {
                    "use_external_sources": True,
                    "expand_search_patterns": True,
                }
            })
        
        # Check structure analysis rating
        if 'structure_analysis' in ratings and ratings['structure_analysis'] < 3:
            actions.append({
                "action": "improve_structure_analysis",
                "description": "Enhance document structure analysis",
                "parameters": {
                    "detect_sections": True,
                    "identify_figures": True,
                }
            })
        
        # Add general improvements based on comments
        comments = feedback.get('comments', "").lower()
        if "ocr" in comments or "scan" in comments:
            actions.append({
                "action": "enhance_ocr",
                "description": "Improve OCR capabilities for scanned documents",
                "parameters": {
                    "use_advanced_ocr": True,
                }
            })
        
        if "table" in comments or "figure" in comments:
            actions.append({
                "action": "enhance_table_extraction",
                "description": "Improve table and figure extraction",
                "parameters": {
                    "detect_tables": True,
                    "extract_table_data": True,
                }
            })
        
        return actions
    
    def apply_extraction_improvements(self, extraction_params: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """Apply extraction improvements based on historical feedback.
        
        Args:
            extraction_params: The current extraction parameters.
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A dictionary containing the updated extraction parameters.
        """
        logger.info("Applying extraction improvements based on historical feedback")
        
        # Get state manager from context if available
        if tool_context:
            state_manager = tool_context.state.get('state_manager', self.state_manager)
        else:
            state_manager = self.state_manager
        
        # Get the current state
        current_state = state_manager.get(tool_context) if tool_context else {}
        
        # Get historical feedback
        extraction_feedback = current_state.get('extraction_feedback', {})
        
        # If no historical feedback, return original params
        if not extraction_feedback:
            return extraction_params
        
        # Collect all improvement actions from feedback
        all_actions = []
        for document_id, feedback_entries in extraction_feedback.items():
            for entry in feedback_entries:
                all_actions.extend(entry.get('improvement_actions', []))
        
        # Apply improvements to extraction parameters
        updated_params = extraction_params.copy()
        
        # Count action frequencies to prioritize common improvements
        action_counts = {}
        for action in all_actions:
            action_type = action.get('action')
            if action_type:
                action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        # Apply the most frequent improvements
        for action_type, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            # Find the most recent action of this type
            for action in reversed(all_actions):
                if action.get('action') == action_type:
                    # Apply parameters from this action
                    params = action.get('parameters', {})
                    for key, value in params.items():
                        updated_params[key] = value
                    break
        
        # Log the improvements
        improvements = [f"{key}: {updated_params[key]}" for key in updated_params if key not in extraction_params or updated_params[key] != extraction_params[key]]
        if improvements:
            logger.info(f"Applied extraction improvements: {', '.join(improvements)}")
        
        return updated_params
    
    def get_extraction_performance_metrics(self, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """Get performance metrics for the extraction process based on feedback.
        
        Args:
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A dictionary containing performance metrics.
        """
        logger.info("Getting extraction performance metrics")
        
        # Get state manager from context if available
        if tool_context:
            state_manager = tool_context.state.get('state_manager', self.state_manager)
        else:
            state_manager = self.state_manager
        
        # Get the current state
        current_state = state_manager.get(tool_context) if tool_context else {}
        
        # Get historical feedback
        extraction_feedback = current_state.get('extraction_feedback', {})
        
        # Initialize metrics
        metrics = {
            "total_documents": len(extraction_feedback),
            "total_feedback_entries": 0,
            "average_ratings": {},
            "improvement_actions": {},
        }
        
        # Calculate metrics
        all_ratings = {}
        all_actions = []
        
        for document_id, feedback_entries in extraction_feedback.items():
            metrics["total_feedback_entries"] += len(feedback_entries)
            
            for entry in feedback_entries:
                # Collect ratings
                ratings = entry.get('ratings', {})
                for rating_type, rating_value in ratings.items():
                    if rating_type not in all_ratings:
                        all_ratings[rating_type] = []
                    all_ratings[rating_type].append(rating_value)
                
                # Collect improvement actions
                all_actions.extend(entry.get('improvement_actions', []))
        
        # Calculate average ratings
        for rating_type, ratings in all_ratings.items():
            metrics["average_ratings"][rating_type] = sum(ratings) / len(ratings) if ratings else 0
        
        # Count improvement actions
        for action in all_actions:
            action_type = action.get('action')
            if action_type:
                metrics["improvement_actions"][action_type] = metrics["improvement_actions"].get(action_type, 0) + 1
        
        return metrics
