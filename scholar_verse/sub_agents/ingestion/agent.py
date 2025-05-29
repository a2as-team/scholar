"""Ingestion Agent for ScholarVerse.

This module defines the Ingestion Agent that processes PDF documents and extracts text and metadata.
"""

from typing import Dict, Any, List, Optional, Any
import os
from pathlib import Path

from google.adk import Agent
from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.function_tool import FunctionTool

from scholar_verse.config import DEFAULT_MODEL
from scholar_verse.prompt import INGESTION_AGENT_INSTRUCTIONS
from scholar_verse.shared_libraries.logging_utils import logger
from scholar_verse.shared_libraries.state_management.adaptive_state import AdaptiveStateManager

# Import ingestion tools
from scholar_verse.sub_agents.ingestion.tools.pdf_extraction import PDFExtractor
from scholar_verse.sub_agents.ingestion.tools.quality_analysis import DocumentQualityAnalyzer
from scholar_verse.sub_agents.ingestion.tools.extraction_improvement import ExtractionImprover
from scholar_verse.sub_agents.ingestion.tools.document_processor import DocumentProcessor


# Initialize tool components
pdf_extractor = PDFExtractor()
quality_analyzer = DocumentQualityAnalyzer()
extraction_improver = ExtractionImprover()
document_processor = DocumentProcessor()


# Define tool functions
def process_document(document_path: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Process a document and extract its content and metadata.
    
    Args:
        document_path: Path to the document to process.
        tool_context: The tool context.
        
    Returns:
        A dictionary containing the processing results.
    """
    logger.info(f"Processing document: {document_path}")
    
    # Initialize state manager if not already in context
    if 'state_manager' not in tool_context.state:
        tool_context.state['state_manager'] = AdaptiveStateManager()
    
    # Process the document
    result = document_processor.process_document(document_path, tool_context)
    
    return result


def process_multiple_documents(document_paths: List[str], tool_context: ToolContext) -> Dict[str, Any]:
    """Process multiple documents.
    
    Args:
        document_paths: List of paths to documents to process.
        tool_context: The tool context.
        
    Returns:
        A dictionary containing the processing results for all documents.
    """
    logger.info(f"Processing multiple documents: {len(document_paths)} documents")
    
    # Initialize state manager if not already in context
    if 'state_manager' not in tool_context.state:
        tool_context.state['state_manager'] = AdaptiveStateManager()
    
    # Process the documents
    result = document_processor.process_multiple_documents(document_paths, tool_context)
    
    return result


def extract_text(pdf_path: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Extract text from a PDF document.
    
    Args:
        pdf_path: Path to the PDF document.
        tool_context: The tool context.
        
    Returns:
        A dictionary containing the extracted text.
    """
    logger.info(f"Extracting text from PDF: {pdf_path}")
    
    # Initialize state manager if not already in context
    if 'state_manager' not in tool_context.state:
        tool_context.state['state_manager'] = AdaptiveStateManager()
    
    # Extract text
    result = pdf_extractor.extract_text(pdf_path, tool_context)
    
    return result


def extract_metadata(pdf_path: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Extract metadata from a PDF document.
    
    Args:
        pdf_path: Path to the PDF document.
        tool_context: The tool context.
        
    Returns:
        A dictionary containing the extracted metadata.
    """
    logger.info(f"Extracting metadata from PDF: {pdf_path}")
    
    # Initialize state manager if not already in context
    if 'state_manager' not in tool_context.state:
        tool_context.state['state_manager'] = AdaptiveStateManager()
    
    # Extract metadata
    result = pdf_extractor.extract_metadata(pdf_path, tool_context)
    
    return result


def analyze_document_quality(document_id: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Analyze the quality of a processed document.
    
    Args:
        document_id: ID of the document to analyze.
        tool_context: The tool context.
        
    Returns:
        A dictionary containing the quality analysis results.
    """
    logger.info(f"Analyzing document quality: {document_id}")
    
    # Initialize state manager if not already in context
    if 'state_manager' not in tool_context.state:
        tool_context.state['state_manager'] = AdaptiveStateManager()
    
    # Get the current state
    state_manager = tool_context.state['state_manager']
    current_state = state_manager.get(tool_context)
    
    # Get document from state
    documents = current_state.get('documents', {})
    if document_id not in documents:
        return {"success": False, "error": f"Document not found: {document_id}"}
    
    document = documents[document_id]
    extraction_result = document.get('extraction_result', {})
    
    # Analyze document quality
    result = quality_analyzer.analyze_document_quality(extraction_result, tool_context)
    
    return result


def provide_extraction_feedback(document_id: str, ratings: Dict[str, int], comments: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Provide feedback on extraction quality to improve future extractions.
    
    Args:
        document_id: ID of the document to provide feedback for.
        ratings: Dictionary of ratings for different aspects of extraction.
        comments: Additional comments on extraction quality.
        tool_context: The tool context.
        
    Returns:
        A dictionary containing the processed feedback and improvement actions.
    """
    logger.info(f"Providing extraction feedback for document: {document_id}")
    
    # Initialize state manager if not already in context
    if 'state_manager' not in tool_context.state:
        tool_context.state['state_manager'] = AdaptiveStateManager()
    
    # Process feedback
    feedback = {
        "ratings": ratings,
        "comments": comments,
    }
    
    result = extraction_improver.process_extraction_feedback(document_id, feedback, tool_context)
    
    return result


def get_document_summary(document_id: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Get a summary of a processed document.
    
    Args:
        document_id: ID of the document to summarize.
        tool_context: The tool context.
        
    Returns:
        A dictionary containing the document summary.
    """
    logger.info(f"Getting summary for document: {document_id}")
    
    # Initialize state manager if not already in context
    if 'state_manager' not in tool_context.state:
        tool_context.state['state_manager'] = AdaptiveStateManager()
    
    # Get document summary
    result = document_processor.get_document_summary(document_id, tool_context)
    
    return result


def get_extraction_performance(tool_context: ToolContext) -> Dict[str, Any]:
    """Get performance metrics for the extraction process.
    
    Args:
        tool_context: The tool context.
        
    Returns:
        A dictionary containing performance metrics.
    """
    logger.info("Getting extraction performance metrics")
    
    # Initialize state manager if not already in context
    if 'state_manager' not in tool_context.state:
        tool_context.state['state_manager'] = AdaptiveStateManager()
    
    # Get performance metrics
    result = extraction_improver.get_extraction_performance_metrics(tool_context)
    
    return result


# Create function tools
process_document_tool = FunctionTool(process_document)
process_multiple_documents_tool = FunctionTool(process_multiple_documents)
extract_text_tool = FunctionTool(extract_text)
extract_metadata_tool = FunctionTool(extract_metadata)
analyze_quality_tool = FunctionTool(analyze_document_quality)
provide_feedback_tool = FunctionTool(provide_extraction_feedback)
get_summary_tool = FunctionTool(get_document_summary)
get_performance_tool = FunctionTool(get_extraction_performance)


class IngestionAgent(Agent):
    """Ingestion Agent for processing and analyzing academic documents.
    
    This agent handles document ingestion, text extraction, metadata extraction,
    and quality analysis of academic documents.
    """
    
    # Define class attributes with type hints for Pydantic
    pdf_extractor: Any = None
    quality_analyzer: Any = None
    extraction_improver: Any = None
    document_processor: Any = None
    
    def __init__(self, **data):
        """Initialize the IngestionAgent with its tools and configuration."""
        # Initialize the base class first
        super().__init__(
            name="ingestion_agent",
            instruction=INGESTION_AGENT_INSTRUCTIONS,
            model=DEFAULT_MODEL,
            tools=[
                process_document_tool,
                process_multiple_documents_tool,
                extract_text_tool,
                extract_metadata_tool,
                analyze_quality_tool,
                provide_feedback_tool,
                get_summary_tool,
                get_performance_tool,
            ],
            **data
        )
        
        # Now set the instance attributes
        self.pdf_extractor = pdf_extractor
        self.quality_analyzer = quality_analyzer
        self.extraction_improver = extraction_improver
        self.document_processor = document_processor


# Create an instance of the IngestionAgent
ingestion_agent = IngestionAgent()

# Create an AgentTool from the Ingestion Agent
ingestion_agent_tool = AgentTool(agent=ingestion_agent)

# Export the IngestionAgent class and the agent instance
__all__ = ['IngestionAgent', 'ingestion_agent', 'ingestion_agent_tool']
