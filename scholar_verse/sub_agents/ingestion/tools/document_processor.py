"""Document processor for the ScholarVerse Ingestion Agent.

This module provides a unified interface for processing academic documents.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import os
from datetime import datetime

from google.adk.tools import ToolContext

from scholar_verse.shared_libraries.logging_utils import logger
from scholar_verse.shared_libraries.state_management.adaptive_state import AdaptiveStateManager
from scholar_verse.sub_agents.ingestion.tools.pdf_extraction import PDFExtractor
from scholar_verse.sub_agents.ingestion.tools.quality_analysis import DocumentQualityAnalyzer
from scholar_verse.sub_agents.ingestion.tools.extraction_improvement import ExtractionImprover


class DocumentProcessor:
    """Document processor for ScholarVerse.
    
    This class provides a unified interface for processing academic documents,
    including extraction, quality analysis, and improvement.
    """
    
    def __init__(self, state_manager: Optional[AdaptiveStateManager] = None):
        """Initialize the document processor.
        
        Args:
            state_manager: The state manager to use for storing processing results.
        """
        self.state_manager = state_manager or AdaptiveStateManager()
        self.pdf_extractor = PDFExtractor(self.state_manager)
        self.quality_analyzer = DocumentQualityAnalyzer(self.state_manager)
        self.extraction_improver = ExtractionImprover(self.state_manager)
    
    def process_document(self, document_path: str, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """Process an academic document.
        
        Args:
            document_path: Path to the document to process.
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A dictionary containing the processing results.
        """
        logger.info(f"Processing document: {document_path}")
        
        # Validate document path
        if not os.path.exists(document_path):
            error_msg = f"Document not found: {document_path}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        # Get document type from file extension
        file_ext = Path(document_path).suffix.lower()
        
        # Currently only supporting PDF documents
        if file_ext != '.pdf':
            error_msg = f"Unsupported document type: {file_ext}. Only PDF documents are currently supported."
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        try:
            # Get state manager from context if available
            if tool_context:
                state_manager = tool_context.state.get('state_manager', self.state_manager)
            else:
                state_manager = self.state_manager
            
            # Get extraction parameters with improvements
            extraction_params = self.extraction_improver.apply_extraction_improvements(
                {"use_ocr": False, "enhance_resolution": False},
                tool_context
            )
            
            # Extract text from document
            extraction_result = self.pdf_extractor.extract_text(document_path, tool_context)
            if not extraction_result.get('success', False):
                return extraction_result
            
            # Extract metadata from document
            metadata_result = self.pdf_extractor.extract_metadata(document_path, tool_context)
            extraction_result['metadata'] = metadata_result
            
            # Analyze document quality
            quality_result = self.quality_analyzer.analyze_document_quality(extraction_result, tool_context)
            
            # Generate document ID
            document_id = Path(document_path).stem
            
            # Create processing result
            processing_result = {
                "success": True,
                "document_id": document_id,
                "document_path": document_path,
                "extraction": {
                    "text": extraction_result.get('success', False),
                    "metadata": metadata_result.get('success', False),
                },
                "quality": {
                    "score": quality_result.get('quality_score', 0.0),
                    "issues": quality_result.get('issues', []),
                    "recommendations": quality_result.get('recommendations', []),
                },
                "metadata": {
                    "title": metadata_result.get('title', 'Unknown Title'),
                    "authors": metadata_result.get('authors', []),
                    "abstract": metadata_result.get('abstract', ''),
                    "doi": metadata_result.get('doi', ''),
                    "publication_date": metadata_result.get('publication_date', ''),
                    "keywords": metadata_result.get('keywords', []),
                },
                "processing_time": datetime.now().isoformat(),
            }
            
            # Store the processing result in the state
            if tool_context:
                current_state = state_manager.get(tool_context)
                
                # Add processing result to state
                if 'documents' not in current_state:
                    current_state['documents'] = {}
                
                current_state['documents'][document_id] = {
                    "processing_result": processing_result,
                    "processed_at": datetime.now().isoformat(),
                }
                
                # Update the state
                state_manager.set(tool_context, current_state)
            
            return processing_result
            
        except Exception as e:
            error_msg = f"Error processing document: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def process_multiple_documents(self, document_paths: List[str], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """Process multiple academic documents.
        
        Args:
            document_paths: List of paths to documents to process.
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A dictionary containing the processing results for all documents.
        """
        logger.info(f"Processing {len(document_paths)} documents")
        
        results = {}
        successful = 0
        failed = 0
        
        for document_path in document_paths:
            result = self.process_document(document_path, tool_context)
            document_id = Path(document_path).stem
            results[document_id] = result
            
            if result.get('success', False):
                successful += 1
            else:
                failed += 1
        
        return {
            "success": True,
            "total": len(document_paths),
            "successful": successful,
            "failed": failed,
            "results": results,
        }
    
    def get_document_summary(self, document_id: str, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """Get a summary of a processed document.
        
        Args:
            document_id: ID of the document to summarize.
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A dictionary containing the document summary.
        """
        logger.info(f"Getting summary for document: {document_id}")
        
        # Get state manager from context if available
        if tool_context:
            state_manager = tool_context.state.get('state_manager', self.state_manager)
        else:
            state_manager = self.state_manager
        
        # Get the current state
        current_state = state_manager.get(tool_context) if tool_context else {}
        
        # Get document from state
        documents = current_state.get('documents', {})
        if document_id not in documents:
            return {"success": False, "error": f"Document not found: {document_id}"}
        
        document = documents[document_id]
        processing_result = document.get('processing_result', {})
        
        # Create document summary
        summary = {
            "success": True,
            "document_id": document_id,
            "title": processing_result.get('metadata', {}).get('title', 'Unknown Title'),
            "authors": processing_result.get('metadata', {}).get('authors', []),
            "abstract": processing_result.get('metadata', {}).get('abstract', ''),
            "quality_score": processing_result.get('quality', {}).get('score', 0.0),
            "processed_at": document.get('processed_at', ''),
        }
        
        return summary
