"""PDF processing tools for ScholarVerse agents."""

import os
from datetime import datetime, UTC
from typing import Dict, Any, Optional
from pathlib import Path

import pypdf
from google.adk import Tool, ToolParameter, ToolParameterType

from scholar_verse.shared_libraries.logging_utils import logger


class ExtractPdfTextTool(Tool):
    """Tool for extracting text from PDF documents.
    
    This tool extracts text and basic metadata from PDF documents.
    It will be expanded in Phase 3 to include more sophisticated extraction capabilities.
    """
    
    def __init__(self):
        """Initialize the PDF text extraction tool."""
        super().__init__(
            name="extract_pdf_text",
            description="Extracts text and metadata from PDF documents",
            parameters=[
                ToolParameter(
                    name="pdf_path",
                    description="Path to the PDF file",
                    type=ToolParameterType.STRING,
                    required=True,
                ),
                ToolParameter(
                    name="extract_metadata",
                    description="Whether to extract metadata",
                    type=ToolParameterType.BOOLEAN,
                    required=False,
                ),
            ],
        )
        logger.info("Initialized PDF text extraction tool")
    
    def execute(self, pdf_path: str, extract_metadata: bool = True) -> Dict[str, Any]:
        """Execute the PDF text extraction tool.
        
        Args:
            pdf_path: Path to the PDF file.
            extract_metadata: Whether to extract metadata.
            
        Returns:
            A dictionary containing the extracted text and metadata.
        """
        try:
            logger.info(f"Extracting text from PDF: {pdf_path}")
            
            # Check if the file exists
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return {"error": f"PDF file not found: {pdf_path}"}
            
            # Open the PDF file
            with open(pdf_path, "rb") as file:
                reader = pypdf.PdfReader(file)
                
                # Extract text
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n\n"
                
                # Extract metadata if requested
                metadata = {}
                if extract_metadata and reader.metadata:
                    metadata = {
                        "title": reader.metadata.title,
                        "author": reader.metadata.author,
                        "subject": reader.metadata.subject,
                        "creator": reader.metadata.creator,
                        "producer": reader.metadata.producer,
                        "page_count": len(reader.pages),
                    }
                
                return {
                    "text": text,
                    "metadata": metadata,
                    "page_count": len(reader.pages),
                }
        
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return {"error": str(e)}


# Create an instance of the tool
extract_pdf_text_tool = ExtractPdfTextTool()
