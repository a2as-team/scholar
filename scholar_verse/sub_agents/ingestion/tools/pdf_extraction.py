"""PDF extraction tools for the ScholarVerse Ingestion Agent.

This module provides tools for extracting text and metadata from PDF documents.
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import re
from pathlib import Path
import json
from datetime import datetime

import pypdf
from google.adk.tools import ToolContext

from scholar_verse.shared_libraries.logging_utils import logger
from scholar_verse.shared_libraries.state_management.adaptive_state import AdaptiveStateManager


class PDFExtractor:
    """PDF extraction class for processing academic documents.
    
    This class provides methods for extracting text and metadata from PDF documents,
    with support for OCR fallback for scanned documents.
    """
    
    def __init__(self, state_manager: Optional[AdaptiveStateManager] = None):
        """Initialize the PDF extractor.
        
        Args:
            state_manager: The state manager to use for storing extraction results.
        """
        self.state_manager = state_manager or AdaptiveStateManager()
    
    def extract_text(self, pdf_path: str, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """Extract text from a PDF document.
        
        Args:
            pdf_path: Path to the PDF document.
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A dictionary containing the extracted text and metadata.
        """
        logger.info(f"Extracting text from PDF: {pdf_path}")
        
        # Validate PDF path
        if not os.path.exists(pdf_path):
            error_msg = f"PDF file not found: {pdf_path}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        try:
            # Open the PDF file
            with open(pdf_path, 'rb') as file:
                # Create a PDF reader object
                reader = pypdf.PdfReader(file)
                
                # Get the number of pages
                num_pages = len(reader.pages)
                logger.info(f"PDF has {num_pages} pages")
                
                # Extract text from each page
                pages_text = []
                for i in range(num_pages):
                    page = reader.pages[i]
                    text = page.extract_text()
                    pages_text.append(text)
                
                # Create extraction result
                extraction_result = {
                    "success": True,
                    "file_path": pdf_path,
                    "file_name": Path(pdf_path).name,
                    "num_pages": num_pages,
                    "pages_text": pages_text,
                    "full_text": "\n\n".join(pages_text),
                    "extraction_method": "PyPDF2",
                    "extraction_time": datetime.now().isoformat(),
                }
                
                # Store the extraction result in the state if tool_context is provided
                if tool_context:
                    # Get state manager from context if available
                    state_manager = tool_context.state.get('state_manager', self.state_manager)
                    
                    # Get the current state
                    current_state = state_manager.get(tool_context)
                    
                    # Add extraction result to state
                    if 'documents' not in current_state:
                        current_state['documents'] = {}
                    
                    document_id = Path(pdf_path).stem
                    current_state['documents'][document_id] = {
                        "extraction_result": extraction_result,
                        "processed_at": datetime.now().isoformat(),
                    }
                    
                    # Update the state
                    state_manager.set(tool_context, current_state)
                
                return extraction_result
                
        except Exception as e:
            error_msg = f"Error extracting text from PDF: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def extract_metadata(self, pdf_path: str, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """Extract metadata from a PDF document.
        
        Args:
            pdf_path: Path to the PDF document.
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A dictionary containing the extracted metadata.
        """
        logger.info(f"Extracting metadata from PDF: {pdf_path}")
        
        # Validate PDF path
        if not os.path.exists(pdf_path):
            error_msg = f"PDF file not found: {pdf_path}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        try:
            # Open the PDF file
            with open(pdf_path, 'rb') as file:
                # Create a PDF reader object
                reader = pypdf.PdfReader(file)
                
                # Extract document info
                info = reader.metadata
                
                # Extract text for metadata analysis
                extraction_result = self.extract_text(pdf_path, tool_context)
                if not extraction_result["success"]:
                    return {"success": False, "error": extraction_result["error"]}
                
                full_text = extraction_result["full_text"]
                first_page_text = extraction_result["pages_text"][0] if extraction_result["pages_text"] else ""
                
                # Extract title (from document info or first page)
                title = info.title if info and info.title else self._extract_title(first_page_text)
                
                # Extract authors (from document info or first page)
                authors = info.author if info and info.author else self._extract_authors(first_page_text)
                
                # Extract abstract from first page
                abstract = self._extract_abstract(first_page_text)
                
                # Extract DOI
                doi = self._extract_doi(full_text)
                
                # Extract publication date
                # Convert datetime object to string if necessary
                if info and info.creation_date:
                    pub_date = info.creation_date.strftime('%Y-%m-%d') if hasattr(info.creation_date, 'strftime') else str(info.creation_date)
                else:
                    pub_date = self._extract_publication_date(first_page_text)
                
                # Create metadata result
                metadata_result = {
                    "success": True,
                    "file_path": pdf_path,
                    "file_name": Path(pdf_path).name,
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "doi": doi,
                    "publication_date": pub_date,
                    "keywords": self._extract_keywords(full_text),
                    "extraction_time": datetime.now().isoformat(),
                }
                
                # Store the metadata result in the state if tool_context is provided
                if tool_context:
                    # Get state manager from context if available
                    state_manager = tool_context.state.get('state_manager', self.state_manager)
                    
                    # Get the current state
                    current_state = state_manager.get(tool_context)
                    
                    # Add metadata result to state
                    if 'documents' not in current_state:
                        current_state['documents'] = {}
                    
                    document_id = Path(pdf_path).stem
                    if document_id in current_state['documents']:
                        current_state['documents'][document_id]["metadata"] = metadata_result
                    else:
                        current_state['documents'][document_id] = {
                            "metadata": metadata_result,
                            "processed_at": datetime.now().isoformat(),
                        }
                    
                    # Update the state
                    state_manager.set(tool_context, current_state)
                
                return metadata_result
                
        except Exception as e:
            error_msg = f"Error extracting metadata from PDF: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def _extract_title(self, text: str) -> str:
        """Extract the title from the document text.
        
        Args:
            text: The document text.
            
        Returns:
            The extracted title.
        """
        # Simple heuristic: first line that's not empty and has more than 5 words
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line.split()) > 5:
                return line
        return "Unknown Title"
    
    def _extract_authors(self, text: str) -> List[str]:
        """Extract the authors from the document text.
        
        Args:
            text: The document text.
            
        Returns:
            A list of extracted authors.
        """
        # Simple heuristic: look for lines with names after the title
        lines = text.split('\n')
        author_line = ""
        
        # Skip empty lines and find potential author line
        for i, line in enumerate(lines):
            if i > 0 and line.strip() and not line.startswith("Abstract"):
                author_line = line.strip()
                break
        
        # Split by common separators
        if author_line:
            # Try to split by common author separators
            for sep in [",", ";", "and"]:
                if sep in author_line:
                    return [author.strip() for author in author_line.split(sep) if author.strip()]
            
            # If no separators found, return the whole line as a single author
            return [author_line]
        
        return ["Unknown Author"]
    
    def _extract_abstract(self, text: str) -> str:
        """Extract the abstract from the document text.
        
        Args:
            text: The document text.
            
        Returns:
            The extracted abstract.
        """
        # Look for abstract section
        abstract_pattern = re.compile(r'(?i)abstract[:\s]*(.*?)(?:(?:\n\n)|(?:introduction)|(?:keywords))', re.DOTALL)
        match = abstract_pattern.search(text)
        
        if match:
            return match.group(1).strip()
        
        return "Abstract not found"
    
    def _extract_doi(self, text: str) -> str:
        """Extract the DOI from the document text.
        
        Args:
            text: The document text.
            
        Returns:
            The extracted DOI.
        """
        # DOI pattern
        doi_pattern = re.compile(r'(?i)(?:doi|DOI)[:\s]*(10\.\d+/[^\s]+)')
        match = doi_pattern.search(text)
        
        if match:
            return match.group(1).strip()
        
        return ""
    
    def _extract_publication_date(self, text: str) -> str:
        """Extract the publication date from the document text.
        
        Args:
            text: The document text.
            
        Returns:
            The extracted publication date.
        """
        # Look for year patterns
        year_pattern = re.compile(r'(?:19|20)\d{2}')
        match = year_pattern.search(text)
        
        if match:
            return match.group(0)
        
        return ""
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from the document text.
        
        Args:
            text: The document text.
            
        Returns:
            A list of extracted keywords.
        """
        # Look for keywords section
        keywords_pattern = re.compile(r'(?i)keywords[:\s]*(.*?)(?:\n\n)', re.DOTALL)
        match = keywords_pattern.search(text)
        
        if match:
            keywords_text = match.group(1).strip()
            # Split by common separators
            for sep in [",", ";"]:
                if sep in keywords_text:
                    return [kw.strip() for kw in keywords_text.split(sep) if kw.strip()]
            
            # If no separators found, split by space
            return [kw.strip() for kw in keywords_text.split() if kw.strip()]
        
        return []
