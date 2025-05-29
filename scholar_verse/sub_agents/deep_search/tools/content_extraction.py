"""Content Extraction Tool for Deep Search Agent.

This module provides tools for extracting and processing content from scholarly sources.
"""

from typing import Dict, Any, List, Optional, Type, Union
import re
import json
import logging
from datetime import datetime

# Import ADK tooling with proper error handling
try:
    from google.adk.tools import BaseTool, ToolContext, ToolParameters
    HAS_ADK = True
except ImportError:
    # For local development without ADK
    HAS_ADK = False
    
    class BaseTool:
        def __init__(self, name: str, description: str, parameters: Type['ToolParameters'] = None):
            self.name = name
            self.description = description
            self.parameters = parameters or type('ToolParameters', (), {})
        
        def _call(self, context: 'ToolContext', **kwargs) -> Any:
            raise NotImplementedError("Tool must implement _call method")
    
    class ToolContext:
        def __init__(self):
            self.state = {}
    
    class ToolParameters:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

# Import NLTK with proper error handling
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass  # Already downloaded or offline
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    
    # Fallback sentence tokenizer
    def sent_tokenize(text):
        # Simple sentence tokenization by splitting on periods, exclamation marks, and question marks
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

# Import Vertex AI with proper error handling
try:
    from vertexai.language_models import TextEmbeddingModel
    HAS_VERTEX_AI = True
except ImportError:
    HAS_VERTEX_AI = False
    
    class TextEmbeddingModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("Vertex AI TextEmbeddingModel is not available")

from scholar_verse.shared_libraries.logging_utils import logger
from scholar_verse.config import get_config

# Export the tool class
__all__ = ['ContentExtractionTool']


class ContentExtractionTool(BaseTool):
    """Tool for extracting and processing content from scholarly sources."""
    
    def __init__(self):
        """Initialize the content extraction tool."""
        # Define parameters schema
        class ContentExtractionParameters(ToolParameters):
            content: Union[str, List[Dict[str, Any]]]
            extraction_type: str = "summary"  # summary, entities, key_points, citations
            max_length: int = 1000
            include_metadata: bool = False
        
        super().__init__(
            name="content_extraction",
            description="Extracts and processes content from scholarly sources",
            parameters=ContentExtractionParameters
        )
        
        self.config = get_config()
        
        # Initialize Vertex AI if available
        if HAS_VERTEX_AI and self.config.get('vertex_ai', {}).get('enabled', False):
            try:
                self.embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
            except Exception as e:
                logger.warning(f"Failed to initialize Vertex AI embedding model: {str(e)}")
    
    def _call(self, context: ToolContext, content: str, extraction_type: str = "key_information",
             metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the content extraction tool.
        
        Args:
            context: The tool context.
            content: The content to process.
            extraction_type: The type of extraction to perform (key_information, entities, summary, etc.).
            metadata: Additional metadata about the content.
            
        Returns:
            A dictionary containing the extracted content.
        """
        logger.info(f"Extracting {extraction_type} from content of length {len(content)}")
        
        try:
            # Initialize metadata if not provided
            metadata = metadata or {}
            
            # Process based on extraction type
            if extraction_type == "key_information":
                result = self._extract_key_information(content, metadata)
            elif extraction_type == "entities":
                result = self._extract_entities(content, metadata)
            elif extraction_type == "summary":
                result = self._extract_summary(content, metadata)
            elif extraction_type == "citations":
                result = self._extract_citations(content, metadata)
            else:
                # Default to key information
                result = self._extract_key_information(content, metadata)
            
            return {
                'success': True,
                'extraction_type': extraction_type,
                'result': result,
                'content_length': len(content),
                'metadata': metadata,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error extracting {extraction_type} from content: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'extraction_type': extraction_type
            }
    def __call__(self, context: ToolContext, **kwargs) -> Any:
        return self._call(context, **kwargs)
    def _extract_key_information(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key information from content.
        
        Args:
            content: The content to process.
            metadata: Additional metadata about the content.
            
        Returns:
            A dictionary containing the extracted key information.
        """
        # In a real implementation, this would use NLP techniques to extract key information
        # For now, we'll implement a simpler approach
        
        # Tokenize content into sentences
        sentences = sent_tokenize(content)
        
        # Extract potentially important sentences (simplified approach)
        important_sentences = []
        key_phrases = ["important", "significant", "key", "essential", "fundamental", "critical"]
        academic_phrases = ["research", "study", "paper", "analysis", "theory", "framework", "method"]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Check for sentences that might contain key information
            if any(phrase in sentence_lower for phrase in key_phrases) or \
               any(phrase in sentence_lower for phrase in academic_phrases):
                important_sentences.append(sentence)
        
        # Identify potential key terms (simplified approach)
        term_pattern = r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)'  # Look for capitalized multi-word phrases
        potential_terms = re.findall(term_pattern, content)
        
        # Deduplicate and limit terms
        unique_terms = list(set(potential_terms))[:10]  # Limit to 10 terms
        
        return {
            'important_sentences': important_sentences[:5],  # Limit to 5 sentences
            'key_terms': unique_terms,
            'content_type': metadata.get('content_type', 'unknown'),
            'extraction_method': 'pattern_matching'
        }
    
    def _extract_entities(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities from content.
        
        Args:
            content: The content to process.
            metadata: Additional metadata about the content.
            
        Returns:
            A dictionary containing the extracted entities.
        """
        # In a real implementation, this would use NER models to extract entities
        # For now, we'll simulate entity extraction
        
        # Simple patterns for different entity types
        person_pattern = r'([A-Z][a-z]+\s[A-Z][a-z]+(?:\s[A-Z][a-z]+)?)'  # Names like "John Smith"
        org_pattern = r'([A-Z][a-z]*(?:\s[A-Z][a-z]+)+\s(?:University|Institute|Organization|Association))'  # Orgs like "Stanford University"
        date_pattern = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2},\s\d{4}|\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{4})'  # Dates
        
        # Extract using patterns
        persons = list(set(re.findall(person_pattern, content)))[:10]
        organizations = list(set(re.findall(org_pattern, content)))[:10]
        dates = list(set(re.findall(date_pattern, content)))[:10]
        
        return {
            'persons': persons,
            'organizations': organizations,
            'dates': dates,
            'content_type': metadata.get('content_type', 'unknown'),
            'extraction_method': 'pattern_matching'
        }
    
    def _extract_summary(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of content.
        
        Args:
            content: The content to summarize.
            metadata: Additional metadata about the content.
            
        Returns:
            A dictionary containing the summary.
        """
        # In a real implementation, this would use summarization models
        # For now, we'll implement a simple extractive approach
        
        # Tokenize content into sentences
        sentences = sent_tokenize(content)
        
        # Use a simple approach - take first 2-3 sentences (often contains key info in academic writing)
        intro_summary = ' '.join(sentences[:min(3, len(sentences))])
        
        # Try to extract a conclusion (often at the end)
        conclusion_summary = ''
        if len(sentences) > 5:
            conclusion_summary = ' '.join(sentences[-3:])
        
        # Calculate approximate word count
        word_count = len(content.split())
        
        return {
            'intro_summary': intro_summary,
            'conclusion_summary': conclusion_summary,
            'full_summary': intro_summary + ' [...] ' + conclusion_summary if conclusion_summary else intro_summary,
            'word_count': word_count,
            'sentence_count': len(sentences),
            'content_type': metadata.get('content_type', 'unknown'),
            'extraction_method': 'extractive_summarization'
        }
    
    def _extract_citations(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract citations from content.
        
        Args:
            content: The content to process.
            metadata: Additional metadata about the content.
            
        Returns:
            A dictionary containing the extracted citations.
        """
        # In a real implementation, this would use specialized citation parsing
        # For now, we'll implement pattern matching for common citation formats
        
        # Citation patterns
        doi_pattern = r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)'  # DOI pattern
        author_year_pattern = r'\(([A-Za-z]+(?:\set\sal\.?)?),?\s(\d{4})\)'  # (Author, 2020) pattern
        reference_pattern = r'\[([\d,\s-]+)\]'  # [1] or [1,2] pattern
        
        # Extract citations using patterns
        dois = list(set(re.findall(doi_pattern, content, re.IGNORECASE)))
        author_year_citations = re.findall(author_year_pattern, content)
        reference_citations = re.findall(reference_pattern, content)
        
        # Process author year citations
        formatted_author_year = []
        for author, year in author_year_citations:
            formatted_author_year.append(f"{author.strip()} ({year.strip()})")
        
        return {
            'dois': dois,
            'author_year_citations': formatted_author_year,
            'reference_citations': reference_citations,
            'content_type': metadata.get('content_type', 'unknown'),
            'source_url': metadata.get('source_url', 'unknown'),
            'extraction_method': 'pattern_matching',
            'total_citations_found': len(dois) + len(formatted_author_year) + len(reference_citations)
        }


# Export the tool class instead of an instance
__all__ = ['ContentExtractionTool']
