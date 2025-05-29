"""Document quality analysis tools for the ScholarVerse Ingestion Agent.

This module provides tools for analyzing the quality of extracted document content.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from google.adk.tools import ToolContext

from scholar_verse.shared_libraries.logging_utils import logger
from scholar_verse.shared_libraries.state_management.adaptive_state import AdaptiveStateManager


class DocumentQualityAnalyzer:
    """Document quality analyzer for ScholarVerse.
    
    This class provides methods for analyzing the quality of extracted document content
    and providing feedback for improvement.
    """
    
    def __init__(self, state_manager: Optional[AdaptiveStateManager] = None):
        """Initialize the document quality analyzer.
        
        Args:
            state_manager: The state manager to use for storing analysis results.
        """
        self.state_manager = state_manager or AdaptiveStateManager()
    
    def analyze_document_quality(self, extraction_result: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """Analyze the quality of an extracted document.
        
        Args:
            extraction_result: The extraction result to analyze.
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A dictionary containing the quality analysis results.
        """
        logger.info(f"Analyzing document quality for: {extraction_result.get('file_name', 'unknown')}")
        
        # Check if extraction was successful
        if not extraction_result.get('success', False):
            return {
                "success": False,
                "error": extraction_result.get('error', 'Unknown extraction error'),
                "quality_score": 0.0,
                "issues": ["Extraction failed"],
                "recommendations": ["Try a different extraction method"],
            }
        
        # Initialize quality metrics
        issues = []
        recommendations = []
        
        # Check text extraction quality
        full_text = extraction_result.get('full_text', '')
        pages_text = extraction_result.get('pages_text', [])
        
        # Calculate text length metrics
        total_text_length = len(full_text)
        avg_page_length = sum(len(page) for page in pages_text) / len(pages_text) if pages_text else 0
        
        # Check for empty or very short pages
        empty_pages = sum(1 for page in pages_text if len(page.strip()) < 50)
        if empty_pages > 0:
            issues.append(f"Found {empty_pages} empty or very short pages")
            recommendations.append("Check if the PDF contains scanned images that require OCR")
        
        # Check for very short overall text
        if total_text_length < 1000:
            issues.append("Extracted text is very short (less than 1000 characters)")
            recommendations.append("Verify PDF content and consider OCR if it's a scanned document")
        
        # Check for common OCR issues
        ocr_issues = self._check_ocr_issues(full_text)
        if ocr_issues:
            issues.extend(ocr_issues)
            recommendations.append("Consider using a more advanced OCR tool for better text extraction")
        
        # Check for metadata completeness
        metadata = extraction_result.get('metadata', {})
        missing_metadata = []
        
        for field in ['title', 'authors', 'abstract', 'doi', 'publication_date']:
            if not metadata.get(field):
                missing_metadata.append(field)
        
        if missing_metadata:
            issues.append(f"Missing metadata: {', '.join(missing_metadata)}")
            recommendations.append("Improve metadata extraction or manually add missing fields")
        
        # Calculate overall quality score (0.0 to 1.0)
        # Base score starts at 1.0 and is reduced for each issue
        quality_score = 1.0
        
        # Reduce score for empty pages
        if empty_pages > 0:
            quality_score -= min(0.3, empty_pages / len(pages_text))
        
        # Reduce score for short text
        if total_text_length < 1000:
            quality_score -= 0.3
        elif total_text_length < 5000:
            quality_score -= 0.1
        
        # Reduce score for OCR issues
        quality_score -= min(0.2, len(ocr_issues) * 0.05)
        
        # Reduce score for missing metadata
        quality_score -= min(0.2, len(missing_metadata) * 0.04)
        
        # Ensure score is between 0.0 and 1.0
        quality_score = max(0.0, min(1.0, quality_score))
        
        # Create quality analysis result
        analysis_result = {
            "success": True,
            "file_name": extraction_result.get('file_name', 'unknown'),
            "quality_score": quality_score,
            "text_length": total_text_length,
            "avg_page_length": avg_page_length,
            "empty_pages": empty_pages,
            "issues": issues,
            "recommendations": recommendations,
            "analysis_time": datetime.now().isoformat(),
        }
        
        # Store the analysis result in the state if tool_context is provided
        if tool_context:
            # Get state manager from context if available
            state_manager = tool_context.state.get('state_manager', self.state_manager)
            
            # Get the current state
            current_state = state_manager.get(tool_context)
            
            # Add analysis result to state
            if 'documents' not in current_state:
                current_state['documents'] = {}
            
            document_id = extraction_result.get('file_name', '').split('.')[0]
            if document_id in current_state['documents']:
                current_state['documents'][document_id]["quality_analysis"] = analysis_result
            else:
                current_state['documents'][document_id] = {
                    "quality_analysis": analysis_result,
                    "processed_at": datetime.now().isoformat(),
                }
            
            # Update the state
            state_manager.set(tool_context, current_state)
        
        return analysis_result
    
    def _check_ocr_issues(self, text: str) -> List[str]:
        """Check for common OCR issues in the extracted text.
        
        Args:
            text: The extracted text to check.
            
        Returns:
            A list of identified OCR issues.
        """
        issues = []
        
        # Check for character confusion (common OCR errors)
        char_confusion_patterns = [
            (r'\bl\b', "Possible 'I' to 'l' confusion"),  # Single 'l' as a word
            (r'[^a-zA-Z0-9\s,.;:?!\-\'\"]', "Non-standard characters detected"),  # Non-standard characters
            (r'\b\d[oO]\b', "Possible '0' to 'o' confusion"),  # Digit followed by 'o'
        ]
        
        import re
        for pattern, issue in char_confusion_patterns:
            if re.search(pattern, text):
                issues.append(issue)
        
        # Check for words stuck together (missing spaces)
        if re.search(r'[a-z][A-Z]', text):
            issues.append("Words may be stuck together (missing spaces)")
        
        # Check for excessive line breaks (common in PDFs with columns)
        lines = text.split('\n')
        short_lines = sum(1 for line in lines if 0 < len(line.strip()) < 40)
        if short_lines > len(lines) * 0.5:  # If more than 50% of lines are short
            issues.append("Excessive line breaks detected, possibly due to column layout")
        
        return issues
    
    def generate_improvement_suggestions(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Generate suggestions for improving document extraction based on quality analysis.
        
        Args:
            analysis_result: The quality analysis result.
            
        Returns:
            A list of improvement suggestions.
        """
        suggestions = []
        
        # Add recommendations from the analysis
        suggestions.extend(analysis_result.get('recommendations', []))
        
        # Add general suggestions based on quality score
        quality_score = analysis_result.get('quality_score', 0.0)
        
        if quality_score < 0.3:
            suggestions.append("Consider manual extraction or a different document source")
        elif quality_score < 0.6:
            suggestions.append("Try alternative extraction methods or pre-process the PDF")
        elif quality_score < 0.8:
            suggestions.append("Minor improvements needed, focus on specific issues")
        
        # Add specific suggestions for common issues
        issues = analysis_result.get('issues', [])
        
        for issue in issues:
            if "empty or very short pages" in issue:
                suggestions.append("Use OCR software for scanned pages")
            elif "metadata" in issue:
                suggestions.append("Extract metadata from document headers or use citation databases")
            elif "OCR issues" in issue or "character confusion" in issue:
                suggestions.append("Use a specialized academic OCR tool with support for scientific notation")
        
        return list(set(suggestions))  # Remove duplicates
