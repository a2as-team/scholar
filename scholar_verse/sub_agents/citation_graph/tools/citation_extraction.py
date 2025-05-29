"""Citation extraction tools for the ScholarVerse Citation Graph Agent.

This module provides tools for extracting citations from academic papers.
"""

from typing import Dict, Any, List, Optional, Tuple
import re
from datetime import datetime

from google.adk.tools import ToolContext

from scholar_verse.shared_libraries.logging_utils import logger
from scholar_verse.shared_libraries.state_management.adaptive_state import AdaptiveStateManager


class CitationExtractor:
    """Citation extractor for ScholarVerse.
    
    This class provides methods for extracting citations from academic papers
    and parsing them into structured data.
    """
    
    def __init__(self, state_manager: Optional[AdaptiveStateManager] = None):
        """Initialize the citation extractor.
        
        Args:
            state_manager: The state manager to use for storing extraction results.
        """
        self.state_manager = state_manager or AdaptiveStateManager()
    
    def extract_citations(self, document_text: str, document_id: str, 
                         tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """Extract citations from document text.
        
        Args:
            document_text: The full text of the document.
            document_id: The ID of the document.
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A dictionary containing the extracted citations.
        """
        logger.info(f"Extracting citations from document: {document_id}")
        
        # Get state manager from context if available
        if tool_context:
            state_manager = tool_context.state.get('state_manager', self.state_manager)
        else:
            state_manager = self.state_manager
        
        try:
            # Find the references section
            references_section = self._extract_references_section(document_text)
            
            if not references_section:
                logger.warning(f"No references section found in document: {document_id}")
                return {
                    "success": False,
                    "document_id": document_id,
                    "error": "No references section found",
                    "citations": [],
                }
            
            # Extract individual citations
            citations = self._parse_references(references_section)
            
            # Extract in-text citations
            in_text_citations = self._extract_in_text_citations(document_text)
            
            # Store the extraction result in the state if tool_context is provided
            if tool_context:
                # Get the current state
                current_state = state_manager.get(tool_context)
                
                # Add extraction result to state
                if 'citations' not in current_state:
                    current_state['citations'] = {}
                
                current_state['citations'][document_id] = {
                    "references": citations,
                    "in_text_citations": in_text_citations,
                    "extracted_at": datetime.now().isoformat(),
                }
                
                # Update the state
                state_manager.set(tool_context, current_state)
            
            return {
                "success": True,
                "document_id": document_id,
                "citations": citations,
                "in_text_citations": in_text_citations,
                "count": len(citations),
            }
        except Exception as e:
            error_msg = f"Error extracting citations: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "document_id": document_id,
                "error": error_msg,
                "citations": [],
            }
    
    def _extract_references_section(self, text: str) -> str:
        """Extract the references section from document text.
        
        Args:
            text: The document text.
            
        Returns:
            The extracted references section.
        """
        # Common section headers for references
        reference_headers = [
            r'References\s*\n',
            r'REFERENCES\s*\n',
            r'Bibliography\s*\n',
            r'BIBLIOGRAPHY\s*\n',
            r'Works Cited\s*\n',
            r'WORKS CITED\s*\n',
            r'Literature Cited\s*\n',
            r'LITERATURE CITED\s*\n',
        ]
        
        # Try to find the references section
        for header in reference_headers:
            match = re.search(header, text, re.IGNORECASE)
            if match:
                start_pos = match.end()
                
                # Find the next section header or end of document
                next_section_match = re.search(r'\n\s*[A-Z][A-Za-z\s]*\s*\n', text[start_pos:])
                if next_section_match:
                    end_pos = start_pos + next_section_match.start()
                    return text[start_pos:end_pos].strip()
                else:
                    return text[start_pos:].strip()
        
        # If no standard header is found, try to find a numbered references section
        match = re.search(r'\n\s*\d+\.\s*References\s*\n', text, re.IGNORECASE)
        if match:
            start_pos = match.end()
            
            # Find the next section header or end of document
            next_section_match = re.search(r'\n\s*\d+\.\s*[A-Z][A-Za-z\s]*\s*\n', text[start_pos:])
            if next_section_match:
                end_pos = start_pos + next_section_match.start()
                return text[start_pos:end_pos].strip()
            else:
                return text[start_pos:].strip()
        
        # If still not found, look for sections with many citation-like patterns
        lines = text.split('\n')
        citation_line_count = 0
        citation_section_start = -1
        citation_section_end = -1
        citation_density = 0
        
        for i, line in enumerate(lines):
            # Check if line looks like a citation
            if (re.search(r'\(\d{4}\)', line) or  # Year in parentheses
                re.search(r'\d{4}\.', line) or     # Year followed by period
                re.search(r'et al\.', line) or      # et al.
                re.search(r'\d+:\d+', line)):      # Page numbers
                
                if citation_section_start == -1:
                    citation_section_start = i
                
                citation_line_count += 1
                citation_section_end = i
        
        # If we found a dense section of citation-like lines
        if citation_section_start != -1 and citation_section_end != -1:
            section_size = citation_section_end - citation_section_start + 1
            if section_size > 0:
                citation_density = citation_line_count / section_size
            
            if citation_density > 0.5 and section_size > 10:  # More than 50% citation-like lines
                return '\n'.join(lines[citation_section_start:citation_section_end+1])
        
        return ""
    
    def _parse_references(self, references_text: str) -> List[Dict[str, Any]]:
        """Parse individual references from the references section.
        
        Args:
            references_text: The references section text.
            
        Returns:
            A list of parsed citations.
        """
        citations = []
        
        # Split into individual references
        # Try different splitting strategies
        
        # Strategy 1: Split by numbered references
        if re.search(r'^\s*\[?\d+\]?\s', references_text, re.MULTILINE):
            references = re.split(r'\n\s*\[?\d+\]?\s', '\n' + references_text)
            references = [ref for ref in references if ref.strip()]
        # Strategy 2: Split by lines starting with author names (typically last name, first initial)
        elif re.search(r'^\s*[A-Z][a-z]+,\s+[A-Z]\.', references_text, re.MULTILINE):
            references = re.split(r'\n\s*[A-Z][a-z]+,\s+[A-Z]\.', '\n' + references_text)
            references = [ref for ref in references if ref.strip()]
        # Strategy 3: Split by blank lines
        else:
            references = re.split(r'\n\s*\n', references_text)
            references = [ref for ref in references if ref.strip()]
        
        # Process each reference
        for i, reference in enumerate(references):
            reference = reference.strip()
            if not reference:
                continue
            
            # Parse the citation
            citation = self._parse_citation(reference)
            citation['index'] = i + 1
            
            citations.append(citation)
        
        return citations
    
    def _parse_citation(self, citation_text: str) -> Dict[str, Any]:
        """Parse a single citation into structured data.
        
        Args:
            citation_text: The citation text.
            
        Returns:
            A dictionary containing the parsed citation data.
        """
        citation = {
            "raw_text": citation_text.strip(),
            "authors": [],
            "year": "",
            "title": "",
            "journal": "",
            "volume": "",
            "issue": "",
            "pages": "",
            "doi": "",
        }
        
        # Extract DOI if present
        doi_match = re.search(r'(?:doi|DOI)[:\s]*(10\.\d+/[^\s]+)', citation_text)
        if doi_match:
            citation["doi"] = doi_match.group(1).strip()
        
        # Extract year
        year_match = re.search(r'\(?(19|20)\d{2}\)?', citation_text)
        if year_match:
            citation["year"] = year_match.group(0).strip('()')
        
        # Extract authors
        # This is complex due to various citation styles
        # Look for patterns like "Author1, A., Author2, B."
        author_section = citation_text
        if year_match:
            author_section = citation_text[:year_match.start()]
        
        # Try to extract authors based on common patterns
        if ',' in author_section:
            # Split by 'and' or '&' first to handle last author differently
            author_parts = re.split(r'\s+and\s+|\s+&\s+', author_section, 1)
            
            if len(author_parts) > 1:
                # Process all authors except the last one
                first_authors = author_parts[0].split(',')
                first_authors = [a.strip() for a in first_authors if a.strip()]
                
                # Process the last author
                last_author = author_parts[1].split(',')[0].strip()
                
                authors = first_authors + [last_author]
            else:
                # Just split by commas
                authors = author_section.split(',')
                authors = [a.strip() for a in authors if a.strip()]
            
            # Clean up author names
            cleaned_authors = []
            for author in authors:
                # Remove et al.
                if 'et al.' in author:
                    author = author.replace('et al.', '').strip()
                    if author:
                        cleaned_authors.append(author)
                    cleaned_authors.append("et al.")
                    break
                
                # Remove initials if they appear as separate items
                if len(author) <= 2 and author.endswith('.'):
                    continue
                
                if author:
                    cleaned_authors.append(author)
            
            citation["authors"] = cleaned_authors[:10]  # Limit to 10 authors
        
        # Extract title
        # Look for text between year and journal, or after authors if no year
        title_start = 0
        if year_match:
            title_start = year_match.end()
        elif citation["authors"]:
            # Estimate where authors end
            author_text = ', '.join(citation["authors"])
            if author_text in citation_text:
                title_start = citation_text.index(author_text) + len(author_text)
        
        # Look for journal indicators
        journal_indicators = ['. In ', '. ', ', vol', ', Vol', ', pp', ', Pages', ', p.']
        title_end = len(citation_text)
        
        for indicator in journal_indicators:
            pos = citation_text.find(indicator, title_start)
            if pos > -1 and pos < title_end:
                title_end = pos
        
        if title_start < title_end:
            title = citation_text[title_start:title_end].strip()
            # Clean up the title
            title = title.strip('" .,').strip()
            citation["title"] = title
        
        # Extract journal, volume, issue, pages
        journal_section = citation_text[title_end:].strip()
        
        # Journal name is typically before volume/issue
        journal_match = re.search(r'([^,]+)(?:,\s+Vol(?:ume)?[\s.]*\d+|,\s+\d+\s*\(\d+\))', journal_section)
        if journal_match:
            citation["journal"] = journal_match.group(1).strip()
        
        # Volume and issue
        volume_match = re.search(r'Vol(?:ume)?[\s.]*([\d\w]+)', journal_section)
        if volume_match:
            citation["volume"] = volume_match.group(1).strip()
        
        issue_match = re.search(r'\((\d+)\)', journal_section)
        if issue_match:
            citation["issue"] = issue_match.group(1).strip()
        
        # Pages
        pages_match = re.search(r'(?:pages|pp|p)\.?\s*(\d+(?:\s*-\s*\d+)?)', journal_section, re.IGNORECASE)
        if pages_match:
            citation["pages"] = pages_match.group(1).strip()
        
        return citation
    
    def _extract_in_text_citations(self, text: str) -> List[Dict[str, Any]]:
        """Extract in-text citations from document text.
        
        Args:
            text: The document text.
            
        Returns:
            A list of in-text citations.
        """
        in_text_citations = []
        
        # Look for citations in parentheses (Author, Year)
        parenthetical_matches = re.finditer(r'\(([^\)]+,\s*\d{4}[^\)]*)\)', text)
        for match in parenthetical_matches:
            citation_text = match.group(1).strip()
            in_text_citations.append({
                "type": "parenthetical",
                "text": citation_text,
                "position": match.start(),
            })
        
        # Look for citations in brackets [1], [2-4], etc.
        bracket_matches = re.finditer(r'\[(\d+(?:[-–,]\d+)*)\]', text)
        for match in bracket_matches:
            citation_text = match.group(1).strip()
            in_text_citations.append({
                "type": "numbered",
                "text": citation_text,
                "position": match.start(),
            })
        
        # Look for author-year citations like "Smith et al. (2020)"
        author_year_matches = re.finditer(r'([A-Z][a-z]+(?:\s+et\s+al\.)?\s+\(\d{4}\))', text)
        for match in author_year_matches:
            citation_text = match.group(1).strip()
            in_text_citations.append({
                "type": "author-year",
                "text": citation_text,
                "position": match.start(),
            })
        
        return in_text_citations
    
    def match_citations_to_papers(self, citations: List[Dict[str, Any]], 
                                 tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """Match extracted citations to papers in the knowledge graph.
        
        Args:
            citations: The list of extracted citations.
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A dictionary with the results of the matching operation.
        """
        logger.info(f"Matching {len(citations)} citations to papers in the knowledge graph")
        
        # Get state manager from context if available
        if tool_context:
            state_manager = tool_context.state.get('state_manager', self.state_manager)
        else:
            state_manager = self.state_manager
        
        # Get the current state
        current_state = state_manager.get(tool_context) if tool_context else {}
        
        # Get papers from the knowledge graph in the state
        papers = current_state.get('knowledge_graph', {}).get('papers', {})
        
        matches = []
        unmatched = []
        
        for citation in citations:
            matched = False
            
            # Try to match by DOI first
            if citation.get('doi'):
                for paper_id, paper in papers.items():
                    if paper.get('doi') == citation.get('doi'):
                        matches.append({
                            "citation": citation,
                            "paper": paper,
                            "match_type": "doi",
                        })
                        matched = True
                        break
            
            # If not matched by DOI, try to match by title
            if not matched and citation.get('title'):
                for paper_id, paper in papers.items():
                    # Simple fuzzy matching by removing punctuation and comparing lowercase
                    citation_title = re.sub(r'[^\w\s]', '', citation.get('title', '').lower())
                    paper_title = re.sub(r'[^\w\s]', '', paper.get('title', '').lower())
                    
                    # Check if one is a substring of the other or if they're very similar
                    if (citation_title in paper_title or 
                        paper_title in citation_title or 
                        self._similarity(citation_title, paper_title) > 0.8):
                        
                        matches.append({
                            "citation": citation,
                            "paper": paper,
                            "match_type": "title",
                        })
                        matched = True
                        break
            
            # If not matched by title, try to match by authors and year
            if not matched and citation.get('authors') and citation.get('year'):
                for paper_id, paper in papers.items():
                    if paper.get('year') == citation.get('year'):
                        # Check if at least one author matches
                        citation_authors = [re.sub(r'[^\w\s]', '', author.lower()) 
                                           for author in citation.get('authors', [])]
                        paper_authors = [re.sub(r'[^\w\s]', '', author.lower()) 
                                        for author in paper.get('authors', [])]
                        
                        for c_author in citation_authors:
                            for p_author in paper_authors:
                                if c_author in p_author or p_author in c_author:
                                    matches.append({
                                        "citation": citation,
                                        "paper": paper,
                                        "match_type": "author-year",
                                    })
                                    matched = True
                                    break
                            if matched:
                                break
                    if matched:
                        break
            
            if not matched:
                unmatched.append(citation)
        
        result = {
            "success": True,
            "total_citations": len(citations),
            "matched": len(matches),
            "unmatched": len(unmatched),
            "matches": matches,
            "unmatched_citations": unmatched,
        }
        
        # Store the result in the state if tool_context is provided
        if tool_context:
            # Initialize citation_matching in state if not present
            if 'citation_matching' not in current_state:
                current_state['citation_matching'] = {}
            
            # Add matching result to state
            current_state['citation_matching']['latest_result'] = {
                "timestamp": datetime.now().isoformat(),
                "matched": len(matches),
                "unmatched": len(unmatched),
            }
            
            # Update the state
            state_manager.set(tool_context, current_state)
        
        return result
    
    def _similarity(self, s1: str, s2: str) -> float:
        """Calculate a simple similarity score between two strings.
        
        Args:
            s1: First string.
            s2: Second string.
            
        Returns:
            A similarity score between 0 and 1.
        """
        # Simple Jaccard similarity on word sets
        words1 = set(s1.split())
        words2 = set(s2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
