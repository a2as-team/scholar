"""Relationship identification tools for the ScholarVerse Citation Graph Agent.

This module provides tools for identifying relationships between papers, authors, and concepts.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import re
from datetime import datetime
from collections import Counter

from google.adk.tools import ToolContext

from scholar_verse.shared_libraries.logging_utils import logger
from scholar_verse.shared_libraries.state_management.adaptive_state import AdaptiveStateManager


class RelationshipIdentifier:
    """Relationship identifier for ScholarVerse.
    
    This class provides methods for identifying relationships between papers, authors,
    and concepts in the knowledge graph.
    """
    
    def __init__(self, state_manager: Optional[AdaptiveStateManager] = None):
        """Initialize the relationship identifier.
        
        Args:
            state_manager: The state manager to use for storing relationship data.
        """
        self.state_manager = state_manager or AdaptiveStateManager()
    
    def identify_paper_relationships(self, paper_data: Dict[str, Any], 
                                    tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """Identify relationships between a paper and other entities.
        
        Args:
            paper_data: The paper data to analyze.
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A dictionary containing the identified relationships.
        """
        logger.info(f"Identifying relationships for paper: {paper_data.get('title', 'Unknown Title')}")
        
        # Get state manager from context if available
        if tool_context:
            state_manager = tool_context.state.get('state_manager', self.state_manager)
        else:
            state_manager = self.state_manager
        
        try:
            # Extract paper properties
            title = paper_data.get('title', 'Unknown Title')
            doi = paper_data.get('doi', '')
            abstract = paper_data.get('abstract', '')
            year = paper_data.get('year') or paper_data.get('publication_date', '')
            if isinstance(year, str) and len(year) >= 4:
                year = year[:4]  # Extract year from date string
            
            # Extract authors
            authors = paper_data.get('authors', [])
            if isinstance(authors, str):
                # Split author string if it's not already a list
                authors = [author.strip() for author in authors.split(',') if author.strip()]
            
            # Extract keywords/concepts
            keywords = paper_data.get('keywords', [])
            if isinstance(keywords, str):
                # Split keywords string if it's not already a list
                keywords = [keyword.strip() for keyword in keywords.split(',') if keyword.strip()]
            
            # Extract citations
            citations = paper_data.get('citations', [])
            
            # Identify author relationships
            author_relationships = self._identify_author_relationships(authors)
            
            # Identify concept relationships
            concept_relationships = self._identify_concept_relationships(keywords, abstract)
            
            # Identify citation relationships
            citation_relationships = self._identify_citation_relationships(citations, tool_context)
            
            # Combine all relationships
            relationships = {
                "author_relationships": author_relationships,
                "concept_relationships": concept_relationships,
                "citation_relationships": citation_relationships,
            }
            
            # Store the relationships in the state if tool_context is provided
            if tool_context:
                # Get the current state
                current_state = state_manager.get(tool_context)
                
                # Initialize relationships in state if not present
                if 'relationships' not in current_state:
                    current_state['relationships'] = {}
                
                # Add relationships to state
                current_state['relationships'][doi or title] = {
                    "paper": {
                        "title": title,
                        "doi": doi,
                    },
                    "relationships": relationships,
                    "identified_at": datetime.now().isoformat(),
                }
                
                # Update the state
                state_manager.set(tool_context, current_state)
            
            return {
                "success": True,
                "paper": {
                    "title": title,
                    "doi": doi,
                },
                "relationships": relationships,
            }
        except Exception as e:
            error_msg = f"Error identifying relationships: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
            }
    
    def _identify_author_relationships(self, authors: List[str]) -> Dict[str, Any]:
        """Identify relationships between authors.
        
        Args:
            authors: The list of authors to analyze.
            
        Returns:
            A dictionary containing the identified author relationships.
        """
        relationships = []
        
        # Identify co-authorship relationships
        for i, author1 in enumerate(authors):
            for j, author2 in enumerate(authors):
                if i < j:  # Avoid duplicates and self-relationships
                    relationships.append({
                        "type": "co-author",
                        "author1": author1,
                        "author2": author2,
                        "strength": 1.0,  # All co-authors have equal relationship strength
                    })
        
        return {
            "co_authorships": relationships,
            "count": len(relationships),
        }
    
    def _identify_concept_relationships(self, keywords: List[str], abstract: str) -> Dict[str, Any]:
        """Identify relationships between concepts.
        
        Args:
            keywords: The list of keywords/concepts to analyze.
            abstract: The paper abstract for additional concept extraction.
            
        Returns:
            A dictionary containing the identified concept relationships.
        """
        relationships = []
        
        # Extract additional concepts from abstract if available
        additional_concepts = set()
        if abstract:
            # Simple extraction of potential concepts (noun phrases)
            # In a production system, this would use more sophisticated NLP
            words = re.findall(r'\b[A-Za-z][A-Za-z-]+\b', abstract)
            word_counts = Counter(word.lower() for word in words)
            
            # Consider words that appear multiple times as potential concepts
            for word, count in word_counts.items():
                if count >= 3 and len(word) > 4 and word not in ['these', 'those', 'their', 'other', 'which', 'where', 'when', 'what', 'this', 'that']:
                    additional_concepts.add(word)
        
        # Combine explicit keywords with extracted concepts
        all_concepts = set(keyword.lower() for keyword in keywords)
        all_concepts.update(additional_concepts)
        all_concepts = list(all_concepts)
        
        # Identify relationships between concepts
        for i, concept1 in enumerate(all_concepts):
            for j, concept2 in enumerate(all_concepts):
                if i < j:  # Avoid duplicates and self-relationships
                    # Calculate relationship strength based on co-occurrence
                    # In a real system, this would use more sophisticated methods
                    strength = 0.5  # Default strength for keyword co-occurrence
                    
                    if abstract:
                        # Increase strength if both concepts appear in abstract
                        if concept1 in abstract.lower() and concept2 in abstract.lower():
                            # Calculate distance between concepts in abstract
                            concept1_pos = abstract.lower().find(concept1)
                            concept2_pos = abstract.lower().find(concept2)
                            
                            if concept1_pos >= 0 and concept2_pos >= 0:
                                distance = abs(concept1_pos - concept2_pos)
                                max_distance = len(abstract)
                                proximity_score = 1.0 - (distance / max_distance)
                                strength = max(strength, 0.5 + (0.5 * proximity_score))
                    
                    relationships.append({
                        "type": "related-concept",
                        "concept1": concept1,
                        "concept2": concept2,
                        "strength": strength,
                    })
        
        return {
            "concept_relationships": relationships,
            "count": len(relationships),
            "extracted_concepts": list(additional_concepts),
        }
    
    def _identify_citation_relationships(self, citations: List[Dict[str, Any]], 
                                        tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """Identify relationships in citation data.
        
        Args:
            citations: The list of citations to analyze.
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A dictionary containing the identified citation relationships.
        """
        direct_citations = []
        bibliographic_coupling = []
        co_citation = []
        
        # Get papers from the knowledge graph in the state if tool_context is provided
        papers = {}
        if tool_context:
            # Get state manager from context
            state_manager = tool_context.state.get('state_manager', self.state_manager)
            
            # Get the current state
            current_state = state_manager.get(tool_context)
            
            # Get papers from the knowledge graph in the state
            papers = current_state.get('knowledge_graph', {}).get('papers', {})
        
        # Process direct citations
        cited_papers = set()
        for citation in citations:
            cited_title = citation.get('title', '')
            cited_doi = citation.get('doi', '')
            
            if cited_title or cited_doi:
                cited_papers.add(cited_doi or cited_title)
                
                direct_citations.append({
                    "type": "cites",
                    "cited_title": cited_title,
                    "cited_doi": cited_doi,
                })
        
        # Identify bibliographic coupling (papers that cite the same sources)
        if papers and cited_papers:
            # Find other papers that cite the same sources
            for paper_id, paper in papers.items():
                paper_citations = paper.get('citations', [])
                if paper_citations:
                    # Extract DOIs or titles of cited papers
                    paper_cited = set()
                    for cite in paper_citations:
                        paper_cited.add(cite.get('doi') or cite.get('title'))
                    
                    # Calculate overlap with the current paper's citations
                    common_citations = cited_papers.intersection(paper_cited)
                    if common_citations:
                        bibliographic_coupling.append({
                            "type": "bibliographic-coupling",
                            "paper": paper.get('title'),
                            "common_citations": len(common_citations),
                            "strength": len(common_citations) / max(len(cited_papers), len(paper_cited)),
                        })
        
        # Identify co-citation (papers that are cited together by other papers)
        # This would require more context about the citation network
        # In a real implementation, this would be calculated based on the full citation graph
        
        return {
            "direct_citations": direct_citations,
            "bibliographic_coupling": bibliographic_coupling,
            "co_citation": co_citation,
            "citation_count": len(direct_citations),
        }
    
    def analyze_citation_network(self, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """Analyze the citation network to identify important papers and authors.
        
        Args:
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A dictionary containing the citation network analysis results.
        """
        logger.info("Analyzing citation network")
        
        # Get state manager from context if available
        if tool_context:
            state_manager = tool_context.state.get('state_manager', self.state_manager)
        else:
            state_manager = self.state_manager
        
        try:
            # Get the current state
            current_state = state_manager.get(tool_context) if tool_context else {}
            
            # Get papers and relationships from the state
            papers = current_state.get('knowledge_graph', {}).get('papers', {})
            relationships = current_state.get('relationships', {})
            
            if not papers:
                return {
                    "success": False,
                    "error": "No papers found in the knowledge graph",
                }
            
            # Calculate citation counts for each paper
            citation_counts = {}
            for paper_id, paper in papers.items():
                citation_counts[paper_id] = 0
            
            for rel_id, rel_data in relationships.items():
                paper_rels = rel_data.get('relationships', {})
                citation_rels = paper_rels.get('citation_relationships', {})
                direct_citations = citation_rels.get('direct_citations', [])
                
                for citation in direct_citations:
                    cited_doi = citation.get('cited_doi', '')
                    cited_title = citation.get('cited_title', '')
                    cited_id = cited_doi or cited_title
                    
                    if cited_id in citation_counts:
                        citation_counts[cited_id] += 1
            
            # Identify influential papers (most cited)
            influential_papers = []
            for paper_id, count in sorted(citation_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                if paper_id in papers:
                    influential_papers.append({
                        "title": papers[paper_id].get('title', 'Unknown'),
                        "doi": papers[paper_id].get('doi', ''),
                        "citation_count": count,
                    })
            
            # Calculate author influence
            author_citation_counts = {}
            for paper_id, paper in papers.items():
                authors = paper.get('authors', [])
                citations = citation_counts.get(paper_id, 0)
                
                for author in authors:
                    if author not in author_citation_counts:
                        author_citation_counts[author] = 0
                    author_citation_counts[author] += citations
            
            # Identify influential authors
            influential_authors = []
            for author, count in sorted(author_citation_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                influential_authors.append({
                    "name": author,
                    "citation_count": count,
                })
            
            # Identify research clusters (groups of related papers)
            # In a real implementation, this would use community detection algorithms
            # Here we use a simplified approach based on shared concepts
            
            # Extract concepts for each paper
            paper_concepts = {}
            for paper_id, paper in papers.items():
                keywords = paper.get('keywords', [])
                if isinstance(keywords, str):
                    keywords = [kw.strip() for kw in keywords.split(',') if kw.strip()]
                paper_concepts[paper_id] = set(kw.lower() for kw in keywords)
            
            # Group papers by shared concepts
            concept_papers = {}
            for paper_id, concepts in paper_concepts.items():
                for concept in concepts:
                    if concept not in concept_papers:
                        concept_papers[concept] = []
                    concept_papers[concept].append(paper_id)
            
            # Identify main research clusters
            research_clusters = []
            for concept, cluster_papers in sorted(concept_papers.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
                if len(cluster_papers) >= 2:  # Only consider clusters with at least 2 papers
                    cluster_titles = [papers[pid].get('title', 'Unknown') for pid in cluster_papers[:5]]
                    research_clusters.append({
                        "concept": concept,
                        "paper_count": len(cluster_papers),
                        "sample_papers": cluster_titles,
                    })
            
            analysis_result = {
                "success": True,
                "paper_count": len(papers),
                "influential_papers": influential_papers,
                "influential_authors": influential_authors,
                "research_clusters": research_clusters,
                "analysis_time": datetime.now().isoformat(),
            }
            
            # Store the analysis result in the state if tool_context is provided
            if tool_context:
                # Initialize citation_network in state if not present
                if 'citation_network' not in current_state:
                    current_state['citation_network'] = {}
                
                # Add analysis result to state
                current_state['citation_network']['analysis'] = analysis_result
                
                # Update the state
                state_manager.set(tool_context, current_state)
            
            return analysis_result
        except Exception as e:
            error_msg = f"Error analyzing citation network: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
            }
    
    def identify_research_trends(self, tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        """Identify research trends based on temporal analysis of the citation network.
        
        Args:
            tool_context: The tool context containing state and other information (optional).
            
        Returns:
            A dictionary containing the identified research trends.
        """
        logger.info("Identifying research trends")
        
        # Get state manager from context if available
        if tool_context:
            state_manager = tool_context.state.get('state_manager', self.state_manager)
        else:
            state_manager = self.state_manager
        
        try:
            # Get the current state
            current_state = state_manager.get(tool_context) if tool_context else {}
            
            # Get papers from the state
            papers = current_state.get('knowledge_graph', {}).get('papers', {})
            
            if not papers:
                return {
                    "success": False,
                    "error": "No papers found in the knowledge graph",
                }
            
            # Group papers by year
            papers_by_year = {}
            for paper_id, paper in papers.items():
                year = paper.get('year')
                if year:
                    if year not in papers_by_year:
                        papers_by_year[year] = []
                    papers_by_year[year].append(paper)
            
            # Extract concepts for each year
            concepts_by_year = {}
            for year, year_papers in papers_by_year.items():
                concepts = Counter()
                for paper in year_papers:
                    keywords = paper.get('keywords', [])
                    if isinstance(keywords, str):
                        keywords = [kw.strip() for kw in keywords.split(',') if kw.strip()]
                    for kw in keywords:
                        concepts[kw.lower()] += 1
                concepts_by_year[year] = concepts
            
            # Identify trending concepts (increasing over time)
            trending_concepts = []
            years = sorted(concepts_by_year.keys())
            if len(years) >= 2:
                # Compare concept frequencies across years
                for concept in set().union(*[set(concepts_by_year[year].keys()) for year in years]):
                    trend_data = []
                    for year in years:
                        trend_data.append({
                            "year": year,
                            "frequency": concepts_by_year[year].get(concept, 0),
                        })
                    
                    # Check if concept frequency is increasing
                    is_trending = True
                    for i in range(1, len(trend_data)):
                        if trend_data[i]['frequency'] <= trend_data[i-1]['frequency']:
                            is_trending = False
                            break
                    
                    if is_trending and trend_data[-1]['frequency'] > trend_data[0]['frequency']:
                        trending_concepts.append({
                            "concept": concept,
                            "trend_data": trend_data,
                            "growth": trend_data[-1]['frequency'] / max(1, trend_data[0]['frequency']),
                        })
            
            # Sort trending concepts by growth rate
            trending_concepts.sort(key=lambda x: x['growth'], reverse=True)
            
            # Identify emerging research areas (concepts that appear in recent years but not earlier)
            emerging_areas = []
            if len(years) >= 2:
                recent_years = years[-2:]  # Last two years
                earlier_years = years[:-2]  # Years before the last two
                
                if earlier_years and recent_years:
                    # Find concepts that appear in recent years but not earlier
                    recent_concepts = set().union(*[set(concepts_by_year[year].keys()) for year in recent_years])
                    earlier_concepts = set().union(*[set(concepts_by_year[year].keys()) for year in earlier_years])
                    
                    new_concepts = recent_concepts - earlier_concepts
                    for concept in new_concepts:
                        # Only include concepts that appear multiple times
                        frequency = sum(concepts_by_year[year].get(concept, 0) for year in recent_years)
                        if frequency >= 2:
                            emerging_areas.append({
                                "concept": concept,
                                "frequency": frequency,
                                "years": recent_years,
                            })
            
            # Sort emerging areas by frequency
            emerging_areas.sort(key=lambda x: x['frequency'], reverse=True)
            
            # Identify research gaps (areas with few papers but high citation impact)
            research_gaps = []
            # This would require more sophisticated analysis in a real implementation
            # Here we use a simple heuristic based on concept frequency and citation count
            
            trends_result = {
                "success": True,
                "years_analyzed": years,
                "paper_counts": {year: len(papers) for year, papers in papers_by_year.items()},
                "trending_concepts": trending_concepts[:10],  # Top 10 trending concepts
                "emerging_areas": emerging_areas[:5],  # Top 5 emerging areas
                "research_gaps": research_gaps,
                "analysis_time": datetime.now().isoformat(),
            }
            
            # Store the trends result in the state if tool_context is provided
            if tool_context:
                # Initialize research_trends in state if not present
                if 'research_trends' not in current_state:
                    current_state['research_trends'] = {}
                
                # Add trends result to state
                current_state['research_trends']['analysis'] = trends_result
                
                # Update the state
                state_manager.set(tool_context, current_state)
            
            return trends_result
        except Exception as e:
            error_msg = f"Error identifying research trends: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
            }
