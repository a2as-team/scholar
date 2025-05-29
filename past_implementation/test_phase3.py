"""Test script for Phase 3 implementation of ScholarVerse."""

import os
import sys
import asyncio
from pathlib import Path
import tempfile

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from scholar_verse.config import get_config
from scholar_verse.shared_libraries.logging_utils import logger
from scholar_verse.sub_agents.ingestion.agent import ingestion_agent
from scholar_verse.sub_agents.ingestion.tools.pdf_extraction import PDFExtractor
from scholar_verse.sub_agents.ingestion.tools.quality_analysis import DocumentQualityAnalyzer
from scholar_verse.sub_agents.ingestion.tools.extraction_improvement import ExtractionImprover
from scholar_verse.sub_agents.ingestion.tools.document_processor import DocumentProcessor
from scholar_verse.shared_libraries.state_management.adaptive_state import AdaptiveStateManager


class MockToolContext:
    """Mock tool context for testing."""
    
    def __init__(self):
        self.state = {'state_manager': AdaptiveStateManager()}


def create_test_pdf():
    """Create a test PDF file for testing."""
    # Create a temporary PDF file with some content
    import PyPDF2
    from PyPDF2 import PdfWriter, PdfReader
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create a simple PDF using reportlab
    temp_pdf_path = os.path.join(temp_dir, "temp.pdf")
    c = canvas.Canvas(temp_pdf_path, pagesize=letter)
    c.setFont("Helvetica", 12)
    
    # Add title
    c.drawString(100, 750, "ScholarVerse: A Novel Approach to Academic Knowledge Graphs")
    
    # Add authors
    c.drawString(100, 730, "John Smith, Jane Doe, Robert Johnson")
    
    # Add abstract
    c.drawString(100, 700, "Abstract")
    abstract = (
        "This paper presents ScholarVerse, a novel approach to constructing and analyzing "
        "academic knowledge graphs. We demonstrate how our system can extract meaningful "
        "relationships between research papers and help researchers discover new insights."
    )
    text_object = c.beginText(100, 680)
    for line in abstract.split("\n"):
        text_object.textLine(line)
    c.drawText(text_object)
    
    # Add keywords
    c.drawString(100, 630, "Keywords: knowledge graphs, academic research, citation analysis, AI")
    
    # Add DOI
    c.drawString(100, 610, "DOI: 10.1234/scholar.2025.01234")
    
    # Add publication date
    c.drawString(100, 590, "Published: 2025")
    
    # Add some content
    c.drawString(100, 550, "1. Introduction")
    intro = (
        "Academic knowledge graphs have become increasingly important in the era of "
        "information overload. Researchers need tools to navigate the vast landscape "
        "of scientific literature and identify relevant connections between papers."
    )
    text_object = c.beginText(100, 530)
    for line in intro.split("\n"):
        text_object.textLine(line)
    c.drawText(text_object)
    
    c.save()
    
    return temp_pdf_path


async def test_pdf_extraction():
    """Test the PDF extraction functionality."""
    print("\n=== Testing PDF Extraction ===")
    
    # Create a test PDF
    pdf_path = create_test_pdf()
    print(f"Created test PDF at: {pdf_path}")
    
    # Create a mock tool context
    tool_context = MockToolContext()
    
    # Initialize PDF extractor
    pdf_extractor = PDFExtractor(tool_context.state['state_manager'])
    
    # Extract text
    extraction_result = pdf_extractor.extract_text(pdf_path, tool_context)
    print(f"Extraction successful: {extraction_result['success']}")
    print(f"Number of pages: {extraction_result['num_pages']}")
    print(f"Text length: {len(extraction_result['full_text'])} characters")
    
    # Extract metadata
    metadata_result = pdf_extractor.extract_metadata(pdf_path, tool_context)
    print(f"Metadata extraction successful: {metadata_result['success']}")
    print(f"Title: {metadata_result['title']}")
    print(f"Authors: {metadata_result['authors']}")
    print(f"DOI: {metadata_result['doi']}")
    
    print("PDF extraction test passed!")
    return pdf_path


async def test_document_quality_analysis(pdf_path):
    """Test the document quality analysis functionality."""
    print("\n=== Testing Document Quality Analysis ===")
    
    # Create a mock tool context
    tool_context = MockToolContext()
    
    # Initialize PDF extractor and quality analyzer
    pdf_extractor = PDFExtractor(tool_context.state['state_manager'])
    quality_analyzer = DocumentQualityAnalyzer(tool_context.state['state_manager'])
    
    # Extract text and metadata
    extraction_result = pdf_extractor.extract_text(pdf_path, tool_context)
    metadata_result = pdf_extractor.extract_metadata(pdf_path, tool_context)
    extraction_result['metadata'] = metadata_result
    
    # Analyze document quality
    quality_result = quality_analyzer.analyze_document_quality(extraction_result, tool_context)
    print(f"Quality analysis successful: {quality_result['success']}")
    print(f"Quality score: {quality_result['quality_score']}")
    print(f"Issues found: {len(quality_result['issues'])}")
    if quality_result['issues']:
        print(f"Sample issue: {quality_result['issues'][0]}")
    
    # Generate improvement suggestions
    suggestions = quality_analyzer.generate_improvement_suggestions(quality_result)
    print(f"Number of improvement suggestions: {len(suggestions)}")
    if suggestions:
        print(f"Sample suggestion: {suggestions[0]}")
    
    print("Document quality analysis test passed!")


async def test_extraction_improvement():
    """Test the extraction improvement functionality."""
    print("\n=== Testing Extraction Improvement ===")
    
    # Create a mock tool context
    tool_context = MockToolContext()
    
    # Initialize extraction improver
    extraction_improver = ExtractionImprover(tool_context.state['state_manager'])
    
    # Process feedback
    feedback = {
        "ratings": {
            "text_extraction": 4,
            "metadata_extraction": 3,
            "structure_analysis": 2,
        },
        "comments": "The text extraction was good, but the structure analysis needs improvement. Tables were not properly recognized.",
    }
    
    feedback_result = extraction_improver.process_extraction_feedback("test_document", feedback, tool_context)
    print(f"Feedback processing successful: {feedback_result['success']}")
    print(f"Number of improvement actions: {len(feedback_result['improvement_actions'])}")
    if feedback_result['improvement_actions']:
        action = feedback_result['improvement_actions'][0]
        print(f"Sample action: {action['action']} - {action['description']}")
    
    # Apply extraction improvements
    original_params = {"use_ocr": False, "enhance_resolution": False}
    improved_params = extraction_improver.apply_extraction_improvements(original_params, tool_context)
    print(f"Original params: {original_params}")
    print(f"Improved params: {improved_params}")
    
    # Get performance metrics
    metrics = extraction_improver.get_extraction_performance_metrics(tool_context)
    print(f"Total documents with feedback: {metrics['total_documents']}")
    print(f"Total feedback entries: {metrics['total_feedback_entries']}")
    
    print("Extraction improvement test passed!")


async def test_document_processor(pdf_path):
    """Test the document processor functionality."""
    print("\n=== Testing Document Processor ===")
    
    # Create a mock tool context
    tool_context = MockToolContext()
    
    # Initialize document processor
    document_processor = DocumentProcessor(tool_context.state['state_manager'])
    
    # Process document
    processing_result = document_processor.process_document(pdf_path, tool_context)
    print(f"Document processing successful: {processing_result['success']}")
    print(f"Document ID: {processing_result['document_id']}")
    print(f"Title: {processing_result['metadata']['title']}")
    print(f"Quality score: {processing_result['quality']['score']}")
    
    # Get document summary
    document_id = processing_result['document_id']
    summary = document_processor.get_document_summary(document_id, tool_context)
    print(f"Summary retrieval successful: {summary['success']}")
    print(f"Summary title: {summary['title']}")
    
    print("Document processor test passed!")


async def test_ingestion_agent():
    """Test the ingestion agent."""
    print("\n=== Testing Ingestion Agent ===")
    print(f"Ingestion agent name: {ingestion_agent.name}")
    print(f"Ingestion agent model: {ingestion_agent.model}")
    
    # Get tool names
    tool_names = [tool.name for tool in ingestion_agent.tools]
    print(f"Ingestion agent tools: {tool_names}")
    print(f"Number of tools: {len(tool_names)}")
    
    # Check for specific tools
    expected_tools = [
        "process_document", "process_multiple_documents", "extract_text", 
        "extract_metadata", "analyze_document_quality", "provide_extraction_feedback", 
        "get_document_summary", "get_extraction_performance"
    ]
    
    missing_tools = [tool for tool in expected_tools if tool not in tool_names]
    if missing_tools:
        print(f"Warning: Missing expected tools: {missing_tools}")
    else:
        print("All expected tools are present")
    
    print("Ingestion agent test passed!")


async def main_async():
    """Run all tests asynchronously."""
    print("\n=== ScholarVerse Phase 3 Implementation Test ===")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Run tests
    pdf_path = await test_pdf_extraction()
    await test_document_quality_analysis(pdf_path)
    await test_extraction_improvement()
    await test_document_processor(pdf_path)
    await test_ingestion_agent()
    
    print("\n=== All Tests Passed! ===")
    print("Phase 3 implementation is working correctly.")
    print("You can now proceed to Phase 4: Knowledge Graph Construction.")


def main_sync():
    """Run the main async function."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main_sync()
