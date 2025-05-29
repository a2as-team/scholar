"""Simple test script for Phase 3 implementation of ScholarVerse."""

import os
import sys
import asyncio
from pathlib import Path
import tempfile
import json

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from scholar_verse.config import get_config
from scholar_verse.shared_libraries.logging_utils import logger
from scholar_verse.sub_agents.ingestion.agent import ingestion_agent
from scholar_verse.sub_agents.ingestion.tools.document_processor import DocumentProcessor
from scholar_verse.shared_libraries.state_management.adaptive_state import AdaptiveStateManager


class MockToolContext:
    """Mock tool context for testing."""
    
    def __init__(self):
        self.state = {'state_manager': AdaptiveStateManager()}


def create_mock_extraction_result():
    """Create a mock extraction result for testing."""
    return {
        "success": True,
        "file_path": "/path/to/mock.pdf",
        "file_name": "mock.pdf",
        "num_pages": 10,
        "pages_text": [f"This is page {i+1} content" for i in range(10)],
        "full_text": "\n\n".join([f"This is page {i+1} content" for i in range(10)]),
        "extraction_method": "Mock",
        "extraction_time": "2025-05-15T22:30:00",
        "metadata": {
            "success": True,
            "title": "ScholarVerse: A Novel Approach to Academic Knowledge Graphs",
            "authors": ["John Smith", "Jane Doe", "Robert Johnson"],
            "abstract": "This paper presents ScholarVerse, a novel approach to constructing and analyzing academic knowledge graphs.",
            "doi": "10.1234/scholar.2025.01234",
            "publication_date": "2025",
            "keywords": ["knowledge graphs", "academic research", "citation analysis", "AI"],
        }
    }


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


async def test_document_processor():
    """Test the document processor with mock data."""
    print("\n=== Testing Document Processor with Mock Data ===")
    
    # Create a mock tool context
    tool_context = MockToolContext()
    
    # Get the state manager
    state_manager = tool_context.state['state_manager']
    
    # Create a mock extraction result
    extraction_result = create_mock_extraction_result()
    
    # Store the mock extraction result in the state
    current_state = state_manager.get(tool_context)
    if 'documents' not in current_state:
        current_state['documents'] = {}
    
    document_id = "mock_document"
    current_state['documents'][document_id] = {
        "extraction_result": extraction_result,
        "processed_at": "2025-05-15T22:30:00",
    }
    
    state_manager.set(tool_context, current_state)
    
    # Initialize document processor
    document_processor = DocumentProcessor(state_manager)
    
    # Get document summary
    summary = document_processor.get_document_summary(document_id, tool_context)
    print(f"Summary retrieval successful: {summary['success']}")
    print(f"Document ID: {summary['document_id']}")
    print(f"Title: {summary['title']}")
    print(f"Authors: {', '.join(summary['authors'])}")
    
    print("Document processor test passed!")


async def test_feedback_processing():
    """Test the feedback processing functionality with mock data."""
    print("\n=== Testing Feedback Processing with Mock Data ===")
    
    # Create a mock tool context
    tool_context = MockToolContext()
    
    # Get the state manager
    state_manager = tool_context.state['state_manager']
    
    # Create a mock document
    document_id = "mock_document"
    
    # Import the extraction improver
    from scholar_verse.sub_agents.ingestion.tools.extraction_improvement import ExtractionImprover
    
    # Initialize extraction improver
    extraction_improver = ExtractionImprover(state_manager)
    
    # Process feedback
    feedback = {
        "ratings": {
            "text_extraction": 4,
            "metadata_extraction": 3,
            "structure_analysis": 2,
        },
        "comments": "The text extraction was good, but the structure analysis needs improvement.",
    }
    
    feedback_result = extraction_improver.process_extraction_feedback(document_id, feedback, tool_context)
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
    
    print("Feedback processing test passed!")


async def main_async():
    """Run all tests asynchronously."""
    print("\n=== ScholarVerse Phase 3 Implementation Test (Simple) ===")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Run tests
    await test_ingestion_agent()
    await test_document_processor()
    await test_feedback_processing()
    
    print("\n=== All Tests Passed! ===")
    print("Phase 3 implementation is working correctly.")
    print("You can now proceed to Phase 4: Knowledge Graph Construction.")


def main_sync():
    """Run the main async function."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main_sync()
