# ScholarVerse: Updated Phase 2 - Intelligent Document Processing Pipeline

*Target Completion: June 30, 2025*  
*Status: Planning*

## Overview
This phase implements an intelligent document processing pipeline using Google ADK's advanced features for document ingestion, analysis, and knowledge extraction.

## Key Components

### 1. Document Processing Agent

#### Base Implementation
```python
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from typing import List, Dict, Optional
import fitz  # PyMuPDF
import io

class DocumentProcessingAgent(LlmAgent):
    def __init__(self):
        super().__init__(
            name="document_processor",
            description="Processes and analyzes academic documents",
            model="gemini-1.5-pro",
            tools=[
                self.extract_text,
                self.identify_sections,
                self.extract_citations,
                self.identify_entities
            ]
        )
    
    @FunctionTool
    async def extract_text(self, document_bytes: bytes) -> Dict:
        """Extract text from PDF document."""
        try:
            doc = fitz.open(stream=document_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return {"status": "success", "text": text}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @FunctionTool
    async def identify_sections(self, text: str) -> Dict:
        """Identify document sections (abstract, methodology, etc.)."""
        # Implementation using LLM
        pass
    
    @FunctionTool
    async def extract_citations(self, text: str) -> Dict:
        """Extract and normalize citations from text."""
        # Implementation using LLM and regex
        pass
    
    @FunctionTool
    async def identify_entities(self, text: str) -> Dict:
        """Identify key entities (authors, institutions, concepts)."""
        # Implementation using LLM
        pass
```

### 2. Document Processing Workflow

#### Document Processing Pipeline
```python
class DocumentPipeline:
    def __init__(self, session_service, memory_service):
        self.session_service = session_service
        self.memory_service = memory_service
        self.agent = DocumentProcessingAgent()
    
    async def process_document(self, user_id: str, document_bytes: bytes, metadata: Dict) -> Dict:
        """Process a document through the entire pipeline."""
        # Create a new session for this processing job
        session = await self.session_service.create_session(
            app_name="scholarverse",
            user_id=user_id,
            session_id=f"doc_{int(time.time())}",
            state={
                "document_metadata": metadata,
                "processing_steps": {},
                "status": "started"
            }
        )
        
        try:
            # Step 1: Extract text
            text_result = await self.agent.extract_text(document_bytes)
            if text_result["status"] != "success":
                raise Exception(f"Text extraction failed: {text_result.get('message')}")
            
            # Update session state
            await session.state.update({
                "processing_steps.text_extraction": "completed",
                "document_text": text_result["text"][:1000] + "..."  # Store preview
            })
            
            # Step 2: Process document sections
            sections = await self.agent.identify_sections(text_result["text"])
            
            # Step 3: Extract citations
            citations = await self.agent.extract_citations(text_result["text"])
            
            # Step 4: Identify entities
            entities = await self.agent.identify_entities(text_result["text"])
            
            # Store results in knowledge base
            doc_id = await self.memory_service.add_document(
                content=text_result["text"],
                metadata={
                    **metadata,
                    "sections": sections,
                    "citations": citations,
                    "entities": entities,
                    "processed_at": datetime.utcnow().isoformat()
                }
            )
            
            # Update session with completion status
            await session.state.update({
                "status": "completed",
                "document_id": doc_id,
                "entities_identified": len(entities.get("entities", [])),
                "citations_found": len(citations.get("citations", []))
            })
            
            return {"status": "success", "document_id": doc_id}
            
        except Exception as e:
            await session.state.update({
                "status": "error",
                "error": str(e)
            })
            raise
```

### 3. Document Quality Analysis

#### Quality Assessment
```python
class DocumentQualityAnalyzer:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def assess_quality(self, text: str) -> Dict:
        """Assess the quality of the document content."""
        prompt = """
        Analyze the following academic document text and assess its quality based on:
        1. Academic rigor
        2. Clarity of methodology
        3. Citation quality
        4. Originality
        
        Document:
        """ + text[:4000]  # First 4000 chars for analysis
        
        response = await self.llm.generate_content(
            prompt,
            temperature=0.2,
            max_output_tokens=500
        )
        
        return self._parse_quality_assessment(response.text)
    
    def _parse_quality_assessment(self, assessment_text: str) -> Dict:
        # Parse the LLM response into a structured format
        # Implementation depends on the LLM's response format
        pass
```

## Implementation Tasks

### 1. Document Processing
- [ ] Implement PDF text extraction with PyMuPDF
- [ ] Develop section identification using LLM
- [ ] Create citation extraction tools
- [ ] Implement entity recognition for academic content

### 2. Quality Analysis
- [ ] Design quality assessment prompts
- [ ] Implement scoring system for document quality
- [ ] Create feedback mechanism for quality improvement

### 3. Integration
- [ ] Connect to knowledge graph for entity resolution
- [ ] Implement document deduplication
- [ ] Create API endpoints for document submission

## Testing Strategy

### Unit Tests
- Document text extraction accuracy
- Section identification precision/recall
- Citation extraction validation
- Entity recognition accuracy

### Integration Tests
- End-to-end document processing
- Quality assessment consistency
- Knowledge graph integration

## Dependencies

```toml
[tool.poetry.dependencies]
python = "^3.9"
PyMuPDF = "^1.23.0"  # For PDF processing
python-multipart = "^0.0.6"  # For file uploads
pydantic = {extras = ["email"], version = "^2.0.0"}
```

## Next Steps
1. Implement citation graph construction
2. Develop cross-paper analysis capabilities
3. Build the deep search functionality
4. Create visualization components

---
*This document will be updated as the implementation progresses.*
