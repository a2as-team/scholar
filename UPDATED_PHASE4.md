# ScholarVerse: Updated Phase 4 - Deep Search & Knowledge Discovery

*Target Completion: July 30, 2025*  
*Status: Planning*

## Overview
This phase implements an advanced search and knowledge discovery system using Google ADK's capabilities, focusing on semantic search, knowledge retrieval, and cross-document analysis.

## Key Components

### 1. Deep Search Agent

#### Base Implementation
```python
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.memory.vertex_ai_rag import VertexAIRagMemoryService
from typing import List, Dict, Optional
import logging

class DeepSearchAgent(LlmAgent):
    def __init__(self, rag_service: VertexAIRagMemoryService):
        self.rag_service = rag_service
        
        super().__init__(
            name="deep_search_agent",
            description="Performs deep semantic search across knowledge sources",
            model="gemini-1.5-pro",
            tools=[
                self.semantic_search,
                self.retrieve_documents,
                self.analyze_relationships,
                self.generate_insights
            ]
        )
    
    @FunctionTool
    async def semantic_search(self, query: str, top_k: int = 5) -> Dict:
        """Perform semantic search across all knowledge sources."""
        try:
            results = await self.rag_service.search(
                query=query,
                top_k=top_k,
                filters={
                    "min_relevance_score": 0.7,
                    "source_types": ["academic_papers", "reports", "books"]
                }
            )
            return {"status": "success", "results": results}
        except Exception as e:
            logging.error(f"Semantic search failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    @FunctionTool
    async def retrieve_documents(self, doc_ids: List[str]) -> Dict:
        """Retrieve full document content by IDs."""
        try:
            documents = []
            for doc_id in doc_ids:
                doc = await self.rag_service.get_document(doc_id)
                if doc:
                    documents.append(doc)
            return {"status": "success", "documents": documents}
        except Exception as e:
            logging.error(f"Document retrieval failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    @FunctionTool
    async def analyze_relationships(self, entity_names: List[str], depth: int = 2) -> Dict:
        """Analyze relationships between entities in the knowledge graph."""
        query = """
        UNWIND $entityNames AS name
        MATCH path = (e1)-[r*1..%d]-(e2)
        WHERE e1.name IN $entityNames
          AND e2.name IN $entityNames
          AND e1 <> e2
        RETURN path
        LIMIT 50
        """ % depth
        
        try:
            result = await self.kg_agent.query_graph(
                query,
                {"entityNames": entity_names}
            )
            return {"status": "success", "relationships": result.get("results", [])}
        except Exception as e:
            logging.error(f"Relationship analysis failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    @FunctionTool
    async def generate_insights(self, query: str, context_docs: List[Dict]) -> Dict:
        """Generate insights based on search results and context."""
        try:
            prompt = self._build_insight_prompt(query, context_docs)
            response = await self.llm.generate_content(
                prompt,
                temperature=0.3,
                max_output_tokens=1000
            )
            return {"status": "success", "insights": response.text}
        except Exception as e:
            logging.error(f"Insight generation failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _build_insight_prompt(self, query: str, context_docs: List[Dict]) -> str:
        """Construct the prompt for insight generation."""
        context_str = "\n\n".join(
            f"Document {i+1} (Relevance: {doc.get('relevance_score', 0):.2f}):\n"
            f"Title: {doc.get('title', 'N/A')}\n"
            f"Content: {doc.get('content', '')[:1000]}..."
            for i, doc in enumerate(context_docs)
        )
        
        return f"""
        Analyze the following research query and relevant documents to generate insights.
        
        Query: {query}
        
        Relevant Documents:
        {context_str}
        
        Please provide:
        1. A summary of key findings
        2. Any contradictions or debates in the literature
        3. Gaps in current research
        4. Potential future research directions
        """
```

### 2. Search Orchestrator

#### Search Workflow
```python
class SearchOrchestrator:
    def __init__(self, search_agent, kg_agent):
        self.search_agent = search_agent
        self.kg_agent = kg_agent
    
    async def process_query(self, query: str, user_id: str, session_id: str) -> Dict:
        """Process a search query end-to-end."""
        # Log the query
        await self._log_query(query, user_id, session_id)
        
        # Step 1: Perform semantic search
        search_results = await self.search_agent.semantic_search(query)
        if search_results["status"] != "success" or not search_results.get("results"):
            return {"status": "error", "message": "No relevant documents found"}
        
        # Step 2: Retrieve top documents
        doc_ids = [r["document_id"] for r in search_results["results"][:3]]
        docs_result = await self.search_agent.retrieve_documents(doc_ids)
        if docs_result["status"] != "success":
            return docs_result
        
        # Step 3: Extract entities for relationship analysis
        entities = self._extract_entities(query, docs_result["documents"])
        
        # Step 4: Analyze relationships
        relationships = {}
        if len(entities) >= 2:
            rel_result = await self.search_agent.analyze_relationships(entities)
            if rel_result["status"] == "success":
                relationships = rel_result["relationships"]
        
        # Step 5: Generate insights
        insights_result = await self.search_agent.generate_insights(
            query,
            docs_result["documents"]
        )
        
        return {
            "status": "success",
            "documents": docs_result["documents"],
            "entities": entities,
            "relationships": relationships,
            "insights": insights_result.get("insights") if insights_result["status"] == "success" else None
        }
    
    async def _log_query(self, query: str, user_id: str, session_id: str) -> None:
        """Log the search query for analytics."""
        try:
            await self.kg_agent.query_graph(
                """
                MERGE (s:SearchSession {id: $session_id})
                ON CREATE SET s.created_at = datetime()
                CREATE (q:SearchQuery {
                    id: randomUuid(),
                    query: $query,
                    timestamp: datetime(),
                    user_id: $user_id
                })
                CREATE (q)-[:PART_OF]->(s)
                """,
                {
                    "session_id": session_id,
                    "query": query,
                    "user_id": user_id
                }
            )
        except Exception as e:
            logging.warning(f"Failed to log query: {str(e)}")
    
    def _extract_entities(self, query: str, documents: List[Dict]) -> List[str]:
        """Extract key entities from query and documents."""
        # Simple implementation - in practice, use NER or LLM
        entities = set()
        
        # Add named entities from query
        # This is a simplified version - use a proper NER model in production
        query_entities = [word for word in query.split() if word.istitle()]
        entities.update(query_entities)
        
        # Add entities from document metadata
        for doc in documents:
            if "entities" in doc.get("metadata", {}):
                entities.update(doc["metadata"]["entities"])
        
        return list(entities)[:10]  # Limit to top 10 entities
```

### 3. Search API Endpoint

#### FastAPI Implementation
```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uuid

app = FastAPI(title="ScholarVerse Search API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependencies
def get_search_orchestrator():
    # Initialize with actual dependencies
    rag_service = VertexAIRagMemoryService(
        project_id=config.GOOGLE_CLOUD_PROJECT,
        location=config.VERTEX_AI_LOCATION,
        rag_corpus=config.RAG_CORPUS_RESOURCE_NAME
    )
    search_agent = DeepSearchAgent(rag_service)
    kg_agent = KnowledgeGraphAgent(
        neo4j_uri=config.NEO4J_URI,
        neo4j_user=config.NEO4J_USER,
        neo4j_password=config.NEO4J_PASSWORD
    )
    return SearchOrchestrator(search_agent, kg_agent)

# Models
class SearchRequest(BaseModel):
    query: str
    user_id: str
    session_id: Optional[str] = None

class SearchResponse(BaseModel):
    status: str
    documents: Optional[List[Dict]] = None
    entities: Optional[List[str]] = None
    relationships: Optional[Dict] = None
    insights: Optional[str] = None
    error: Optional[str] = None

# Endpoints
@app.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    orchestrator: SearchOrchestrator = Depends(get_search_orchestrator)
):
    """Handle search requests."""
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Process the search query
        result = await orchestrator.process_query(
            query=request.query,
            user_id=request.user_id,
            session_id=session_id
        )
        
        return {
            "status": "success",
            "session_id": session_id,
            **result
        }
        
    except Exception as e:
        logging.error(f"Search failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "error": str(e)}
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Implementation Tasks

### 1. Search Infrastructure
- [ ] Set up Vertex AI RAG Engine
- [ ] Configure document ingestion pipeline
- [ ] Implement semantic search capabilities

### 2. Knowledge Integration
- [ ] Connect to knowledge graph
- [ ] Implement entity extraction and linking
- [ ] Create relationship analysis tools

### 3. API Development
- [ ] Implement RESTful API endpoints
- [ ] Add authentication and rate limiting
- [ ] Create API documentation

## Testing Strategy

### Unit Tests
- Search functionality
- Entity extraction
- Relationship analysis

### Integration Tests
- End-to-end search workflow
- Knowledge graph integration
- API endpoints

## Dependencies

```toml
[tool.poetry.dependencies]
fastapi = "^0.100.0"
uvicorn = {extras = ["standard"], version = "^0.23.0"}
pydantic = {extras = ["email"], version = "^2.0.0"}
google-cloud-aiplatform = "^1.35.0"
neo4j = {extras = ["asyncio"], version = "^5.0.0"}
```

## Next Steps
1. Implement recommendation system
2. Develop visualization components
3. Create user feedback loop
4. Optimize search performance

---
*This document will be updated as the implementation progresses.*
