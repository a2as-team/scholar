# ScholarVerse: Updated Phase 1 - Foundation & Core ADK Integration

*Target Completion: June 15, 2025*  
*Status: In Progress*

## Overview
This phase establishes the core infrastructure of ScholarVerse using Google ADK's advanced features, focusing on session management, state handling, and the foundational agent architecture.

## Key Components

### 1. ADK Core Integration
- **Session Management**
  - Implement `VertexAiSessionService` for production-grade session handling
  - Session state management with automatic persistence
  - User and application state isolation
  
- **Memory Services**
  - Integration with `VertexAiRagMemoryService`
  - Long-term knowledge retention
  - Semantic search capabilities

### 2. Agent Architecture

#### Base Agent Implementation
```python
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.sessions import VertexAiSessionService
from google.adk.memory import VertexAiRagMemoryService

class ScholarVerseAgent(LlmAgent):
    def __init__(self, name, description, tools=None):
        self.session_service = VertexAiSessionService(
            project_id=config.GOOGLE_CLOUD_PROJECT,
            location=config.VERTEX_AI_LOCATION
        )
        
        self.memory_service = VertexAiRagMemoryService(
            rag_corpus=config.RAG_CORPUS_RESOURCE_NAME,
            similarity_top_k=5
        )
        
        super().__init__(
            name=name,
            description=description,
            model="gemini-1.5-pro",
            tools=tools or [],
            state_schema={
                'type': 'object',
                'properties': {
                    'research_goals': {'type': 'array', 'items': {'type': 'string'}},
                    'findings': {'type': 'array', 'items': {'type': 'string'}},
                    'citations': {'type': 'array', 'items': {'type': 'string'}}
                }
            }
        )
```

### 3. State Management Implementation

#### Session State Handler
```python
class SessionStateManager:
    def __init__(self, session_service):
        self.session_service = session_service
    
    async def initialize_session(self, user_id, session_id):
        """Initialize a new session with default state."""
        session = await self.session_service.create_session(
            app_name="scholarverse",
            user_id=user_id,
            session_id=session_id,
            state={
                'research_context': {},
                'preferences': {},
                'conversation_history': []
            }
        )
        return session
    
    async def update_state(self, session, updates):
        """Update session state with new values."""
        await session.state.update(updates)
        return await self.session_service.update_session(session)
```

### 4. Memory Management

#### Knowledge Graph Integration
```python
class KnowledgeGraphManager:
    def __init__(self, memory_service):
        self.memory_service = memory_service
    
    async def add_research_context(self, session, content, metadata=None):
        """Add research context to the knowledge base."""
        return await self.memory_service.add_session_to_memory(
            session,
            content=content,
            metadata=metadata or {}
        )
    
    async def search_knowledge(self, query, user_id, top_k=5):
        """Search across user's research history."""
        return await self.memory_service.search_memory(
            app_name="scholarverse",
            user_id=user_id,
            query=query,
            top_k=top_k
        )
```

## Implementation Tasks

### 1. Core Infrastructure
- [ ] Set up Google Cloud project with required APIs
- [ ] Configure Vertex AI and RAG corpus
- [ ] Implement base agent class with ADK integration
- [ ] Set up session and state management
- [ ] Implement memory service with RAG integration

### 2. Development Environment
- [ ] Configure Poetry for dependency management
- [ ] Set up testing framework with pytest
- [ ] Implement CI/CD pipeline
- [ ] Configure monitoring and logging

### 3. Documentation
- [ ] Document architecture decisions
- [ ] Create API documentation
- [ ] Write setup and deployment guides

## Testing Strategy

### Unit Tests
- Session state management
- Memory service operations
- Agent initialization and configuration

### Integration Tests
- End-to-end agent workflows
- Session persistence
- Memory retrieval accuracy

## Dependencies

```toml
[tool.poetry.dependencies]
python = "^3.9"
google-adk = {extras = ["vertexai"], version = "^1.0.0"
google-cloud-aiplatform = "^1.35.0"
pydantic = "^2.0.0"
pytest-asyncio = "^0.21.0"
```

## Next Steps
1. Implement authentication and authorization
2. Develop the document ingestion pipeline
3. Build the knowledge graph construction workflow
4. Implement the deep search functionality

---
*This document will be updated as the implementation progresses.*
