# ScholarVerse: Updated Phase 3 - Knowledge Graph Construction

*Target Completion: July 15, 2025*  
*Status: Planning*

## Overview
This phase implements an intelligent knowledge graph construction system using Google ADK, focusing on entity resolution, relationship extraction, and graph-based knowledge representation.

## Key Components

### 1. Knowledge Graph Agent

#### Base Implementation
```python
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from typing import List, Dict, Optional
from neo4j import GraphDatabase
import logging

class KnowledgeGraphAgent(LlmAgent):
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password)
        )
        
        super().__init__(
            name="knowledge_graph_agent",
            description="Manages knowledge graph construction and queries",
            model="gemini-1.5-pro",
            tools=[
                self.create_node,
                self.create_relationship,
                self.query_graph,
                self.resolve_entity
            ]
        )
    
    @FunctionTool
    async def create_node(self, label: str, properties: Dict) -> Dict:
        """Create a node in the knowledge graph."""
        query = f"""
        CREATE (n:{label} $props)
        RETURN id(n) as node_id, properties(n) as properties
        """
        with self.driver.session() as session:
            result = session.run(query, props=properties).single()
            return {"status": "success", "node_id": result["node_id"], "properties": result["properties"]}
    
    @FunctionTool
    async def create_relationship(self, 
                               node1_id: int, 
                               node2_id: int, 
                               rel_type: str,
                               properties: Optional[Dict] = None) -> Dict:
        """Create a relationship between two nodes."""
        query = f"""
        MATCH (a), (b)
        WHERE id(a) = $id1 AND id(b) = $id2
        CREATE (a)-[r:{rel_type} $props]->(b)
        RETURN type(r) as rel_type, properties(r) as properties
        """
        with self.driver.session() as session:
            result = session.run(
                query, 
                id1=node1_id, 
                id2=node2_id,
                props=properties or {}
            ).single()
            return {"status": "success", "relationship": result.data()}
    
    @FunctionTool
    async def query_graph(self, cypher_query: str, params: Optional[Dict] = None) -> Dict:
        """Execute a Cypher query on the knowledge graph."""
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, params or {})
                return {
                    "status": "success",
                    "results": [dict(record) for record in result]
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @FunctionTool
    async def resolve_entity(self, entity_name: str, entity_type: str) -> Dict:
        """Resolve an entity to a node in the knowledge graph."""
        query = """
        MATCH (n {name: $name, type: $type})
        RETURN id(n) as node_id, properties(n) as properties
        LIMIT 1
        """
        with self.driver.session() as session:
            result = session.run(query, name=entity_name, type=entity_type).single()
            if result:
                return {"status": "found", "node_id": result["node_id"], "properties": result["properties"]}
            return {"status": "not_found"}
```

### 2. Knowledge Graph Construction Workflow

#### Graph Construction Pipeline
```python
class KnowledgeGraphBuilder:
    def __init__(self, kg_agent, llm_client):
        self.kg_agent = kg_agent
        self.llm = llm_client
    
    async def process_document(self, document: Dict) -> Dict:
        """Process a document and update the knowledge graph."""
        # 1. Create document node
        doc_node = await self.kg_agent.create_node(
            label="Document",
            properties={
                "title": document["title"],
                "authors": document.get("authors", []),
                "publication_date": document.get("publication_date"),
                "source": document.get("source", ""),
                "processed_at": datetime.utcnow().isoformat()
            }
        )
        
        # 2. Process authors
        author_nodes = []
        for author in document.get("authors", []):
            # Check if author exists
            author_node = await self.kg_agent.resolve_entity(author, "Author")
            if author_node["status"] == "not_found":
                author_node = await self.kg_agent.create_node(
                    label="Author",
                    properties={"name": author, "type": "Author"}
                )
            else:
                author_node = {"node_id": author_node["node_id"]}
            
            # Create authored_by relationship
            await self.kg_agent.create_relationship(
                author_node["node_id"],
                doc_node["node_id"],
                "AUTHORED_BY"
            )
            author_nodes.append(author_node)
        
        # 3. Process concepts and entities
        for entity in document.get("entities", []):
            entity_node = await self.kg_agent.resolve_entity(entity["name"], entity["type"])
            if entity_node["status"] == "not_found":
                entity_node = await self.kg_agent.create_node(
                    label=entity["type"],
                    properties={
                        "name": entity["name"],
                        "type": entity["type"],
                        "description": entity.get("description", "")
                    }
                )
            
            # Create MENTIONED_IN relationship
            await self.kg_agent.create_relationship(
                entity_node["node_id"],
                doc_node["node_id"],
                "MENTIONED_IN",
                {"context": entity.get("context", "")}
            )
        
        # 4. Process citations
        for citation in document.get("citations", []):
            # Try to find cited document
            cited_doc = await self.kg_agent.query_graph(
                """
                MATCH (d:Document {title: $title})
                RETURN id(d) as node_id
                LIMIT 1
                """,
                {"title": citation["title"]}
            )
            
            if cited_doc["status"] == "success" and cited_doc["results"]:
                # Create CITES relationship
                await self.kg_agent.create_relationship(
                    doc_node["node_id"],
                    cited_doc["results"][0]["node_id"],
                    "CITES",
                    {"context": citation.get("context", "")}
                )
        
        return {"status": "success", "document_id": doc_node["node_id"]}
```

### 3. Entity Resolution Service

#### Entity Resolution
```python
class EntityResolutionService:
    def __init__(self, kg_agent, llm_client):
        self.kg_agent = kg_agent
        self.llm = llm_client
    
    async def resolve_entity(self, entity_name: str, entity_type: str, context: str = "") -> Dict:
        """Resolve an entity with potential disambiguation."""
        # First try exact match
        result = await self.kg_agent.resolve_entity(entity_name, entity_type)
        
        if result["status"] == "found":
            return result
        
        # If not found, try fuzzy matching
        similar_entities = await self.find_similar_entities(entity_name, entity_type)
        
        if not similar_entities:
            return {"status": "not_found"}
        
        # If we have context, use LLM to disambiguate
        if context:
            return await self.disambiguate_with_llm(entity_name, entity_type, context, similar_entities)
        
        return {
            "status": "ambiguous",
            "candidates": similar_entities
        }
    
    async def find_similar_entities(self, name: str, entity_type: str, threshold: float = 0.7) -> List[Dict]:
        """Find similar entities using fuzzy matching."""
        query = """
        MATCH (n)
        WHERE n.type = $type
        WITH n, apoc.text.levenshteinSimilarity(
            apoc.text.clean(n.name), 
            apoc.text.clean($name)
        ) AS similarity
        WHERE similarity > $threshold
        RETURN id(n) as node_id, n.name as name, similarity
        ORDER BY similarity DESC
        LIMIT 5
        """
        result = await self.kg_agent.query_graph(
            query,
            {"name": name, "type": entity_type, "threshold": threshold}
        )
        
        return result.get("results", []) if result["status"] == "success" else []
    
    async def disambiguate_with_llm(self, name: str, entity_type: str, context: str, candidates: List[Dict]) -> Dict:
        """Use LLM to disambiguate between similar entities."""
        prompt = f"""
        Given the following context and candidate entities, 
        determine which entity is most likely being referred to.
        
        Entity to resolve: {name} ({entity_type})
        Context: {context}
        
        Candidates:
        {"\n".join(f"- {c['name']} (ID: {c['node_id']}, Similarity: {c.get('similarity', 1.0):.2f})" for c in candidates)}
        
        Return the most likely node_id or null if none match.
        """
        
        response = await self.llm.generate_content(prompt)
        # Parse response to find the best match
        # Implementation depends on LLM response format
        
        # For now, return the first candidate
        return {
            "status": "found" if candidates else "not_found",
            "node_id": candidates[0]["node_id"] if candidates else None,
            "properties": {k: v for k, v in candidates[0].items() if k != "node_id"} if candidates else {}
        }
```

## Implementation Tasks

### 1. Knowledge Graph Schema
- [ ] Design node and relationship types
- [ ] Define property schemas
- [ ] Implement indexing strategy

### 2. Entity Resolution
- [ ] Implement fuzzy matching
- [ ] Create disambiguation workflows
- [ ] Handle entity merging

### 3. Graph Construction
- [ ] Implement document processing pipeline
- [ ] Create relationship extraction
- [ ] Handle incremental updates

## Testing Strategy

### Unit Tests
- Node creation and retrieval
- Relationship management
- Query execution

### Integration Tests
- End-to-end graph construction
- Entity resolution accuracy
- Performance with large datasets

## Dependencies

```toml
[tool.poetry.dependencies]
neo4j = "^5.0.0"
neo4j[asyncio] = "^5.0.0"
apoc = "^0.0.4"  # For graph algorithms
python-levenshtein = "^0.12.2"  # For fuzzy matching
```

## Next Steps
1. Implement cross-paper analysis
2. Develop deep search capabilities
3. Create visualization components
4. Build recommendation system

---
*This document will be updated as the implementation progresses.*
