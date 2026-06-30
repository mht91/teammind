"""
Knowledge Graph Agent with Groq LLM
"""
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentOutput
from neo4j import GraphDatabase
import re
import json
from loguru import logger


class KnowledgeGraphAgent(BaseAgent):
    """Agent responsible for building knowledge graphs using Groq"""
    
    def __init__(self, 
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password",
                 groq_api_key: Optional[str] = None,
                 groq_model: str = "llama-3.3-70b-versatile"):
        super().__init__("KnowledgeGraphAgent", groq_api_key, groq_model)
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.driver = None
        
        try:
            self.driver = GraphDatabase.driver(
                neo4j_uri, 
                auth=(neo4j_user, neo4j_password)
            )
            logger.info(f"Connected to Neo4j at {neo4j_uri}")
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j: {e}")
    
    def process(self, context: Dict[str, Any]) -> AgentOutput:
        """
        Build knowledge graph from meeting content
        
        Args:
            context: Dict containing meeting data
        
        Returns:
            AgentOutput with graph insights
        """
        try:
            meeting_text = context.get('meeting_text', '')
            retrieved_chunks = context.get('retrieved_chunks', [])
            
            if not meeting_text:
                meeting_text = self._extract_from_chunks(retrieved_chunks)
            
            if not meeting_text:
                return AgentOutput(
                    type="knowledge_graph",
                    content={},
                    confidence=0.0
                )
            
            # Use LLM for entity and relationship extraction
            entities, relationships = self._extract_graph_with_llm(meeting_text)
            
            # Fallback to pattern matching
            if not entities:
                entities = self._extract_entities_with_patterns(meeting_text)
                relationships = self._extract_relationships_with_patterns(meeting_text, entities)
            
            # Store in Neo4j if available
            if self.driver:
                self._store_in_graph(entities, relationships)
            
            # Query for insights
            insights = self._query_graph_insights(entities)
            
            return AgentOutput(
                type="knowledge_graph",
                content={
                    'entities': entities,
                    'relationships': relationships,
                    'insights': insights
                },
                confidence=0.7 if entities else 0.0,
                metadata={
                    'entity_count': len(entities),
                    'relationship_count': len(relationships)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in knowledge graph agent: {e}")
            return AgentOutput(
                type="knowledge_graph",
                content={},
                confidence=0.0
            )
    
    def _extract_from_chunks(self, chunks: List) -> str:
        """Extract text from chunks"""
        texts = []
        for chunk in chunks:
            payload = chunk.get('payload', {})
            if 'text' in payload:
                texts.append(payload['text'])
            elif 'content' in payload:
                texts.append(payload['content'])
        return ' '.join(texts)
    
    def _extract_graph_with_llm(self, text: str) -> tuple:
        """Extract entities and relationships using Groq LLM"""
        system_prompt = """You are a knowledge graph extraction expert. Extract entities and relationships from meeting text.
        
        Entities should include:
        - People (names)
        - Organizations (companies, teams, departments)
        - Projects
        - Topics
        
        Relationships should include:
        - Source entity
        - Target entity
        - Type: 'assigned_to', 'led', 'reports_to', 'belongs_to', 'created', 'working_on'
        
        Return as JSON with format:
        {
            "entities": [{"name": "...", "type": "..."}],
            "relationships": [{"source": "...", "target": "...", "type": "..."}]
        }"""
        
        user_prompt = f"""
Extract entities and relationships from the following meeting text:

{text[:3000]}...

Return as a JSON object with 'entities' and 'relationships' arrays.
"""
        
        response = self._get_llm_response(system_prompt, user_prompt)
        
        if not response:
            return [], []
        
        try:
            data = self._parse_json_response(response)
            if data:
                entities = data.get('entities', [])
                relationships = data.get('relationships', [])
                return entities, relationships
        except:
            pass
        
        return [], []
    
    def _extract_entities_with_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using pattern matching (fallback)"""
        entities = []
        
        # Person names
        person_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        for match in re.findall(person_pattern, text):
            entities.append({
                'name': match,
                'type': 'person',
                'confidence': 0.7
            })
        
        # Organizations
        org_pattern = r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+(?:Corp|Inc|Ltd|LLC|Company|Team|Department|Group)'
        for match in re.findall(org_pattern, text):
            entities.append({
                'name': match.strip(),
                'type': 'organization',
                'confidence': 0.7
            })
        
        # Projects
        project_pattern = r'\bProject\s+([A-Z][a-zA-Z]*\s*[A-Z]?[a-zA-Z]*)\b'
        for match in re.findall(project_pattern, text):
            entities.append({
                'name': f"Project {match}",
                'type': 'project',
                'confidence': 0.6
            })
        
        return entities
    
    def _extract_relationships_with_patterns(self, text: str, entities: List) -> List[Dict]:
        """Extract relationships using pattern matching (fallback)"""
        relationships = []
        
        relationship_patterns = [
            (r'([A-Z][a-z]+)\s+(?:assigned|gave|delegated)\s+(?:to|for)\s+([A-Z][a-z]+)', 'assigned_to'),
            (r'([A-Z][a-z]+)\s+(?:led|managed|oversaw)\s+([A-Z][a-zA-Z]+)', 'led'),
            (r'([A-Z][a-z]+)\s+(?:reported|reports)\s+(?:to|for)\s+([A-Z][a-z]+)', 'reports_to'),
            (r'([A-Z][a-zA-Z]+)\s+(?:belongs|part of|member of)\s+([A-Z][a-zA-Z]+)', 'belongs_to'),
            (r'([A-Z][a-zA-Z]+)\s+(?:created|launched|initiated)\s+([A-Z][a-zA-Z]+)', 'created'),
            (r'([A-Z][a-zA-Z]+)\s+(?:working on|handling)\s+([A-Z][a-zA-Z]+)', 'working_on')
        ]
        
        for pattern, rel_type in relationship_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                source, target = match
                if any(e['name'].lower() == source.lower() for e in entities) or \
                   any(e['name'].lower() == target.lower() for e in entities):
                    relationships.append({
                        'source': source,
                        'target': target,
                        'type': rel_type,
                        'confidence': 0.6
                    })
        
        return relationships
    
    def _store_in_graph(self, entities: List, relationships: List):
        """Store entities and relationships in Neo4j"""
        if not self.driver:
            return
        
        try:
            with self.driver.session() as session:
                # Create entities
                for entity in entities:
                    session.run(
                        """
                        MERGE (e:Entity {name: $name})
                        SET e.type = $type,
                            e.confidence = $confidence
                        """,
                        name=entity['name'],
                        type=entity['type'],
                        confidence=entity.get('confidence', 0.5)
                    )
                
                # Create relationships
                for rel in relationships:
                    session.run(
                        """
                        MATCH (source:Entity {name: $source})
                        MATCH (target:Entity {name: $target})
                        MERGE (source)-[r:RELATES_TO {type: $type}]->(target)
                        SET r.confidence = $confidence
                        """,
                        source=rel['source'],
                        target=rel['target'],
                        type=rel['type'],
                        confidence=rel.get('confidence', 0.5)
                    )
            
            logger.info(f"Stored {len(entities)} entities and {len(relationships)} relationships")
            
        except Exception as e:
            logger.error(f"Error storing in Neo4j: {e}")
    
    def _query_graph_insights(self, entities: List) -> List[Dict[str, Any]]:
        """Query Neo4j for graph insights"""
        if not self.driver or not entities:
            return []
        
        insights = []
        
        try:
            with self.driver.session() as session:
                # Find central nodes
                result = session.run(
                    """
                    MATCH (e:Entity)-[r:RELATES_TO]-()
                    RETURN e.name as name, count(r) as degree
                    ORDER BY degree DESC
                    LIMIT 5
                    """
                )
                
                for record in result:
                    insights.append({
                        'entity': record['name'],
                        'degree': record['degree'],
                        'type': 'central_node'
                    })
            
        except Exception as e:
            logger.error(f"Error querying Neo4j: {e}")
        
        return insights
    
    def query(self, query: str) -> List[Dict]:
        """Execute a custom Cypher query"""
        if not self.driver:
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run(query)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return []
    
    def __del__(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()