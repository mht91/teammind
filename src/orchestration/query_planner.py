"""
Query Planner - Understands and plans retrieval for user queries
"""
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class QueryPlan:
    """Query plan for retrieval"""
    intent: str
    topics: List[str]
    filters: Dict[str, Any]
    modalities: List[str]  # text, image, meeting, slide, document
    query_rewrites: List[str]
    plan_type: str = "retrieval"


class QueryPlanner:
    """Plan retrieval strategies for user queries"""
    
    def __init__(self, llm_model=None):
        self.llm_model = llm_model
        
    def plan(self, query: str, context: Optional[Dict] = None) -> QueryPlan:
        """
        Create a retrieval plan for the query
        
        Args:
            query: User query
            context: Additional context
        
        Returns:
            QueryPlan with retrieval strategy
        """
        try:
            # Parse query intent
            intent, topics, filters = self._parse_query(query)
            
            # Generate query rewrites
            rewrites = self._generate_rewrites(query, intent)
            
            # Determine which modalities to search
            modalities = self._determine_modalities(query, intent)
            
            return QueryPlan(
                intent=intent,
                topics=topics,
                filters=filters,
                modalities=modalities,
                query_rewrites=rewrites,
                plan_type="retrieval"
            )
            
        except Exception as e:
            logger.error(f"Error in query planning: {e}")
            return QueryPlan(
                intent="general",
                topics=[query],
                filters={},
                modalities=["text"],
                query_rewrites=[query]
            )
    
    def _parse_query(self, query: str) -> Tuple[str, List[str], Dict[str, Any]]:
        """Parse query to extract intent, topics, and filters"""
        query_lower = query.lower()
        
        # Determine intent
        intent = "general"
        if any(word in query_lower for word in ['decision', 'decide', 'approve']):
            intent = "decision"
        elif any(word in query_lower for word in ['risk', 'issue', 'problem']):
            intent = "risk"
        elif any(word in query_lower for word in ['summary', 'summarize', 'overview']):
            intent = "summary"
        elif any(word in query_lower for word in ['action', 'task', 'todo']):
            intent = "action"
        elif any(word in query_lower for word in ['schedule', 'timeline', 'when']):
            intent = "temporal"
        
        # Extract potential topics and filters
        topics = [query]  # Default
        filters = {}
        
        # Extract time filters
        import re
        time_pattern = r'\b(?:on|in|during)\s+(\w+\s+\d{4})'
        time_match = re.search(time_pattern, query)
        if time_match:
            filters['date'] = time_match.group(1)
        
        # Extract department filters
        dept_pattern = r'\b(?:from|in)\s+(\w+)\s+(?:department|dept|team)'
        dept_match = re.search(dept_pattern, query_lower)
        if dept_match:
            filters['department'] = dept_match.group(1).capitalize()
        
        return intent, topics, filters
    
    def _generate_rewrites(self, query: str, intent: str) -> List[str]:
        """Generate query rewrites for better retrieval"""
        rewrites = [query]  # Start with original
        
        # If LLM is available, use it for rewrites
        if self.llm_model:
            try:
                prompt = f"""
                Generate 3 alternative versions of this query to improve retrieval:
                Query: {query}
                Intent: {intent}
                
                Each version should preserve meaning but use different phrasing.
                """
                response = self._get_llm_response(prompt)
                lines = response.strip().split('\n')
                for line in lines[:3]:
                    if line.strip():
                        rewrites.append(line.strip())
            except:
                pass
        
        return rewrites
    
    def _determine_modalities(self, query: str, intent: str) -> List[str]:
        """Determine which modalities to search"""
        modalities = ["text"]  # Always search text
        
        query_lower = query.lower()
        
        # Search images if query mentions visual elements
        if any(word in query_lower for word in ['chart', 'graph', 'image', 'figure', 'slide']):
            modalities.append("image")
            modalities.append("slide")
        
        # Search meetings if query mentions temporal elements
        if any(word in query_lower for word in ['meeting', 'discussion', 'said', 'talk']):
            modalities.append("meeting")
        
        # Search documents for detailed information
        if any(word in query_lower for word in ['document', 'report', 'file', 'pdf']):
            modalities.append("document")
        
        return list(set(modalities))  # Remove duplicates