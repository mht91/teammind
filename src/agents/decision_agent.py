"""
Decision Agent with Groq LLM
"""
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentOutput
import re
import json
from loguru import logger


class DecisionAgent(BaseAgent):
    """Agent responsible for extracting decisions using Groq"""
    
    def __init__(self, 
                 groq_api_key: Optional[str] = None,
                 groq_model: str = "llama-3.3-70b-versatile"):
        super().__init__("DecisionAgent", groq_api_key, groq_model)
    
    def process(self, context: Dict[str, Any]) -> AgentOutput:
        """
        Extract decisions from meeting context
        
        Args:
            context: Dict containing meeting transcript and metadata
        
        Returns:
            AgentOutput with decisions
        """
        try:
            meeting_text = context.get('meeting_text', '')
            retrieved_chunks = context.get('retrieved_chunks', [])
            
            if not meeting_text:
                meeting_text = self._extract_from_chunks(retrieved_chunks)
            
            if not meeting_text:
                return AgentOutput(
                    type="decision",
                    content=[],
                    confidence=0.0
                )
            
            # Use LLM for decision extraction
            decisions = self._extract_decisions_with_llm(meeting_text)
            
            # Fallback to pattern matching if LLM fails
            if not decisions:
                decisions = self._extract_decisions_with_patterns(meeting_text)
            
            return AgentOutput(
                type="decision",
                content=decisions,
                confidence=0.85 if decisions else 0.0,
                metadata={
                    'decision_count': len(decisions)
                },
                reasoning=f"Found {len(decisions)} decisions in meeting content"
            )
            
        except Exception as e:
            logger.error(f"Error in decision agent: {e}")
            return AgentOutput(
                type="decision",
                content=[],
                confidence=0.0
            )
    
    def _extract_from_chunks(self, chunks: List) -> str:
        """Extract meeting text from retrieved chunks"""
        texts = []
        for chunk in chunks:
            payload = chunk.get('payload', {})
            if 'text' in payload:
                texts.append(payload['text'])
            elif 'content' in payload:
                texts.append(payload['content'])
        return ' '.join(texts)
    
    def _extract_decisions_with_llm(self, text: str) -> List[Dict[str, Any]]:
        """Extract decisions using Groq LLM"""
        system_prompt = """You are a decision extraction expert. Analyze meeting text and extract all decisions made.
        For each decision, provide:
        - text: The decision description
        - type: 'approved', 'rejected', 'postponed', or 'recorded'
        - confidence: 0-1 score
        - reasoning: Brief explanation
        
        Return as a JSON array."""
        
        user_prompt = f"""
Extract all decisions from the following meeting text:

{text[:3000]}...

Return the decisions as a JSON array. If no decisions are found, return [].
"""
        
        response = self._get_llm_response(system_prompt, user_prompt)
        
        if not response:
            return []
        
        try:
            # Parse JSON response
            decisions = self._parse_json_response(response)
            if isinstance(decisions, list):
                return decisions
            elif isinstance(decisions, dict) and 'decisions' in decisions:
                return decisions['decisions']
            return []
        except:
            return []
    
    def _extract_decisions_with_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract decisions using pattern matching (fallback)"""
        decisions = []
        
        decision_patterns = [
            r'(?:we|team|I)\s+(?:decided|agreed|approved|resolved|concluded)\s+(?:that|to)?\s+([^.!?]+[.!?])',
            r'(?:the|The)\s+(?:decision|conclusion)\s+(?:was|is)\s+(?:that|to)?\s+([^.!?]+[.!?])',
            r'(?:it was|It was)\s+(?:decided|agreed)\s+(?:that|to)?\s+([^.!?]+[.!?])',
            r'(?:we will|We will)\s+([^.!?]+[.!?])',
            r'(?:approved|Approved)\s+(?:the|the)?\s+([^.!?]+[.!?])',
        ]
        
        for pattern in decision_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                decisions.append({
                    'text': match.strip(),
                    'type': self._classify_decision(match),
                    'confidence': 0.7
                })
        
        return decisions
    
    def _classify_decision(self, text: str) -> str:
        """Classify the type of decision"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['approve', 'approval', 'approved']):
            return 'approved'
        elif any(word in text_lower for word in ['reject', 'rejected', 'deny']):
            return 'rejected'
        elif any(word in text_lower for word in ['postpone', 'delayed', 'later']):
            return 'postponed'
        else:
            return 'recorded'