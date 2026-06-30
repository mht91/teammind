"""
Risk Agent with Groq LLM
"""
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentOutput
import re
import json
from loguru import logger


class RiskAgent(BaseAgent):
    """Agent responsible for identifying risks using Groq"""
    
    def __init__(self, 
                 groq_api_key: Optional[str] = None,
                 groq_model: str = "llama-3.3-70b-versatile"):
        super().__init__("RiskAgent", groq_api_key, groq_model)
        self.risk_keywords = [
            'risk', 'issue', 'problem', 'concern', 'challenge',
            'delay', 'budget', 'resource', 'missing', 'uncertainty',
            'threat', 'vulnerability', 'constraint', 'dependency',
            'conflict', 'misalignment', 'gap'
        ]
    
    def process(self, context: Dict[str, Any]) -> AgentOutput:
        """
        Identify risks from meeting context
        
        Args:
            context: Dict containing meeting transcript and metadata
        
        Returns:
            AgentOutput with risks
        """
        try:
            meeting_text = context.get('meeting_text', '')
            retrieved_chunks = context.get('retrieved_chunks', [])
            
            if not meeting_text:
                meeting_text = self._extract_from_chunks(retrieved_chunks)
            
            if not meeting_text:
                return AgentOutput(
                    type="risk",
                    content=[],
                    confidence=0.0
                )
            
            # Use LLM for risk extraction
            risks = self._extract_risks_with_llm(meeting_text)
            
            # Fallback to pattern matching if LLM fails
            if not risks:
                risks = self._extract_risks_with_patterns(meeting_text)
            
            # Categorize risks
            categorized_risks = self._categorize_risks(risks)
            
            return AgentOutput(
                type="risk",
                content=categorized_risks,
                confidence=0.8 if risks else 0.0,
                metadata={
                    'risk_count': len(risks),
                    'categories': list(categorized_risks.keys())
                }
            )
            
        except Exception as e:
            logger.error(f"Error in risk agent: {e}")
            return AgentOutput(
                type="risk",
                content=[],
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
    
    def _extract_risks_with_llm(self, text: str) -> List[Dict[str, Any]]:
        """Extract risks using Groq LLM"""
        system_prompt = """You are a risk assessment expert. Identify all risks mentioned in meeting text.
        For each risk, provide:
        - text: Description of the risk
        - type: 'schedule', 'financial', 'resource', 'technical', 'legal', 'general'
        - severity: 'high', 'medium', 'low'
        - confidence: 0-1 score
        
        Return as a JSON array."""
        
        user_prompt = f"""
Identify all risks in the following meeting text:

{text[:3000]}...

Return the risks as a JSON array. If no risks are found, return [].
"""
        
        response = self._get_llm_response(system_prompt, user_prompt)
        
        if not response:
            return []
        
        try:
            risks = self._parse_json_response(response)
            if isinstance(risks, list):
                return risks
            elif isinstance(risks, dict) and 'risks' in risks:
                return risks['risks']
            return []
        except:
            return []
    
    def _extract_risks_with_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract risks using pattern matching (fallback)"""
        risks = []
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            risk_score = 0
            risk_type = None
            
            for keyword in self.risk_keywords:
                if keyword in sentence_lower:
                    risk_score += 1
                    if keyword in ['delay', 'timeline', 'schedule']:
                        risk_type = 'schedule'
                    elif keyword in ['budget', 'cost', 'financial']:
                        risk_type = 'financial'
                    elif keyword in ['resource', 'staff', 'team']:
                        risk_type = 'resource'
                    elif keyword in ['technical', 'technology', 'system']:
                        risk_type = 'technical'
                    elif keyword in ['legal', 'compliance', 'regulatory']:
                        risk_type = 'legal'
            
            if risk_score >= 2:
                risks.append({
                    'text': sentence.strip(),
                    'type': risk_type or 'general',
                    'severity': 'high' if risk_score >= 3 else 'medium' if risk_score >= 2 else 'low',
                    'confidence': min(risk_score / 3, 1.0)
                })
        
        return risks
    
    def _categorize_risks(self, risks: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize risks by type"""
        categories = {
            'schedule': [],
            'financial': [],
            'resource': [],
            'technical': [],
            'legal': [],
            'general': []
        }
        
        for risk in risks:
            risk_type = risk.get('type', 'general')
            if risk_type in categories:
                categories[risk_type].append(risk)
            else:
                categories['general'].append(risk)
        
        return {k: v for k, v in categories.items() if v}