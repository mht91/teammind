"""
Weak Signal Agent with Groq LLM
"""
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentOutput
from collections import Counter
import re
import json
from loguru import logger


class WeakSignalAgent(BaseAgent):
    """Agent responsible for detecting emerging patterns using Groq"""
    
    def __init__(self, 
                 groq_api_key: Optional[str] = None,
                 groq_model: str = "llama-3.3-70b-versatile"):
        super().__init__("WeakSignalAgent", groq_api_key, groq_model)
    
    def process(self, context: Dict[str, Any]) -> AgentOutput:
        """
        Detect weak signals and emerging patterns
        
        Args:
            context: Dict containing meeting data
        
        Returns:
            AgentOutput with weak signals
        """
        try:
            meeting_text = context.get('meeting_text', '')
            retrieved_chunks = context.get('retrieved_chunks', [])
            
            if not meeting_text:
                meeting_text = self._extract_from_chunks(retrieved_chunks)
            
            if not meeting_text:
                return AgentOutput(
                    type="weak_signal",
                    content=[],
                    confidence=0.0
                )
            
            # Use LLM for weak signal detection
            signals = self._detect_signals_with_llm(meeting_text)
            
            # Fallback to pattern matching
            if not signals:
                signals = self._detect_signals_with_patterns(meeting_text)
            
            return AgentOutput(
                type="weak_signal",
                content=signals,
                confidence=0.7 if signals else 0.0,
                metadata={
                    'signal_count': len(signals),
                    'categories': list(set(s.get('category', 'general') for s in signals))
                }
            )
            
        except Exception as e:
            logger.error(f"Error in weak signal agent: {e}")
            return AgentOutput(
                type="weak_signal",
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
    
    def _detect_signals_with_llm(self, text: str) -> List[Dict[str, Any]]:
        """Detect weak signals using Groq LLM"""
        system_prompt = """You are an organizational health analyst. Identify weak signals in meeting text.
        Weak signals are subtle indicators of emerging issues, trends, or opportunities.
        
        For each signal, provide:
        - text: Description of the signal
        - category: 'customer', 'workforce', 'financial', 'operational', 'market', 'general'
        - strength: 0-1 score
        - reasoning: Brief explanation
        
        Return as a JSON array."""
        
        user_prompt = f"""
Identify weak signals in the following meeting text:

{text[:3000]}...

Return the signals as a JSON array. If no weak signals are found, return [].
"""
        
        response = self._get_llm_response(system_prompt, user_prompt)
        
        if not response:
            return []
        
        try:
            signals = self._parse_json_response(response)
            if isinstance(signals, list):
                return signals
            elif isinstance(signals, dict) and 'signals' in signals:
                return signals['signals']
            return []
        except:
            return []
    
    def _detect_signals_with_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Detect weak signals using pattern matching (fallback)"""
        signals = []
        signal_keywords = [
            'concern', 'worry', 'anxiety', 'frustration', 'dissatisfaction',
            'uncertainty', 'confusion', 'lack of', 'shortage', 'insufficient',
            'overwhelmed', 'burnout', 'stress', 'pressure', 'tension',
            'conflict', 'disagreement', 'misalignment', 'gap', 'disconnect'
        ]
        
        sentences = re.split(r'[.!?]+', text)
        signal_counter = Counter()
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            signal_score = 0
            detected_keywords = []
            
            for keyword in signal_keywords:
                if keyword in sentence_lower:
                    signal_score += 1
                    detected_keywords.append(keyword)
                    signal_counter[keyword] += 1            
            if signal_score >= 2:
                signals.append({
                    'text': sentence.strip(),
                    'category': self._categorize_signal(sentence_lower),
                    'strength': min(signal_score / 3, 1.0),
                    'keywords': detected_keywords,
                    'frequency': signal_counter.most_common(1)[0][1] if signal_counter else 0,
                    'reasoning': f"Found {signal_score} signal indicators"
                })
        
        return signals
    
    def _categorize_signal(self, text: str) -> str:
        """Categorize the type of signal"""
        if any(word in text for word in ['customer', 'client', 'user']):
            return 'customer'
        elif any(word in text for word in ['team', 'staff', 'employee']):
            return 'workforce'
        elif any(word in text for word in ['budget', 'cost', 'financial']):
            return 'financial'
        elif any(word in text for word in ['timeline', 'delay', 'schedule']):
            return 'operational'
        elif any(word in text for word in ['market', 'competitor', 'industry']):
            return 'market'
        else:
            return 'general'