"""
Context Fusion Module - Combines retrieved information from multiple modalities
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import json
from loguru import logger


@dataclass
class FusedContext:
    """Fused context from multiple modalities"""
    text: str
    modalities: List[str]
    metadata: Dict[str, Any]
    source_chunks: List[Dict[str, Any]]
    agent_inputs: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility"""
        return {
            'text': self.text,
            'modalities': self.modalities,
            'metadata': self.metadata,
            'source_chunks': self.source_chunks,
            'agent_inputs': self.agent_inputs
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get method for compatibility"""
        if hasattr(self, key):
            return getattr(self, key)
        return default


class ContextFusion:
    """Fuse retrieved information from different modalities"""
    
    def __init__(self, max_context_length: int = 4000):
        self.max_context_length = max_context_length
    
    def fuse(self, retrieved_chunks: List[Dict[str, Any]], 
             query_plan: Any) -> FusedContext:
        """
        Fuse retrieved chunks into a unified context
        
        Args:
            retrieved_chunks: List of retrieved chunks
            query_plan: Query plan with intent and modalities
        
        Returns:
            FusedContext with combined information
        """
        try:
            # Separate chunks by modality
            modalities = {
                'text': [],
                'meeting': [],
                'image': [],
                'slide': [],
                'document': [],
                'email': [],
                'chat': []
            }
            
            for chunk in retrieved_chunks:
                payload = chunk.get('payload', {})
                modality = payload.get('modality', 'text')
                
                if modality in modalities:
                    modalities[modality].append(chunk)
                else:
                    modalities['text'].append(chunk)
            
            # Build context text
            context_parts = []
            
            # Priority order: meeting > slide > document > email > chat > image > text
            priority_order = ['meeting', 'slide', 'document', 'email', 'chat', 'image', 'text']
            
            for modality in priority_order:
                chunks = modalities.get(modality, [])
                if chunks:
                    context_parts.append(f"--- {modality.upper()} ---")
                    for chunk in chunks:
                        text = self._extract_text(chunk)
                        if text:
                            context_parts.append(text)
            
            # Combine with truncation
            full_context = '\n'.join(context_parts)
            
            # Truncate if needed
            if len(full_context) > self.max_context_length:
                full_context = full_context[:self.max_context_length] + "..."
            
            # Prepare agent inputs
            agent_inputs = self._prepare_agent_inputs(retrieved_chunks)
            
            return FusedContext(
                text=full_context,
                modalities=list(modalities.keys()),
                metadata={
                    'chunk_count': len(retrieved_chunks),
                    'modalities_present': [m for m, c in modalities.items() if c]
                },
                source_chunks=retrieved_chunks,
                agent_inputs=agent_inputs
            )
            
        except Exception as e:
            logger.error(f"Error in context fusion: {e}")
            return FusedContext(
                text="",
                modalities=[],
                metadata={'error': str(e)},
                source_chunks=[],
                agent_inputs={}
            )
    
    def _extract_text(self, chunk: Dict[str, Any]) -> str:
        """Extract text from chunk"""
        payload = chunk.get('payload', {})
        
        if 'text' in payload:
            return payload['text']
        elif 'content' in payload:
            return payload['content']
        elif 'description' in payload:
            return payload['description']
        elif 'transcript' in payload:
            return payload['transcript']
        else:
            return ''
    
    def _prepare_agent_inputs(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare inputs for each agent"""
        agent_inputs = {
            'summary': {},
            'decision': {},
            'risk': {},
            'task': {},
            'weak_signal': {},
            'knowledge_graph': {}
        }
        
        # Extract meeting transcript if available
        meeting_text = []
        for chunk in chunks:
            payload = chunk.get('payload', {})
            if payload.get('modality') == 'meeting':
                if 'text' in payload:
                    meeting_text.append(payload['text'])
                elif 'content' in payload:
                    meeting_text.append(payload['content'])
        
        if meeting_text:
            full_meeting_text = ' '.join(meeting_text)
            agent_inputs['summary']['meeting_text'] = full_meeting_text
            agent_inputs['decision']['meeting_text'] = full_meeting_text
            agent_inputs['risk']['meeting_text'] = full_meeting_text
            agent_inputs['task']['meeting_text'] = full_meeting_text
            agent_inputs['weak_signal']['meeting_text'] = full_meeting_text
            agent_inputs['knowledge_graph']['meeting_text'] = full_meeting_text
        
        # Add all chunks for agents
        modalities = list(set([c.get('payload', {}).get('modality', 'text') for c in chunks]))
        for agent in agent_inputs:
            agent_inputs[agent]['retrieved_chunks'] = chunks
            agent_inputs[agent]['metadata'] = {'modalities': modalities}
        
        return agent_inputs