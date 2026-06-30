"""
Summary Agent with Groq LLM
"""
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentOutput
from loguru import logger


class SummaryAgent(BaseAgent):
    """Agent responsible for generating meeting summaries using Groq"""
    
    def __init__(self, 
                 groq_api_key: Optional[str] = None,
                 groq_model: str = "llama-3.3-70b-versatile",
                 max_summary_length: int = 500):
        super().__init__("SummaryAgent", groq_api_key, groq_model)
        self.max_summary_length = max_summary_length
    
    def process(self, context: Dict[str, Any]) -> AgentOutput:
        """
        Generate meeting summary from context
        
        Args:
            context: Dict containing meeting transcript and metadata
        
        Returns:
            AgentOutput with summary
        """
        try:
            meeting_text = context.get('meeting_text', '')
            metadata = context.get('metadata', {})
            retrieved_chunks = context.get('retrieved_chunks', [])
            
            if not meeting_text and not retrieved_chunks:
                return AgentOutput(
                    type="summary",
                    content="No meeting content available",
                    confidence=0.0
                )
            
            # Prepare prompts
            system_prompt = """You are an expert meeting summarizer. Generate concise, structured summaries 
            of business meetings. Focus on key decisions, action items, and important discussions."""
            
            user_prompt = self._build_summary_prompt(meeting_text, metadata, retrieved_chunks)
            
            # Get LLM response
            summary = self._get_llm_response(system_prompt, user_prompt)
            
            if not summary:
                return AgentOutput(
                    type="summary",
                    content={"error": "Failed to generate summary"},
                    confidence=0.0
                )
            
            # Extract sections
            sections = self._extract_sections(summary)
            
            return AgentOutput(
                type="summary",
                content={
                    'full_summary': summary,
                    'sections': sections,
                    'key_points': self._extract_key_points(summary)
                },
                confidence=0.85,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error in summary agent: {e}")
            return AgentOutput(
                type="summary",
                content={"error": str(e)},
                confidence=0.0
            )
    
    def _build_summary_prompt(self, meeting_text: str, metadata: Dict, 
                             retrieved_chunks: List) -> str:
        """Build prompt for summary generation"""
        context_parts = []
        
        if metadata.get('date'):
            context_parts.append(f"Date: {metadata['date']}")
        if metadata.get('meeting_type'):
            context_parts.append(f"Type: {metadata['meeting_type']}")
        if metadata.get('participants'):
            context_parts.append(f"Participants: {', '.join(metadata['participants'])}")
        
        context = '\n'.join(context_parts)
        
        # Truncate meeting text if too long
        truncated_text = meeting_text[:3000] + "..." if len(meeting_text) > 3000 else meeting_text
        
        prompt = f"""
Meeting Context:
{context}

Meeting Transcript:
{truncated_text}

Additional Retrieved Information:
{self._format_retrieved_chunks(retrieved_chunks)}

Please generate a comprehensive summary that includes:
1. Main purpose and agenda
2. Key discussion points
3. Major decisions made
4. Action items identified
5. Next steps

Format the summary with clear sections and bullet points for key items.
"""
        return prompt
    
    def _format_retrieved_chunks(self, chunks: List) -> str:
        """Format retrieved chunks for prompt"""
        if not chunks:
            return "No additional context"
        
        formatted = []
        for i, chunk in enumerate(chunks[:5]):
            text = chunk.get('payload', {}).get('text', '')
            if text:
                formatted.append(f"Chunk {i+1}: {text[:300]}...")
        
        return '\n'.join(formatted)
    
    def _extract_sections(self, summary: str) -> Dict[str, str]:
        """Extract sections from summary"""
        sections = {
            'purpose': '',
            'discussion': '',
            'decisions': '',
            'actions': '',
            'next_steps': ''
        }
        
        lines = summary.split('\n')
        current_section = None
        section_content = []
        
        for line in lines:
            line_lower = line.lower()
            if 'purpose' in line_lower or 'agenda' in line_lower:
                current_section = 'purpose'
            elif 'discussion' in line_lower:
                current_section = 'discussion'
            elif 'decision' in line_lower:
                current_section = 'decisions'
            elif 'action' in line_lower:
                current_section = 'actions'
            elif 'next' in line_lower:
                current_section = 'next_steps'
            elif current_section and line.strip() and not line.startswith('#'):
                sections[current_section] += line.strip() + ' '
        
        return sections
    
    def _extract_key_points(self, summary: str) -> List[str]:
        """Extract key points from summary"""
        import re
        points = []
        
        # Find bullet points
        bullet_pattern = r'[-•*]\s*(.*?)(?=\n|$)'
        bullets = re.findall(bullet_pattern, summary)
        points.extend([b.strip() for b in bullets if b.strip()])
        
        # Find numbered points
        numbered_pattern = r'\d+\.\s*(.*?)(?=\n|$)'
        numbered = re.findall(numbered_pattern, summary)
        points.extend([n.strip() for n in numbered if n.strip()])
        
        return points[:10]