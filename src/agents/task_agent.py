"""
Task Agent with Groq LLM
"""
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentOutput
import re
import json
from loguru import logger


class TaskAgent(BaseAgent):
    """Agent responsible for extracting action items using Groq"""
    
    def __init__(self, 
                 groq_api_key: Optional[str] = None,
                 groq_model: str = "llama-3.3-70b-versatile"):
        super().__init__("TaskAgent", groq_api_key, groq_model)
    
    def process(self, context: Dict[str, Any]) -> AgentOutput:
        """
        Extract action items from meeting context
        
        Args:
            context: Dict containing meeting transcript and metadata
        
        Returns:
            AgentOutput with tasks
        """
        try:
            meeting_text = context.get('meeting_text', '')
            retrieved_chunks = context.get('retrieved_chunks', [])
            
            if not meeting_text:
                meeting_text = self._extract_from_chunks(retrieved_chunks)
            
            if not meeting_text:
                return AgentOutput(
                    type="task",
                    content=[],
                    confidence=0.0
                )
            
            # Use LLM for task extraction
            tasks = self._extract_tasks_with_llm(meeting_text)
            
            # Fallback to pattern matching if LLM fails
            if not tasks:
                tasks = self._extract_tasks_with_patterns(meeting_text)
            
            # Structure tasks
            structured_tasks = self._structure_tasks(tasks)
            
            return AgentOutput(
                type="task",
                content=structured_tasks,
                confidence=0.8 if tasks else 0.0,
                metadata={
                    'task_count': len(tasks),
                    'has_owners': any(t.get('owner') for t in tasks),
                    'has_deadlines': any(t.get('deadline') for t in tasks)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in task agent: {e}")
            return AgentOutput(
                type="task",
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
    
    def _extract_tasks_with_llm(self, text: str) -> List[Dict[str, Any]]:
        """Extract tasks using Groq LLM"""
        system_prompt = """You are a task extraction expert. Extract all action items from meeting text.
        For each task, provide:
        - title: Brief task description
        - owner: Person responsible (if mentioned)
        - deadline: Due date (if mentioned)
        - status: 'pending', 'in_progress', or 'completed'
        - priority: 'high', 'medium', 'low'
        - confidence: 0-1 score
        
        Return as a JSON array."""
        
        user_prompt = f"""
Extract all action items from the following meeting text:

{text[:3000]}...

Return the tasks as a JSON array. If no tasks are found, return [].
"""
        
        response = self._get_llm_response(system_prompt, user_prompt)
        
        if not response:
            return []
        
        try:
            tasks = self._parse_json_response(response)
            if isinstance(tasks, list):
                return tasks
            elif isinstance(tasks, dict) and 'tasks' in tasks:
                return tasks['tasks']
            return []
        except:
            return []
    
    def _extract_tasks_with_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract tasks using pattern matching (fallback)"""
        tasks = []
        sentences = re.split(r'[.!?]+', text)
        
        task_indicators = [
            'action item', 'todo', 'to-do', 'task', 'assign',
            'responsible', 'owner', 'deadline', 'by when',
            'need to', 'should', 'must', 'will do'
        ]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in task_indicators):
                task = {
                    'title': self._extract_title(sentence),
                    'owner': self._extract_owner(sentence),
                    'deadline': self._extract_deadline(sentence),
                    'status': 'pending',
                    'priority': 'medium',
                    'confidence': 0.6
                }
                tasks.append(task)
        
        return tasks
    
    def _extract_title(self, text: str) -> str:
        """Extract task title from text"""
        task_indicators = [
            'action item', 'todo', 'to-do', 'task', 'assign'
        ]
        title = text
        for indicator in task_indicators:
            if indicator in text.lower():
                title = text.split(indicator, 1)[-1].strip()
                break
        return title[:100]
    
    def _extract_owner(self, text: str) -> Optional[str]:
        """Extract task owner from text"""
        owner_patterns = [
            r'(?:assigned to|by|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:owner|responsible):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:will|to|shall)'
        ]
        
        for pattern in owner_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def _extract_deadline(self, text: str) -> Optional[str]:
        """Extract deadline from text"""
        deadline_patterns = [
            r'(?:deadline|by|due)\s+(?:on|by)?\s*([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})',
            r'(?:by|until)\s+([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?)',
            r'(\d{1,2}/\d{1,2}/\d{4})'
        ]
        
        for pattern in deadline_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def _structure_tasks(self, tasks: List[Dict]) -> List[Dict[str, Any]]:
        """Structure tasks with consistent format"""
        structured = []
        for task in tasks:
            structured.append({
                'title': task.get('title', 'Unnamed Task'),
                'description': task.get('text', task.get('description', '')),
                'owner': task.get('owner', 'Unassigned'),
                'deadline': task.get('deadline', 'Not specified'),
                'status': task.get('status', 'pending'),
                'priority': task.get('priority', 'medium'),
                'confidence': task.get('confidence', 0.5),
                'source': 'meeting'
            })
        return structured