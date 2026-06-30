"""
Synthesizer with Groq Integration - Combines agent outputs into final response
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import json
import os
from loguru import logger
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


@dataclass
class FinalResponse:
    """Final synthesized response"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    agent_outputs: Dict[str, Any]
    metadata: Dict[str, Any]


class Synthesizer:
    """Synthesize agent outputs into final response using Groq"""
    
    def __init__(self, 
                 groq_api_key: Optional[str] = None,
                 groq_model: str = "llama-3.3-70b-versatile"):
        """
        Initialize synthesizer with Groq
        
        Args:
            groq_api_key: Groq API key
            groq_model: Groq model to use
        """
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.groq_model = groq_model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        
        try:
            self.llm = ChatGroq(
                api_key=self.groq_api_key,
                model=self.groq_model,
                temperature=0.3,
                max_tokens=2000,
                timeout=60,
                max_retries=2
            )
            logger.info(f"✅ Synthesizer initialized with Groq: {self.groq_model}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Groq synthesizer: {e}")
            self.llm = None
    
    def synthesize(self, query: str, context: Any, 
                  query_plan: Any) -> FinalResponse:
        """
        Synthesize final response from agent outputs
        
        Args:
            query: Original user query
            context: Fused context (FusedContext object or dict)
            query_plan: Query plan for the request
        
        Returns:
            FinalResponse with synthesized answer
        """
        try:
            # Convert context to dict if it's a FusedContext object
            if hasattr(context, 'to_dict'):
                context_dict = context.to_dict()
            elif isinstance(context, dict):
                context_dict = context
            else:
                # Try to access attributes directly
                context_dict = {
                    'text': getattr(context, 'text', ''),
                    'modalities': getattr(context, 'modalities', []),
                    'metadata': getattr(context, 'metadata', {}),
                    'source_chunks': getattr(context, 'source_chunks', []),
                    'agent_inputs': getattr(context, 'agent_inputs', {})
                }
            
            # Get agent outputs - from the run_agents function result
            # The agent outputs should already be in the context's agent_inputs
            # or we'll get them from the metadata
            agent_outputs = context_dict.get('agent_inputs', {})
            
            # If agent_outputs is empty, try to get from metadata
            if not agent_outputs:
                agent_outputs = context_dict.get('metadata', {}).get('agent_outputs', {})
            
            # Determine primary agent based on intent
            intent = query_plan.intent if hasattr(query_plan, 'intent') else 'general'
            primary_agent = self._select_primary_agent(intent)
            
            # Build response
            if self.llm:
                answer = self._synthesize_with_llm(agent_outputs, query, primary_agent, context_dict)
            else:
                answer = self._synthesize_fallback(agent_outputs, query, primary_agent)
            
            # Extract sources
            sources = self._extract_sources(context_dict)
            
            # Calculate confidence
            confidence = self._calculate_confidence(agent_outputs)
            
            return FinalResponse(
                answer=answer,
                sources=sources,
                confidence=confidence,
                agent_outputs=agent_outputs,
                metadata={
                    'intent': intent,
                    'modalities': context_dict.get('modalities', []),
                    'response_type': 'structured_report',
                    'llm_provider': 'Groq',
                    'model': self.groq_model
                }
            )
            
        except Exception as e:
            logger.error(f"Error in synthesis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return FinalResponse(
                answer="I encountered an error processing your request. Please try again.",
                sources=[],
                confidence=0.0,
                agent_outputs={},
                metadata={'error': str(e)}
            )
    
    def _select_primary_agent(self, intent: str) -> str:
        """Select primary agent based on intent"""
        intent_map = {
            "decision": "decision",
            "risk": "risk",
            "action": "task",
            "summary": "summary",
            "temporal": "summary",
            "general": "summary"
        }
        return intent_map.get(intent, "summary")
    
    def _synthesize_with_llm(self, agent_outputs: Dict, query: str, 
                            primary_agent: str, context_dict: Dict) -> str:
        """Synthesize response using Groq LLM"""
        # Prepare agent outputs for prompt
        formatted_outputs = self._format_agent_outputs(agent_outputs)
        context_text = context_dict.get('text', '')[:1000]
        
        system_prompt = """You are a meeting intelligence synthesizer. Create a comprehensive, well-structured 
        response that answers the user's question based on the agent analysis provided.
        
        Format the response with:
        1. Clear sections with headers
        2. Bullet points for key items
        3. Confidence indicators where appropriate
        4. Actionable insights"""
        
        user_prompt = f"""
User Question: {query}

Primary Analysis Type: {primary_agent}

Context Text: {context_text}

Agent Outputs:
{formatted_outputs}

Please synthesize these outputs into a clear, structured response that directly answers the user's question.
Focus on the most relevant information and provide actionable insights.
"""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"LLM synthesis error: {e}")
            return self._synthesize_fallback(agent_outputs, query, primary_agent)
    
    def _synthesize_fallback(self, agent_outputs: Dict, query: str, primary_agent: str) -> str:
        """Fallback synthesis without LLM"""
        parts = []
        
        # Add query
        parts.append(f"**Question:** {query}\n")
        
        # If no agent outputs, return basic response
        if not agent_outputs:
            return "No agent outputs available to synthesize. Please try uploading some documents first."
        
        # Find primary agent output
        primary_output = None
        for key, value in agent_outputs.items():
            if primary_agent in key.lower():
                primary_output = value
                break
        
        # If primary not found, use first available
        if not primary_output:
            primary_output = next(iter(agent_outputs.values())) if agent_outputs else None
        
        if primary_output:
            # Extract content
            if isinstance(primary_output, dict):
                content = primary_output.get('content', {})
                
                if isinstance(content, dict):
                    for key, value in content.items():
                        if value:
                            if isinstance(value, list) and value:
                                parts.append(f"## {key.capitalize()}")
                                for item in value[:10]:
                                    if isinstance(item, dict):
                                        parts.append(f"- {item.get('title', item.get('text', str(item)))}")
                                    else:
                                        parts.append(f"- {item}")
                            elif isinstance(value, str):
                                parts.append(f"## {key.capitalize()}")
                                parts.append(value[:500])
                elif isinstance(content, list):
                    parts.append(f"## {primary_agent.capitalize()} Results")
                    for item in content[:10]:
                        if isinstance(item, dict):
                            parts.append(f"- {item.get('title', item.get('text', str(item)))}")
                        else:
                            parts.append(f"- {item}")
                elif isinstance(content, str):
                    parts.append(content[:500])
            elif isinstance(primary_output, str):
                parts.append(primary_output[:500])
        else:
            # Show all agent outputs
            for agent_name, output in agent_outputs.items():
                if isinstance(output, dict):
                    content = output.get('content', {})
                    if content:
                        parts.append(f"## {agent_name.capitalize()}")
                        if isinstance(content, dict):
                            for key, value in content.items():
                                if value:
                                    parts.append(f"**{key}:** {str(value)[:200]}")
                        elif isinstance(content, list):
                            for item in content[:5]:
                                if isinstance(item, dict):
                                    parts.append(f"- {item.get('title', item.get('text', str(item)))}")
                                else:
                                    parts.append(f"- {item}")
                        elif isinstance(content, str):
                            parts.append(content[:300])
        
        return '\n\n'.join(parts) if len(parts) > 1 else "No relevant information found."
    
    def _format_agent_outputs(self, agent_outputs: Dict) -> str:
        """Format agent outputs for LLM prompt"""
        if not agent_outputs:
            return "No agent outputs available."
        
        formatted = []
        
        for agent_name, output in agent_outputs.items():
            formatted.append(f"=== {agent_name.upper()} ===\n")
            
            if isinstance(output, dict):
                content = output.get('content')
                confidence = output.get('confidence', 0.0)
                
                formatted.append(f"Confidence: {confidence:.2f}\n")
                
                if isinstance(content, dict):
                    for key, value in content.items():
                        if value:
                            if isinstance(value, list) and len(value) > 0:
                                formatted.append(f"{key}:")
                                for item in value[:5]:
                                    if isinstance(item, dict):
                                        formatted.append(f"  - {item.get('title', item.get('text', str(item)))}")
                                    else:
                                        formatted.append(f"  - {item}")
                            elif isinstance(value, str):
                                formatted.append(f"{key}: {value[:200]}")
                            else:
                                formatted.append(f"{key}: {str(value)[:100]}")
                elif isinstance(content, list):
                    formatted.append(f"Items found: {len(content)}")
                    for item in content[:5]:
                        if isinstance(item, dict):
                            formatted.append(f"- {item.get('title', item.get('text', str(item)))}")
                        else:
                            formatted.append(f"- {item}")
                elif isinstance(content, str):
                    formatted.append(content[:500])
            elif isinstance(output, str):
                formatted.append(output[:500])
            else:
                formatted.append(str(output)[:200])
            
            formatted.append("")
        
        return '\n'.join(formatted) if formatted else "No formatted outputs available."
    
    def _extract_sources(self, context: Dict) -> List[Dict[str, Any]]:
        """Extract source references"""
        sources = []
        retrieved = context.get('source_chunks', [])
        
        for chunk in retrieved[:5]:
            payload = chunk.get('payload', {})
            sources.append({
                'source_file': payload.get('source_file', 'Unknown'),
                'text': payload.get('text', '')[:200],
                'score': chunk.get('score', 0),
                'metadata': {
                    k: v for k, v in payload.items() 
                    if k not in ['text', 'content', 'vector']
                }
            })
        
        return sources
    
    def _calculate_confidence(self, agent_outputs: Dict) -> float:
        """Calculate overall confidence from agent outputs"""
        if not agent_outputs:
            return 0.0
        
        confidences = []
        for output in agent_outputs.values():
            if isinstance(output, dict):
                # Check for confidence in output
                if 'confidence' in output:
                    confidences.append(output['confidence'])
                # Check content for confidence
                content = output.get('content', {})
                if isinstance(content, dict) and 'confidence' in content:
                    confidences.append(content['confidence'])
            elif isinstance(output, (int, float)):
                confidences.append(float(output))
        
        if not confidences:
            return 0.5
        
        # Return average confidence
        return sum(confidences) / len(confidences)