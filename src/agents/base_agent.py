"""
Base Agent with Groq LLM Integration
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import json
import re

from loguru import logger
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


@dataclass
class AgentOutput:
    """Standardized output from an agent."""
    type: str  # summary, decision, risk, task, weak_signal
    content: Any
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    reasoning: Optional[str] = None


class BaseAgent(ABC):
    """Abstract base class for all agents with Groq support."""

    def __init__(
        self,
        name: str,
        groq_api_key: Optional[str] = None,
        groq_model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ):
        """
        Initialize agent with Groq LLM.

        Args:
            name: Agent name.
            groq_api_key: Groq API key.
            groq_model: Groq model to use.
            temperature: Temperature for generation.
            max_tokens: Maximum tokens to generate.
        """
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize Groq LLM
        try:
            self.llm = ChatGroq(
                api_key=groq_api_key,
                model=groq_model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=60,
                max_retries=2,
            )
            logger.info(f"✅ {name} initialized with Groq model: {groq_model}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Groq for {name}: {e}")
            self.llm = None

    @abstractmethod
    def process(self, context: Dict[str, Any]) -> AgentOutput:
        """
        Process the given context and produce output.

        Args:
            context: Dictionary containing relevant information.

        Returns:
            AgentOutput containing the results.
        """
        pass

    def _prepare_messages(
        self, system_prompt: str, user_prompt: str
    ) -> List[HumanMessage | SystemMessage]:
        """Prepare messages for the Groq LLM."""
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

    def _get_llm_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        Get a response from the Groq LLM.

        Args:
            system_prompt: System instructions.
            user_prompt: User query.

        Returns:
            Response text from the LLM.
        """
        if self.llm is None:
            logger.warning(f"{self.name}: No LLM available")
            return ""

        try:
            messages = self._prepare_messages(system_prompt, user_prompt)
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"{self.name} LLM error: {e}")
            return ""

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse a JSON response from the LLM.
        """
        try:
            # Try extracting JSON from markdown or surrounding text
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

            return json.loads(response)

        except Exception as e:
            logger.warning(f"{self.name}: Failed to parse JSON response: {e}")
            return None