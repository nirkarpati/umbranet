"""Conversation summarization service for rolling summary functionality."""

import json
import logging

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ...core.config import settings
from ...core.domain.memory import ConversationTurn, SummarizationRequest
from ...core.utils.tokens import token_counter

logger = logging.getLogger(__name__)


class SummarizationError(Exception):
    """Exception raised during summarization process."""
    pass


class ConversationSummarizer:
    """Service for summarizing conversation history using LLM."""
    
    def __init__(self) -> None:
        """Initialize the summarizer."""
        self.client = httpx.AsyncClient()
        self.model = "gpt-4o-mini"  # Cheaper model for summarization
    
    async def __aenter__(self) -> "ConversationSummarizer":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.client.aclose()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.RequestError),
        reraise=True
    )
    async def _call_openai(self, prompt: str, max_tokens: int = 500) -> str:
        """Make API call to OpenAI with retry logic."""
        if not settings.openai_api_key:
            raise SummarizationError("OpenAI API key not configured")
        
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system", 
                    "content": (
                        "You are a precise conversation summarizer. Create concise "
                        "summaries that preserve key facts, decisions, and context."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1  # Low temperature for consistent summaries
        }
        
        try:
            response = await self.client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30.0
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("OpenAI API rate limit hit, retrying...")
                raise  # Will be retried by tenacity
            else:
                logger.error(
                    f"OpenAI API error {e.response.status_code}: {e.response.text}"
                )
                raise SummarizationError(
                    f"API error: {e.response.status_code}"
                ) from e
        
        except (httpx.RequestError, json.JSONDecodeError) as e:
            logger.error(f"Request failed: {str(e)}")
            raise SummarizationError(f"Request failed: {str(e)}") from e
    
    def _format_turns_for_summary(self, turns: list[ConversationTurn]) -> str:
        """Format conversation turns for summarization prompt."""
        formatted_turns = []
        for i, turn in enumerate(turns, 1):
            formatted_turns.append(f"Turn {i}:")
            formatted_turns.append(f"User: {turn.user_message}")
            formatted_turns.append(f"Assistant: {turn.assistant_response}")
            formatted_turns.append("")  # Empty line between turns
        
        return "\n".join(formatted_turns)
    
    async def summarize_conversation(
        self, 
        request: SummarizationRequest
    ) -> str:
        """Summarize conversation history.
        
        Args:
            request: Summarization request with existing summary and new messages
            
        Returns:
            Updated summary incorporating new messages
        """
        if not request.messages_to_summarize:
            return request.existing_summary or ""
        
        # Format the conversation turns
        new_messages_text = self._format_turns_for_summary(request.messages_to_summarize)
        
        if request.existing_summary:
            # Merge with existing summary
            prompt = f"""Please update the following conversation summary by incorporating the new messages below.

EXISTING SUMMARY:
{request.existing_summary}

NEW CONVERSATION TURNS:
{new_messages_text}

Create an updated summary that:
1. Preserves important facts and context from the existing summary
2. Incorporates key information from the new conversation turns
3. Maintains a concise, factual tone
4. Stays under {request.max_summary_tokens} tokens
5. Focuses on actionable information and user preferences

Updated Summary:"""
        else:
            # Create initial summary
            prompt = f"""Please create a concise summary of the following conversation that captures:

CONVERSATION TURNS:
{new_messages_text}

The summary should:
1. Highlight key facts, decisions, and user preferences
2. Maintain context for future conversations
3. Stay under {request.max_summary_tokens} tokens
4. Use a factual, third-person tone
5. Focus on actionable information

Summary:"""
        
        try:
            summary = await self._call_openai(prompt, request.max_summary_tokens)
            
            # Verify token count
            actual_tokens = token_counter.count_tokens(summary)
            if actual_tokens > request.max_summary_tokens * 1.1:  # 10% buffer
                logger.warning(
                    f"Summary exceeded token limit: {actual_tokens} > {request.max_summary_tokens}"
                )
            
            return summary
        
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            # Fallback: return existing summary or create a simple one
            if request.existing_summary:
                return request.existing_summary
            else:
                # Create a basic fallback summary
                recent_user_messages = [turn.user_message for turn in request.messages_to_summarize[-3:]]
                return f"Recent conversation topics: {', '.join(recent_user_messages[:2])}"


# Global summarizer instance
async def get_summarizer() -> ConversationSummarizer:
    """Get a summarizer instance (async context manager pattern)."""
    return ConversationSummarizer()