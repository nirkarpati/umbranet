"""Analyze node implementation.

The analyze node processes user input using LLM capabilities to understand
intent, extract requirements, and determine if tool usage is needed.
"""

import json
import logging
import re
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ...config import settings
from ...domain.state import GovernorState, StateNode
from ..base import NodeFunction

logger = logging.getLogger(__name__)


class AnalyzeNode(NodeFunction):
    """Node that analyzes user input for intent and tool requirements.

    This node uses LLM analysis to understand what the user wants
    and determines the appropriate response strategy.
    """

    @property
    def name(self) -> str:
        """Get the node name."""
        return StateNode.ANALYZE.value

    async def __call__(self, state: GovernorState) -> GovernorState:
        """Analyze user input and determine response strategy.

        Args:
            state: Current workflow state

        Returns:
            Updated state with analysis results
        """
        user_input = state.last_user_input
        if not user_input:
            state.record_error("No user input to analyze")
            state.transition_to(StateNode.RESPOND)
            return state

        # Perform LLM-based intent analysis
        try:
            analysis_result = await self._analyze_with_llm(user_input, state)
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            state.record_error(f"Analysis failed: {str(e)}")
            state.transition_to(StateNode.RESPOND)
            return state

        # Store analysis results in context
        state.context.update(
            {
                "analysis": analysis_result,
                "analysis_completed_at": state.updated_at.isoformat(),
            }
        )

        # Determine next state based on analysis
        if analysis_result.get("needs_retrieval", False):
            state.transition_to(StateNode.TOOL_DECISION)
        else:
            state.transition_to(StateNode.RESPOND)

        return state

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.RequestError),
        reraise=True,
    )
    async def _call_openai(self, prompt: str, max_tokens: int = 300) -> str:
        """Make API call to OpenAI with retry logic."""
        if not settings.openai_api_key:
            raise Exception("OpenAI API key not configured")

        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": "gpt-4o-mini",  # Efficient model for analysis
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an AI intent classifier. Analyze user input and return "
                        "structured JSON only. Be precise and consistent."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,  # Low temperature for consistency
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30.0,
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
                raise Exception(f"OpenAI API error: {e.response.status_code}") from e

        except (httpx.RequestError, json.JSONDecodeError) as e:
            logger.error(f"Request failed: {str(e)}")
            raise Exception(f"Request failed: {str(e)}") from e

    def _get_recent_conversation_summary(self, state: GovernorState) -> str:
        """Get a summary of recent conversation for context."""
        if not hasattr(state, "conversation_history") or not state.conversation_history:
            return "No prior conversation context available."

        # Get last few exchanges
        recent_messages = state.conversation_history[-3:]
        summary_parts = []

        for i, turn in enumerate(recent_messages, 1):
            if hasattr(turn, "user_message") and hasattr(turn, "assistant_response"):
                summary_parts.append(
                    f"Exchange {i}: User said '{turn.user_message}', Assistant replied '{turn.assistant_response}'"
                )
            elif hasattr(turn, "content"):
                summary_parts.append(f"Message {i}: {turn.content}")

        return (
            " | ".join(summary_parts)
            if summary_parts
            else "No clear conversation context."
        )

    async def _analyze_with_llm(
        self, user_input: str, state: GovernorState
    ) -> dict[str, Any]:
        """Analyze user input using LLM for intent classification and tool selection.

        Args:
            user_input: The user's message
            state: Current workflow state for context

        Returns:
            Dictionary containing structured analysis results
        """
        # Get recent conversation context
        recent_context = self._get_recent_conversation_summary(state)

        # Construct analysis prompt
        analysis_prompt = f"""Analyze the following user input for intent and determine if memory retrieval tools are needed.

**Current User Input:** "{user_input}"

**Recent Conversation Context:** {recent_context}

**Session Context:**
- Total conversation turns: {state.total_turns}
- Session age: {(state.updated_at - state.created_at).total_seconds() / 60:.1f} minutes
- Error count: {state.error_count}

**Analysis Task:**
Determine if the user is asking about information that might not be present in the recent conversation context above. Consider if they're referring to:
- Past events, conversations, or decisions
- People, places, or facts mentioned previously but not in recent context
- Historical information or stored knowledge
- Personal details or preferences from earlier sessions

**Required Output Format (JSON only):**
{{
    "intent": "retrieval_needed" | "action_request" | "chat",
    "needs_retrieval": true | false,
    "suggested_retrieval_tool": "search_episodic_memory" | "query_knowledge_graph" | null,
    "reasoning": "Brief explanation of why retrieval is/isn't needed",
    "confidence": 0.0-1.0,
    "urgency": "low" | "normal" | "high"
}}

**Guidelines:**
- Use "search_episodic_memory" for questions about past conversations, events, or temporal information
- Use "query_knowledge_graph" for questions about entities, relationships, facts, or structured knowledge
- Set needs_retrieval=true if user references information not clearly present in recent context
- Set intent="retrieval_needed" when needs_retrieval=true
- Set intent="action_request" for requests to do something specific
- Set intent="chat" for general conversation that doesn't require retrieval
"""

        # Call LLM for analysis
        response = await self._call_openai(analysis_prompt)

        # Parse JSON response
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                # Fallback to basic analysis
                logger.warning(f"Could not parse LLM response as JSON: {response}")
                data = {
                    "intent": "chat",
                    "needs_retrieval": False,
                    "suggested_retrieval_tool": None,
                    "reasoning": "Failed to parse LLM response, defaulting to chat",
                    "confidence": 0.5,
                    "urgency": "normal",
                }

        # Validate and ensure required fields
        validated_result = {
            "intent": data.get("intent", "chat"),
            "needs_retrieval": data.get("needs_retrieval", False),
            "suggested_retrieval_tool": data.get("suggested_retrieval_tool"),
            "reasoning": data.get("reasoning", "No reasoning provided"),
            "confidence": float(data.get("confidence", 0.7)),
            "urgency": data.get("urgency", "normal"),
            "analysis_method": "llm",
            "session_context": {
                "total_turns": state.total_turns,
                "session_duration_minutes": (
                    state.updated_at - state.created_at
                ).total_seconds()
                / 60,
                "error_count": state.error_count,
            },
        }

        return validated_result


class AnalyzeConditional:
    """Conditional routing from analyze state."""

    @property
    def name(self) -> str:
        """Get the conditional name."""
        return "analyze_routing"

    def __call__(self, state: GovernorState) -> str:
        """Determine next node from analyze state.

        Args:
            state: Current workflow state

        Returns:
            Next node name
        """
        analysis = state.context.get("analysis", {})

        # If analysis indicates retrieval is needed, go to tool decision
        if analysis.get("needs_retrieval", False):
            return StateNode.TOOL_DECISION.value

        # If there's an error during analysis, go to respond with error
        if state.error_count > 0:
            return StateNode.RESPOND.value

        # Otherwise, go straight to response generation
        return StateNode.RESPOND.value
