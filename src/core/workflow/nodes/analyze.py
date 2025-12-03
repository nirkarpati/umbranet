"""Analyze node implementation.

The analyze node processes user input using LLM capabilities to understand
intent, extract requirements, and determine if tool usage is needed.
"""

import json
import re
from typing import Any

from ...domain.state import GovernorState, StateNode
from ..base import NodeFunction


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
        
        # Perform intent analysis
        analysis_result = await self._analyze_user_intent(user_input, state)
        
        # Store analysis results in context
        state.context.update({
            "analysis": analysis_result,
            "analysis_completed_at": state.updated_at.isoformat()
        })
        
        # Determine next state based on analysis
        if analysis_result.get("needs_tools", False):
            state.transition_to(StateNode.TOOL_DECISION)
        else:
            state.transition_to(StateNode.RESPOND)
        
        return state
    
    async def _analyze_user_intent(
        self, 
        user_input: str, 
        state: GovernorState
    ) -> dict[str, Any]:
        """Analyze user input to understand intent.
        
        Args:
            user_input: The user's message
            state: Current workflow state for context
            
        Returns:
            Dictionary containing analysis results
        """
        # For now, implement rule-based analysis
        # In production, this would use LLM capabilities
        
        # Basic intent detection patterns
        question_patterns = [
            r'\b(what|how|where|when|why|who)\b',
            r'\?',
            r'\b(explain|tell me|show me)\b'
        ]
        
        action_patterns = [
            r'\b(send|email|message|call|remind|schedule)\b',
            r'\b(create|make|generate|build)\b',
            r'\b(search|find|look up|get)\b',
            r'\b(calculate|compute|analyze)\b',
            r'\b(save|store|remember)\b',
            r'\b(delete|remove|cancel)\b'
        ]
        
        tool_requiring_patterns = [
            r'\b(weather|temperature|forecast)\b',
            r'\b(email|send message|contact)\b',
            r'\b(calendar|schedule|appointment|meeting)\b',
            r'\b(search|google|lookup)\b',
            r'\b(file|document|save|open)\b',
            r'\b(calculate|math|compute)\b'
        ]
        
        input_lower = user_input.lower()
        
        # Detect if it's a question
        is_question = any(re.search(pattern, input_lower) for pattern in question_patterns)
        
        # Detect if it's an action request
        is_action = any(re.search(pattern, input_lower) for pattern in action_patterns)
        
        # Detect if tools are likely needed
        needs_tools = any(re.search(pattern, input_lower) for pattern in tool_requiring_patterns)
        
        # Extract potential entities
        entities = self._extract_entities(user_input)
        
        # Determine urgency
        urgency_indicators = ["urgent", "asap", "immediately", "now", "emergency"]
        urgency = "high" if any(indicator in input_lower for indicator in urgency_indicators) else "normal"
        
        # Analyze sentiment
        positive_words = ["please", "thanks", "good", "great", "excellent", "perfect"]
        negative_words = ["problem", "issue", "error", "wrong", "bad", "terrible"]
        
        positive_count = sum(1 for word in positive_words if word in input_lower)
        negative_count = sum(1 for word in negative_words if word in input_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "intent": self._classify_intent(is_question, is_action, needs_tools),
            "is_question": is_question,
            "is_action": is_action,
            "needs_tools": needs_tools,
            "entities": entities,
            "urgency": urgency,
            "sentiment": sentiment,
            "confidence": self._calculate_confidence(is_question, is_action, needs_tools),
            "suggested_tools": self._suggest_tools(input_lower),
            "context_variables": {
                "user_history_turns": state.total_turns,
                "session_duration": (state.updated_at - state.created_at).total_seconds(),
                "error_count": state.error_count
            }
        }
    
    def _classify_intent(self, is_question: bool, is_action: bool, needs_tools: bool) -> str:
        """Classify the user's intent based on analysis."""
        if is_question and needs_tools:
            return "information_request"
        elif is_action and needs_tools:
            return "action_request" 
        elif is_question:
            return "simple_question"
        elif is_action:
            return "simple_action"
        else:
            return "conversation"
    
    def _extract_entities(self, text: str) -> dict[str, list[str]]:
        """Extract entities from user input."""
        entities = {}
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            entities["emails"] = emails
        
        # Phone numbers (simple pattern)
        phone_pattern = r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b'
        phones = re.findall(phone_pattern, text)
        if phones:
            entities["phones"] = phones
        
        # URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        if urls:
            entities["urls"] = urls
        
        # Dates (simple patterns)
        date_patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b(today|tomorrow|yesterday)\b'
        ]
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, text, re.IGNORECASE))
        if dates:
            entities["dates"] = dates
        
        # Times
        time_pattern = r'\b\d{1,2}:\d{2}\s*(am|pm|AM|PM)?\b'
        times = re.findall(time_pattern, text)
        if times:
            entities["times"] = [f"{t[0]} {t[1]}".strip() for t in times]
        
        return entities
    
    def _calculate_confidence(self, is_question: bool, is_action: bool, needs_tools: bool) -> float:
        """Calculate confidence score for the analysis."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence for clear patterns
        if is_question:
            confidence += 0.2
        if is_action:
            confidence += 0.2
        if needs_tools:
            confidence += 0.1
        
        # Cap at 1.0
        return min(confidence, 1.0)
    
    def _suggest_tools(self, input_lower: str) -> list[str]:
        """Suggest tools that might be useful for this input."""
        suggested = []
        
        tool_mappings = {
            "weather": ["weather", "temperature", "forecast", "rain", "snow"],
            "email": ["email", "send message", "contact", "mail"],
            "calendar": ["calendar", "schedule", "meeting", "appointment", "remind"],
            "search": ["search", "find", "lookup", "google"],
            "file_manager": ["file", "document", "save", "open", "folder"],
            "calculator": ["calculate", "math", "compute", "sum", "multiply"],
            "web_browser": ["website", "browse", "url", "link"],
            "timer": ["timer", "alarm", "countdown", "stopwatch"]
        }
        
        for tool, keywords in tool_mappings.items():
            if any(keyword in input_lower for keyword in keywords):
                suggested.append(tool)
        
        return suggested


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
        
        # If analysis indicates tools are needed, go to tool decision
        if analysis.get("needs_tools", False):
            return StateNode.TOOL_DECISION.value
        
        # If there's an error during analysis, go to respond with error
        if state.error_count > 0:
            return StateNode.RESPOND.value
        
        # Otherwise, go straight to response generation
        return StateNode.RESPOND.value