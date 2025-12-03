"""Basic state machine implementation for the Governor system.

This is a minimal implementation to support the webhook interface.
A full LangGraph implementation should be created as part of Step 2.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from ...core.domain.state import GovernorState, StateNode
from ...core.domain.events import GovernorEvent, GovernorResponse, ResponseType, ChannelType
from ...core.domain.tools import ToolCall

logger = logging.getLogger(__name__)


class GovernorGraph:
    """Basic state machine for the Governor system.
    
    This is a minimal implementation that provides the core interface
    needed by the webhook system. A full LangGraph implementation
    should replace this in the future.
    """
    
    def __init__(self):
        """Initialize the state machine."""
        self.name = "BasicGovernorGraph"
        logger.info("Initialized basic Governor state machine")
    
    async def invoke(self, graph_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through the state machine.
        
        Args:
            graph_input: Dictionary containing:
                - state: GovernorState
                - event: GovernorEvent
                - context_prompt: str
                - available_tools: Dict[str, Any]
        
        Returns:
            Dictionary containing the response
        """
        try:
            state = graph_input.get("state")
            event = graph_input.get("event")
            context_prompt = graph_input.get("context_prompt")
            available_tools = graph_input.get("available_tools", {})
            
            if not state or not event:
                raise ValueError("State and event are required")
            
            # Update state
            state.current_node = StateNode.ANALYZE
            
            # Create a basic response based on the input
            response = self._generate_basic_response(state, event, available_tools)
            
            # Update conversation history
            if hasattr(state, 'conversation_history'):
                state.conversation_history.append({
                    "role": "assistant",
                    "content": response.content,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            logger.info(f"Processed event for {event.user_id}: {event.content[:50]}...")
            
            return {"response": response}
            
        except Exception as e:
            logger.error(f"Error in state machine: {e}")
            
            # Return error response
            error_response = GovernorResponse(
                user_id=event.user_id if event else "unknown",
                session_id=event.session_id if event else "unknown",
                content=f"I encountered an error processing your request: {str(e)}",
                response_type=ResponseType.ERROR,
                channel=event.channel if event else ChannelType.API,
                metadata={"error": str(e)}
            )
            
            return {"response": error_response}
    
    def _generate_basic_response(
        self, 
        state: GovernorState, 
        event: GovernorEvent, 
        available_tools: Dict[str, Any]
    ) -> GovernorResponse:
        """Generate a basic response based on the input.
        
        Args:
            state: Current state
            event: Input event
            available_tools: Available tools
        
        Returns:
            Basic response
        """
        content = event.content.lower()
        
        # Basic response logic
        if "weather" in content:
            response_text = "I can help you check the weather, but I need access to weather APIs to provide real data. This is a basic response from the minimal state machine."
        elif "email" in content:
            response_text = "I can help you send emails, but this would require confirmation since it's a potentially risky operation. This is a basic response from the minimal state machine."
        elif "hello" in content or "hi" in content:
            response_text = f"Hello! I'm the Headless Governor. I have access to {len(available_tools)} tools and am ready to help you. This is a basic response from the minimal state machine."
        elif "help" in content:
            tool_list = ", ".join(available_tools) if available_tools else "no tools"
            response_text = f"I'm here to help! I currently have access to: {tool_list}. This is a basic response from the minimal state machine."
        else:
            response_text = f"I received your message: '{event.content}'. I'm a basic state machine implementation that will be replaced with a full LangGraph system. I have access to {len(available_tools)} tools."
        
        return GovernorResponse(
            user_id=event.user_id,
            session_id=event.session_id,
            content=response_text,
            response_type=ResponseType.TEXT,
            channel=event.channel,
            metadata={
                "state_machine": "basic",
                "available_tools": list(available_tools),
                "processed_at": datetime.utcnow().isoformat()
            }
        )