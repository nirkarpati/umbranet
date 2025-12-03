"""Idle node implementation.

The idle node is the starting/resting state of the Governor workflow.
It processes incoming user messages and determines the next action.
"""

from typing import Any

from ...domain.events import GovernorEvent
from ...domain.state import GovernorState, StateNode
from ..base import NodeFunction


class IdleNode(NodeFunction):
    """Node that handles the idle state of the Governor workflow.
    
    In the idle state, the system waits for user input and performs
    initial processing to determine the next workflow step.
    """
    
    @property
    def name(self) -> str:
        """Get the node name."""
        return StateNode.IDLE.value
    
    async def __call__(self, state: GovernorState) -> GovernorState:
        """Process the idle state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state ready for next node
        """
        # Check if we have a new user input to process
        if not state.last_user_input:
            # Nothing to process, stay idle
            return state
        
        # Update state to indicate we're moving to analysis
        state.transition_to(StateNode.ANALYZE)
        
        # Add context about the input type for analysis
        input_context = {
            "message_length": len(state.last_user_input),
            "contains_question": "?" in state.last_user_input,
            "contains_command": any(
                state.last_user_input.lower().startswith(cmd) 
                for cmd in ["please", "can you", "help me", "do", "make", "create"]
            ),
            "is_greeting": any(
                greeting in state.last_user_input.lower()
                for greeting in ["hello", "hi", "hey", "good morning", "good afternoon"]
            )
        }
        
        state.context.update({
            "idle_analysis": input_context,
            "processing_started_at": state.updated_at.isoformat()
        })
        
        return state


class IdleConditional:
    """Conditional routing from idle state."""
    
    @property
    def name(self) -> str:
        """Get the conditional name."""
        return "idle_routing"
    
    def __call__(self, state: GovernorState) -> str:
        """Determine next node from idle state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node name
        """
        # If no user input, stay idle
        if not state.last_user_input:
            return StateNode.IDLE.value
            
        # If awaiting confirmation, go to confirmation handler
        if state.awaiting_confirmation:
            return StateNode.AWAIT_CONFIRMATION.value
            
        # If there's an error to handle, go to respond
        if state.last_error and state.error_count > 0:
            return StateNode.RESPOND.value
            
        # Normal flow: move to analysis
        return StateNode.ANALYZE.value