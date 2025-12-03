"""Await confirmation node implementation.

The await confirmation node handles situations where user confirmation
is required before proceeding with tool execution or other actions.
"""

from datetime import datetime, timedelta
from typing import Any

from ...domain.state import GovernorState, StateNode
from ..base import NodeFunction


class AwaitConfirmationNode(NodeFunction):
    """Node that handles confirmation waiting states.
    
    This node manages the flow when user confirmation is required
    for high-risk operations or tool executions.
    """
    
    @property
    def name(self) -> str:
        """Get the node name."""
        return StateNode.AWAIT_CONFIRMATION.value
    
    async def __call__(self, state: GovernorState) -> GovernorState:
        """Handle confirmation waiting logic.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state based on confirmation status
        """
        if not state.awaiting_confirmation:
            # No confirmation needed, proceed to execution or response
            if state.pending_tools:
                state.transition_to(StateNode.EXECUTE)
            else:
                state.transition_to(StateNode.RESPOND)
            return state
        
        # Check for confirmation timeout
        confirmation_timeout = self._check_confirmation_timeout(state)
        if confirmation_timeout:
            self._handle_confirmation_timeout(state)
            state.transition_to(StateNode.RESPOND)
            return state
        
        # Parse user input for confirmation response
        if state.last_user_input:
            confirmation_response = self._parse_confirmation_response(state.last_user_input)
            
            if confirmation_response == "approve":
                self._handle_confirmation_approval(state)
                state.transition_to(StateNode.EXECUTE)
            elif confirmation_response == "deny":
                self._handle_confirmation_denial(state)
                state.transition_to(StateNode.RESPOND)
            elif confirmation_response == "modify":
                # User wants to modify the request
                self._handle_confirmation_modification(state)
                state.transition_to(StateNode.ANALYZE)  # Re-analyze with modifications
            else:
                # Unclear response, ask for clarification
                self._handle_unclear_confirmation(state)
                # Stay in await_confirmation state
        
        # Update confirmation context
        state.context.update({
            "confirmation_status": {
                "waiting_since": state.confirmation_context.get("requested_at") if state.confirmation_context else None,
                "timeout_at": self._calculate_confirmation_timeout(state),
                "last_user_input": state.last_user_input,
                "clarification_needed": self._needs_clarification(state)
            }
        })
        
        return state
    
    def _check_confirmation_timeout(self, state: GovernorState) -> bool:
        """Check if confirmation has timed out.
        
        Args:
            state: Current workflow state
            
        Returns:
            True if confirmation has timed out
        """
        if not state.confirmation_context:
            return False
        
        requested_at = state.confirmation_context.get("requested_at")
        if not requested_at:
            return False
        
        timeout_minutes = state.confirmation_context.get("timeout_minutes", 10)
        timeout_at = datetime.fromisoformat(requested_at) + timedelta(minutes=timeout_minutes)
        
        return datetime.utcnow() > timeout_at
    
    def _calculate_confirmation_timeout(self, state: GovernorState) -> str:
        """Calculate when confirmation will timeout.
        
        Args:
            state: Current workflow state
            
        Returns:
            ISO formatted timeout timestamp
        """
        if not state.confirmation_context:
            return (datetime.utcnow() + timedelta(minutes=10)).isoformat()
        
        requested_at = state.confirmation_context.get("requested_at")
        if not requested_at:
            return (datetime.utcnow() + timedelta(minutes=10)).isoformat()
        
        timeout_minutes = state.confirmation_context.get("timeout_minutes", 10)
        timeout_at = datetime.fromisoformat(requested_at) + timedelta(minutes=timeout_minutes)
        
        return timeout_at.isoformat()
    
    def _parse_confirmation_response(self, user_input: str) -> str:
        """Parse user input to determine confirmation response.
        
        Args:
            user_input: User's response to confirmation request
            
        Returns:
            Parsed response: "approve", "deny", "modify", or "unclear"
        """
        input_lower = user_input.lower().strip()
        
        # Approval patterns
        approval_patterns = [
            "yes", "y", "ok", "okay", "approve", "approved", "confirm", "confirmed",
            "go ahead", "proceed", "do it", "sure", "absolutely", "definitely",
            "i approve", "i confirm", "please do", "go for it"
        ]
        
        # Denial patterns
        denial_patterns = [
            "no", "n", "deny", "denied", "cancel", "cancelled", "stop", "abort",
            "don't", "do not", "refuse", "declined", "reject", "rejected",
            "i deny", "i decline", "please don't", "no way"
        ]
        
        # Modification patterns
        modification_patterns = [
            "change", "modify", "edit", "update", "different", "instead",
            "but", "however", "actually", "wait", "hold on", "let me think",
            "can you change", "make it", "try this instead"
        ]
        
        # Check for exact matches first
        if input_lower in approval_patterns:
            return "approve"
        elif input_lower in denial_patterns:
            return "deny"
        elif any(pattern in input_lower for pattern in modification_patterns):
            return "modify"
        
        # Check for partial matches
        if any(pattern in input_lower for pattern in approval_patterns):
            return "approve"
        elif any(pattern in input_lower for pattern in denial_patterns):
            return "deny"
        
        return "unclear"
    
    def _handle_confirmation_approval(self, state: GovernorState) -> None:
        """Handle user approval of confirmation request.
        
        Args:
            state: Current workflow state
        """
        if not state.confirmation_context:
            return
        
        # Mark tools as confirmed if this is tool confirmation
        confirmation_type = state.confirmation_context.get("confirmation_type")
        
        if confirmation_type == "tool_execution":
            tools_awaiting = state.confirmation_context.get("tools_awaiting_confirmation", [])
            confirmed_tool_ids = [tool["execution_id"] for tool in tools_awaiting]
            
            # Update confirmation context with approved tools
            updated_context = state.confirmation_context.copy()
            updated_context["confirmed_tools"] = confirmed_tool_ids
            updated_context["approval_timestamp"] = datetime.utcnow().isoformat()
            state.confirmation_context = updated_context
            
            # Mark tools as no longer requiring confirmation
            for tool in state.pending_tools:
                if tool.execution_id in confirmed_tool_ids:
                    tool.requires_confirmation = False
        
        # Clear awaiting confirmation status
        state.awaiting_confirmation = False
        
        # Log approval
        state.context.setdefault("confirmation_log", []).append({
            "action": "approved",
            "timestamp": datetime.utcnow().isoformat(),
            "user_input": state.last_user_input,
            "confirmation_type": confirmation_type
        })
    
    def _handle_confirmation_denial(self, state: GovernorState) -> None:
        """Handle user denial of confirmation request.
        
        Args:
            state: Current workflow state
        """
        if not state.confirmation_context:
            return
        
        confirmation_type = state.confirmation_context.get("confirmation_type")
        
        if confirmation_type == "tool_execution":
            # Remove denied tools from pending
            tools_awaiting = state.confirmation_context.get("tools_awaiting_confirmation", [])
            denied_tool_ids = [tool["execution_id"] for tool in tools_awaiting]
            
            # Remove denied tools
            state.pending_tools = [
                tool for tool in state.pending_tools 
                if tool.execution_id not in denied_tool_ids
            ]
            
            # Mark tools as cancelled
            for tool in list(state.pending_tools):
                if tool.execution_id in denied_tool_ids:
                    tool.cancel_execution()
        
        # Clear confirmation state
        state.resolve_confirmation(approved=False)
        
        # Log denial
        state.context.setdefault("confirmation_log", []).append({
            "action": "denied",
            "timestamp": datetime.utcnow().isoformat(),
            "user_input": state.last_user_input,
            "confirmation_type": confirmation_type
        })
        
        # Set context for response generation
        state.context["response_reason"] = "user_denied_confirmation"
    
    def _handle_confirmation_modification(self, state: GovernorState) -> None:
        """Handle user request to modify the confirmation request.
        
        Args:
            state: Current workflow state
        """
        # Clear current confirmation state
        state.resolve_confirmation(approved=False)
        
        # Cancel current tools since user wants modifications
        for tool in state.pending_tools:
            tool.cancel_execution()
        state.pending_tools.clear()
        
        # Set context for re-analysis
        state.context["reanalysis_reason"] = "user_requested_modification"
        state.context["modification_request"] = state.last_user_input
        
        # Log modification request
        state.context.setdefault("confirmation_log", []).append({
            "action": "modification_requested",
            "timestamp": datetime.utcnow().isoformat(),
            "user_input": state.last_user_input,
            "modification_details": state.last_user_input
        })
    
    def _handle_confirmation_timeout(self, state: GovernorState) -> None:
        """Handle confirmation timeout.
        
        Args:
            state: Current workflow state
        """
        # Cancel pending operations due to timeout
        for tool in state.pending_tools:
            if tool.requires_confirmation:
                tool.cancel_execution()
        
        # Remove tools that required confirmation
        state.pending_tools = [
            tool for tool in state.pending_tools 
            if not tool.requires_confirmation
        ]
        
        # Clear confirmation state
        state.resolve_confirmation(approved=False)
        
        # Set context for timeout response
        state.context["response_reason"] = "confirmation_timeout"
        
        # Log timeout
        state.context.setdefault("confirmation_log", []).append({
            "action": "timeout",
            "timestamp": datetime.utcnow().isoformat(),
            "timeout_duration_minutes": state.confirmation_context.get("timeout_minutes", 10) if state.confirmation_context else 10
        })
    
    def _handle_unclear_confirmation(self, state: GovernorState) -> None:
        """Handle unclear confirmation response from user.
        
        Args:
            state: Current workflow state
        """
        # Track unclear responses
        state.context.setdefault("unclear_responses", []).append({
            "user_input": state.last_user_input,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Set context for clarification response
        state.context["needs_clarification"] = True
        state.context["clarification_attempt"] = len(state.context.get("unclear_responses", []))
        
        # If too many unclear responses, timeout
        max_unclear_attempts = 3
        if len(state.context.get("unclear_responses", [])) >= max_unclear_attempts:
            self._handle_confirmation_timeout(state)
            state.transition_to(StateNode.RESPOND)
    
    def _needs_clarification(self, state: GovernorState) -> bool:
        """Check if clarification is needed for confirmation.
        
        Args:
            state: Current workflow state
            
        Returns:
            True if clarification is needed
        """
        return state.context.get("needs_clarification", False)


class AwaitConfirmationConditional:
    """Conditional routing from await confirmation state."""
    
    @property
    def name(self) -> str:
        """Get the conditional name."""
        return "await_confirmation_routing"
    
    def __call__(self, state: GovernorState) -> str:
        """Determine next node from await confirmation state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node name
        """
        # If no longer awaiting confirmation, check what to do next
        if not state.awaiting_confirmation:
            # Check if user requested modifications
            if state.context.get("reanalysis_reason") == "user_requested_modification":
                return StateNode.ANALYZE.value
            
            # Check if we have tools to execute
            if state.pending_tools and not any(tool.requires_confirmation for tool in state.pending_tools):
                return StateNode.EXECUTE.value
            
            # Otherwise go to respond
            return StateNode.RESPOND.value
        
        # If there's an error or timeout, go to respond
        if (state.context.get("response_reason") in ["confirmation_timeout"] or 
            state.error_count > 0):
            return StateNode.RESPOND.value
        
        # If still awaiting and needs clarification, stay in this state
        # The response will be generated but we'll come back here
        if state.context.get("needs_clarification"):
            return StateNode.RESPOND.value
        
        # Continue waiting for confirmation
        return StateNode.AWAIT_CONFIRMATION.value