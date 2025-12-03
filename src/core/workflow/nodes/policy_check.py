"""Policy check node implementation.

The policy check node evaluates tool calls against security policies
and determines whether they should be executed, denied, or require confirmation.
"""

from datetime import datetime
from typing import Any

from ...domain.state import GovernorState, StateNode
from ...domain.tools import DecisionType, PolicyDecision, RiskLevel, ToolCall
from ..base import NodeFunction


class PolicyCheckNode(NodeFunction):
    """Node that evaluates tool calls against security policies.
    
    This node implements the security layer of the Governor system,
    ensuring that tool executions comply with security policies and
    user permissions.
    """
    
    @property
    def name(self) -> str:
        """Get the node name."""
        return StateNode.POLICY_CHECK.value
    
    async def __call__(self, state: GovernorState) -> GovernorState:
        """Evaluate pending tool calls against security policies.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with policy decisions
        """
        if not state.pending_tools:
            state.record_error("No pending tools to evaluate")
            state.transition_to(StateNode.RESPOND)
            return state
        
        policy_decisions = []
        tools_requiring_confirmation = []
        approved_tools = []
        denied_tools = []
        
        # Evaluate each pending tool call
        for tool_call in state.pending_tools[:]:  # Create copy to modify during iteration
            decision = await self._evaluate_tool_call(tool_call, state)
            policy_decisions.append(decision)
            
            if decision.decision == DecisionType.ALLOW:
                approved_tools.append(tool_call)
            elif decision.decision == DecisionType.DENY:
                denied_tools.append(tool_call)
                # Remove denied tools from pending
                state.pending_tools.remove(tool_call)
            elif decision.decision == DecisionType.REQUIRE_CONFIRMATION:
                tools_requiring_confirmation.append(tool_call)
                tool_call.requires_confirmation = True
        
        # Store policy evaluation results
        state.context.update({
            "policy_evaluation": {
                "total_tools_evaluated": len(policy_decisions),
                "approved_count": len(approved_tools),
                "denied_count": len(denied_tools),
                "confirmation_required_count": len(tools_requiring_confirmation),
                "policy_decisions": [
                    {
                        "tool_name": decision.tool_call.tool_name,
                        "decision": decision.decision.value,
                        "risk_score": decision.risk_score,
                        "reasoning": decision.reasoning
                    }
                    for decision in policy_decisions
                ],
                "evaluation_completed_at": datetime.utcnow().isoformat()
            }
        })
        
        # Determine next state based on evaluation results
        if tools_requiring_confirmation:
            # Set up confirmation context
            confirmation_context = {
                "tools_awaiting_confirmation": [
                    {
                        "tool_name": tool.tool_name,
                        "arguments": tool.arguments,
                        "risk_level": tool.risk_level.value,
                        "execution_id": tool.execution_id
                    }
                    for tool in tools_requiring_confirmation
                ],
                "confirmation_type": "tool_execution",
                "auto_approve_after": None,  # Manual approval required
            }
            
            state.request_confirmation(confirmation_context)
            state.transition_to(StateNode.AWAIT_CONFIRMATION)
        elif approved_tools:
            state.transition_to(StateNode.EXECUTE)
        else:
            # All tools denied, go to respond with explanation
            state.transition_to(StateNode.RESPOND)
        
        return state
    
    async def _evaluate_tool_call(self, tool_call: ToolCall, state: GovernorState) -> PolicyDecision:
        """Evaluate a single tool call against security policies.
        
        Args:
            tool_call: Tool call to evaluate
            state: Current workflow state
            
        Returns:
            Policy decision for this tool call
        """
        # Calculate risk score based on multiple factors
        risk_score = self._calculate_risk_score(tool_call, state)
        
        # Make policy decision based on risk score and tool characteristics
        decision_type = self._make_policy_decision(risk_score, tool_call, state)
        
        # Generate reasoning for the decision
        reasoning = self._generate_decision_reasoning(decision_type, risk_score, tool_call, state)
        
        # Get applied policy rules
        applied_rules = self._get_applied_rules(tool_call, state)
        
        # Get user permissions context
        user_permissions = self._get_user_permissions(tool_call, state)
        
        return PolicyDecision(
            decision=decision_type,
            risk_score=risk_score,
            reasoning=reasoning,
            tool_call=tool_call,
            policy_version="v1.0",
            applied_rules=applied_rules,
            user_permissions=user_permissions,
            metadata={
                "evaluation_method": "rule_based_v1",
                "session_context": {
                    "error_count": state.error_count,
                    "total_tools_executed": state.total_tools_executed,
                    "session_age_seconds": (datetime.utcnow() - state.created_at).total_seconds()
                }
            }
        )
    
    def _calculate_risk_score(self, tool_call: ToolCall, state: GovernorState) -> float:
        """Calculate risk score for a tool call.
        
        Args:
            tool_call: Tool call to evaluate
            state: Current workflow state
            
        Returns:
            Risk score between 0.0 and 1.0
        """
        base_risk = {
            RiskLevel.SAFE: 0.1,
            RiskLevel.SENSITIVE: 0.5,
            RiskLevel.DANGEROUS: 0.9
        }[tool_call.risk_level]
        
        # Adjust risk based on various factors
        risk_modifiers = 0.0
        
        # User history modifiers
        if state.error_count > 3:
            risk_modifiers += 0.1  # Recent errors increase risk
        
        if state.total_tools_executed > 50:
            risk_modifiers -= 0.05  # Experienced users get slight risk reduction
        
        # Tool-specific modifiers
        high_risk_tools = ["email", "file_manager", "system_command", "database_query"]
        if tool_call.tool_name in high_risk_tools:
            risk_modifiers += 0.2
        
        # Argument-based modifiers
        arguments = tool_call.arguments
        
        # Check for sensitive data in arguments
        sensitive_patterns = ["password", "token", "key", "secret", "credential"]
        if any(pattern in str(arguments).lower() for pattern in sensitive_patterns):
            risk_modifiers += 0.3
        
        # Check for external connections
        external_patterns = ["http://", "https://", "ftp://", "@"]
        if any(pattern in str(arguments) for pattern in external_patterns):
            risk_modifiers += 0.1
        
        # Check for file operations
        file_operations = ["delete", "remove", "rm", "format", "wipe"]
        if any(op in str(arguments).lower() for op in file_operations):
            risk_modifiers += 0.2
        
        # Session context modifiers
        session_age_hours = (datetime.utcnow() - state.created_at).total_seconds() / 3600
        if session_age_hours > 24:
            risk_modifiers += 0.1  # Long sessions increase risk
        
        # Time-based modifiers (higher risk outside business hours)
        current_hour = datetime.utcnow().hour
        if current_hour < 6 or current_hour > 22:  # Night hours
            risk_modifiers += 0.05
        
        # Calculate final risk score
        final_risk = base_risk + risk_modifiers
        
        # Ensure risk score is within bounds
        return max(0.0, min(1.0, final_risk))
    
    def _make_policy_decision(
        self, 
        risk_score: float, 
        tool_call: ToolCall, 
        state: GovernorState
    ) -> DecisionType:
        """Make policy decision based on risk score and other factors.
        
        Args:
            risk_score: Calculated risk score
            tool_call: Tool call being evaluated
            state: Current workflow state
            
        Returns:
            Policy decision type
        """
        # Default thresholds
        deny_threshold = 0.8
        confirmation_threshold = 0.4
        
        # Adjust thresholds based on user trust level
        user_trust_level = self._get_user_trust_level(state)
        
        if user_trust_level == "high":
            deny_threshold = 0.9
            confirmation_threshold = 0.6
        elif user_trust_level == "low":
            deny_threshold = 0.6
            confirmation_threshold = 0.2
        
        # Make decision
        if risk_score >= deny_threshold:
            return DecisionType.DENY
        elif risk_score >= confirmation_threshold:
            return DecisionType.REQUIRE_CONFIRMATION
        else:
            return DecisionType.ALLOW
    
    def _get_user_trust_level(self, state: GovernorState) -> str:
        """Determine user trust level based on history.
        
        Args:
            state: Current workflow state
            
        Returns:
            Trust level: "low", "medium", or "high"
        """
        # Calculate trust based on various factors
        trust_score = 0.5  # Base trust
        
        # Positive factors
        if state.total_tools_executed > 20:
            trust_score += 0.2
        
        if state.error_count == 0:
            trust_score += 0.1
        
        # Negative factors
        if state.error_count > 5:
            trust_score -= 0.3
        
        session_age_days = (datetime.utcnow() - state.created_at).total_seconds() / (24 * 3600)
        if session_age_days > 30:
            trust_score += 0.1  # Long-term user
        
        # Classify trust level
        if trust_score >= 0.7:
            return "high"
        elif trust_score <= 0.3:
            return "low"
        else:
            return "medium"
    
    def _generate_decision_reasoning(
        self, 
        decision: DecisionType, 
        risk_score: float, 
        tool_call: ToolCall, 
        state: GovernorState
    ) -> str:
        """Generate human-readable reasoning for the policy decision.
        
        Args:
            decision: Policy decision made
            risk_score: Calculated risk score
            tool_call: Tool call being evaluated
            state: Current workflow state
            
        Returns:
            Reasoning explanation
        """
        reasoning_parts = [
            f"Risk score: {risk_score:.2f} for {tool_call.tool_name} tool"
        ]
        
        # Add risk level context
        risk_level_text = {
            RiskLevel.SAFE: "low-risk",
            RiskLevel.SENSITIVE: "medium-risk", 
            RiskLevel.DANGEROUS: "high-risk"
        }[tool_call.risk_level]
        
        reasoning_parts.append(f"Base classification: {risk_level_text} tool")
        
        # Add decision-specific reasoning
        if decision == DecisionType.ALLOW:
            reasoning_parts.append("Approved for immediate execution")
        elif decision == DecisionType.DENY:
            reasoning_parts.append("Denied due to high risk score exceeding safety threshold")
        elif decision == DecisionType.REQUIRE_CONFIRMATION:
            reasoning_parts.append("Requires user confirmation due to elevated risk")
        
        # Add context about user trust
        trust_level = self._get_user_trust_level(state)
        reasoning_parts.append(f"User trust level: {trust_level}")
        
        # Add session context if relevant
        if state.error_count > 0:
            reasoning_parts.append(f"Session has {state.error_count} previous error(s)")
        
        return ". ".join(reasoning_parts)
    
    def _get_applied_rules(self, tool_call: ToolCall, state: GovernorState) -> list[str]:
        """Get list of policy rules that were applied to this decision.
        
        Args:
            tool_call: Tool call being evaluated
            state: Current workflow state
            
        Returns:
            List of applied rule names
        """
        applied = []
        
        # Base risk level rules
        applied.append(f"risk_level_{tool_call.risk_level.value}_rule")
        
        # Tool-specific rules
        if tool_call.tool_name in ["email", "file_manager", "system_command"]:
            applied.append("high_impact_tool_rule")
        
        # Argument validation rules
        if any(pattern in str(tool_call.arguments).lower() 
               for pattern in ["password", "token", "key", "secret"]):
            applied.append("sensitive_data_detection_rule")
        
        # User history rules
        if state.error_count > 3:
            applied.append("error_history_risk_adjustment_rule")
        
        # Session context rules
        session_age_hours = (datetime.utcnow() - state.created_at).total_seconds() / 3600
        if session_age_hours > 24:
            applied.append("extended_session_risk_rule")
        
        # Time-based rules
        current_hour = datetime.utcnow().hour
        if current_hour < 6 or current_hour > 22:
            applied.append("off_hours_risk_rule")
        
        return applied
    
    def _get_user_permissions(self, tool_call: ToolCall, state: GovernorState) -> dict[str, Any]:
        """Get relevant user permissions for this tool call.
        
        Args:
            tool_call: Tool call being evaluated
            state: Current workflow state
            
        Returns:
            Dictionary of relevant permissions
        """
        # In production, this would query a user permissions system
        # For now, simulate based on tool type and user history
        
        permissions = {
            "user_id": state.user_id,
            "session_id": state.session_id,
            "permission_level": "standard"  # Would be looked up in real system
        }
        
        # Tool-specific permissions
        tool_permissions = {
            "email": state.total_tools_executed > 5,  # Require experience
            "file_manager": state.error_count < 3,   # Require good track record
            "calendar": True,                        # Generally allowed
            "weather": True,                         # Generally allowed
            "search": True,                          # Generally allowed
            "calculator": True                       # Generally allowed
        }
        
        tool_name = tool_call.tool_name
        if tool_name in tool_permissions:
            permissions[f"{tool_name}_access"] = tool_permissions[tool_name]
        
        # Add trust-based permissions
        trust_level = self._get_user_trust_level(state)
        permissions["trust_level"] = trust_level
        permissions["elevated_privileges"] = trust_level == "high"
        
        return permissions


class PolicyCheckConditional:
    """Conditional routing from policy check state."""
    
    @property
    def name(self) -> str:
        """Get the conditional name."""
        return "policy_check_routing"
    
    def __call__(self, state: GovernorState) -> str:
        """Determine next node from policy check state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node name
        """
        policy_eval = state.context.get("policy_evaluation", {})
        
        # If there was an error in policy evaluation, go to respond
        if state.error_count > 0:
            return StateNode.RESPOND.value
        
        # If tools require confirmation, go to await confirmation
        if policy_eval.get("confirmation_required_count", 0) > 0:
            return StateNode.AWAIT_CONFIRMATION.value
        
        # If tools were approved, go to execute
        if policy_eval.get("approved_count", 0) > 0:
            return StateNode.EXECUTE.value
        
        # If all tools were denied, go to respond with explanation
        return StateNode.RESPOND.value