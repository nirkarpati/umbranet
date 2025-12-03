"""Enhanced Policy Engine implementation.

This module provides the main PolicyEngine class that evaluates
tool execution requests against security policies and user permissions.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

from ...core.domain.state import GovernorState
from ...core.domain.tools import DecisionType, PolicyDecision, ToolCall
from .rules import PolicyRule, RiskAssessment

logger = logging.getLogger(__name__)


class PolicyEngine:
    """Enhanced security firewall for tool execution.
    
    The PolicyEngine evaluates tool calls against multiple security rules
    and user permissions to determine whether execution should be allowed,
    denied, or require user confirmation.
    """
    
    def __init__(self, custom_rules: List[PolicyRule] | None = None):
        """Initialize the policy engine.
        
        Args:
            custom_rules: Optional custom policy rules to use instead of defaults
        """
        self.risk_assessment = RiskAssessment(custom_rules)
        self.policy_version = "v1.1"  # Enhanced version
        
    def evaluate_tool_call(
        self, 
        tool_call: ToolCall, 
        state: GovernorState
    ) -> PolicyDecision:
        """Evaluate whether a tool call should be allowed.
        
        Args:
            tool_call: Tool call to evaluate
            state: Current conversation state for context
            
        Returns:
            PolicyDecision with evaluation results
        """
        start_time = datetime.utcnow()
        
        # Build context for rule evaluation
        context = self._build_evaluation_context(tool_call, state)
        
        # Perform risk assessment
        risk_score, applied_rules = self.risk_assessment.assess_risk(tool_call, context)
        
        # Make policy decision
        decision_type = self.risk_assessment.make_decision(risk_score, tool_call)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(decision_type, risk_score, tool_call, applied_rules)
        
        # Calculate evaluation time
        evaluation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Get user permissions context
        user_permissions = self._get_user_permissions_context(tool_call, state)
        
        policy_decision = PolicyDecision(
            decision=decision_type,
            risk_score=risk_score,
            reasoning=reasoning,
            tool_call=tool_call,
            policy_version=self.policy_version,
            evaluated_at=start_time,
            evaluation_time_ms=evaluation_time,
            applied_rules=applied_rules,
            user_permissions=user_permissions,
            metadata={
                "evaluation_method": "enhanced_multi_rule_v1.1",
                "context_factors": list(context.keys()),
                "decision_thresholds": {
                    "deny_threshold": 0.8,
                    "confirmation_threshold": 0.4
                }
            }
        )
        
        logger.info(
            f"Policy evaluation for {tool_call.tool_name}: "
            f"risk={risk_score:.2f}, decision={decision_type.value}, "
            f"time={evaluation_time:.1f}ms"
        )
        
        return policy_decision
    
    def _build_evaluation_context(self, tool_call: ToolCall, state: GovernorState) -> Dict[str, Any]:
        """Build context information for policy rule evaluation.
        
        Args:
            tool_call: Tool call being evaluated
            state: Current conversation state
            
        Returns:
            Context dictionary for rule evaluation
        """
        session_age_hours = (datetime.utcnow() - state.created_at).total_seconds() / 3600
        
        user_context = {
            "user_id": state.user_id,
            "error_count": state.error_count,
            "total_tools_executed": state.total_tools_executed,
            "session_age_days": session_age_hours / 24,
            "success_rate": self._calculate_success_rate(state)
        }
        
        session_context = {
            "session_id": state.session_id,
            "session_age_hours": session_age_hours,
            "recent_errors": min(state.error_count, 10),  # Cap for calculation
            "rapid_requests": self._detect_rapid_requests(state),
            "conversation_turns": state.total_turns
        }
        
        tool_context = {
            "tool_name": tool_call.tool_name,
            "risk_level": tool_call.risk_level.value,
            "argument_count": len(tool_call.arguments),
            "requires_auth": getattr(tool_call, 'requires_auth', False)
        }
        
        return {
            "user_context": user_context,
            "session_context": session_context,
            "tool_context": tool_context,
            "evaluation_time": datetime.utcnow().isoformat()
        }
    
    def _calculate_success_rate(self, state: GovernorState) -> float:
        """Calculate user's tool execution success rate.
        
        Args:
            state: Current conversation state
            
        Returns:
            Success rate between 0.0 and 1.0
        """
        if state.total_tools_executed == 0:
            return 1.0  # No history = perfect score
        
        # Simple calculation: assume errors indicate failed tools
        failures = min(state.error_count, state.total_tools_executed)
        return max(0.0, (state.total_tools_executed - failures) / state.total_tools_executed)
    
    def _detect_rapid_requests(self, state: GovernorState) -> bool:
        """Detect if user is making requests too rapidly.
        
        Args:
            state: Current conversation state
            
        Returns:
            True if rapid requests detected
        """
        # Simple heuristic: more than 10 turns in less than 5 minutes
        session_minutes = (datetime.utcnow() - state.created_at).total_seconds() / 60
        if session_minutes < 5 and state.total_turns > 10:
            return True
        
        # Or very high turn rate
        if session_minutes > 0 and (state.total_turns / session_minutes) > 5:
            return True
        
        return False
    
    def _generate_reasoning(
        self, 
        decision: DecisionType, 
        risk_score: float, 
        tool_call: ToolCall,
        applied_rules: List[str]
    ) -> str:
        """Generate human-readable reasoning for the policy decision.
        
        Args:
            decision: Policy decision made
            risk_score: Calculated risk score
            tool_call: Tool call being evaluated
            applied_rules: List of rules that were applied
            
        Returns:
            Human-readable reasoning explanation
        """
        reasoning_parts = [
            f"Tool '{tool_call.tool_name}' evaluated with risk score {risk_score:.2f}"
        ]
        
        # Add base risk level context
        risk_descriptions = {
            "safe": "low-risk tool with minimal security concerns",
            "sensitive": "medium-risk tool requiring enhanced monitoring",
            "dangerous": "high-risk tool with significant security implications"
        }
        
        risk_desc = risk_descriptions.get(
            tool_call.risk_level.value,
            "unknown risk level"
        )
        reasoning_parts.append(f"Base classification: {risk_desc}")
        
        # Add decision-specific reasoning
        if decision == DecisionType.ALLOW:
            reasoning_parts.append("Approved for immediate execution based on low risk assessment")
        elif decision == DecisionType.DENY:
            reasoning_parts.append("Denied due to high risk score exceeding security threshold")
        elif decision == DecisionType.REQUIRE_CONFIRMATION:
            reasoning_parts.append("User confirmation required due to elevated risk factors")
        
        # Add applied rules information
        if applied_rules:
            reasoning_parts.append(f"Applied {len(applied_rules)} security rules: {', '.join(applied_rules[:3])}{'...' if len(applied_rules) > 3 else ''}")
        
        return ". ".join(reasoning_parts)
    
    def _get_user_permissions_context(self, tool_call: ToolCall, state: GovernorState) -> Dict[str, Any]:
        """Get user permissions context for the decision.
        
        Args:
            tool_call: Tool call being evaluated
            state: Current conversation state
            
        Returns:
            User permissions context dictionary
        """
        # Query user permissions based on state and tool requirements
        
        permissions = {
            "user_id": state.user_id,
            "session_id": state.session_id,
            "permission_level": self._determine_permission_level(state),
            "tool_access_granted": True  # Basic access granted based on authentication
        }
        
        # Tool-specific permissions evaluation
        tool_permissions = {
            "email_access": state.total_tools_executed > 5 and state.error_count < 3,
            "file_access": state.error_count < 2,
            "external_api_access": state.total_tools_executed > 10,
            "sensitive_data_access": state.total_tools_executed > 20 and state.error_count == 0
        }
        
        permissions.update(tool_permissions)
        
        return permissions
    
    def _determine_permission_level(self, state: GovernorState) -> str:
        """Determine user's permission level based on history.
        
        Args:
            state: Current conversation state
            
        Returns:
            Permission level string
        """
        if state.total_tools_executed > 50 and state.error_count < 5:
            return "elevated"
        elif state.total_tools_executed > 20 and state.error_count < 10:
            return "standard_plus"
        elif state.error_count > state.total_tools_executed * 0.3:
            return "restricted"
        else:
            return "standard"
    
    def get_policy_stats(self) -> Dict[str, Any]:
        """Get policy engine statistics.
        
        Returns:
            Dictionary with policy engine statistics
        """
        return {
            "policy_version": self.policy_version,
            "active_rules": len(self.risk_assessment.rules),
            "rule_names": [rule.name for rule in self.risk_assessment.rules],
            "rule_weights": {rule.name: rule.weight for rule in self.risk_assessment.rules}
        }
    
    def add_custom_rule(self, rule: PolicyRule) -> None:
        """Add a custom policy rule to the engine.
        
        Args:
            rule: Custom policy rule to add
        """
        self.risk_assessment.rules.append(rule)
        logger.info(f"Added custom policy rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a policy rule by name.
        
        Args:
            rule_name: Name of the rule to remove
            
        Returns:
            True if rule was removed, False if not found
        """
        initial_count = len(self.risk_assessment.rules)
        self.risk_assessment.rules = [
            rule for rule in self.risk_assessment.rules
            if rule.name != rule_name
        ]
        removed = len(self.risk_assessment.rules) < initial_count
        
        if removed:
            logger.info(f"Removed policy rule: {rule_name}")
        
        return removed