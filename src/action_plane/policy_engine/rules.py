"""Policy rules and risk assessment logic.

This module defines the policy rules and risk assessment algorithms
used by the policy engine to evaluate tool execution requests.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List

from ...core.domain.tools import RiskLevel, ToolCall, DecisionType


class PolicyRule(ABC):
    """Abstract base class for policy rules.
    
    Policy rules evaluate specific aspects of a tool call
    and contribute to the overall risk assessment.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the rule name for logging and debugging."""
        pass
    
    @property
    @abstractmethod
    def weight(self) -> float:
        """Get the rule weight for risk calculation (0.0 to 1.0)."""
        pass
    
    @abstractmethod
    def evaluate(self, tool_call: ToolCall, context: Dict[str, Any]) -> float:
        """Evaluate the rule against a tool call.
        
        Args:
            tool_call: Tool call to evaluate
            context: Additional context (user history, session data, etc.)
            
        Returns:
            Risk score contribution (0.0 = no risk, 1.0 = maximum risk)
        """
        pass


class BaseRiskLevelRule(PolicyRule):
    """Base rule that maps tool risk levels to scores."""
    
    @property
    def name(self) -> str:
        return "base_risk_level"
    
    @property 
    def weight(self) -> float:
        return 0.4  # 40% weight for base risk level
    
    def evaluate(self, tool_call: ToolCall, context: Dict[str, Any]) -> float:
        risk_mapping = {
            RiskLevel.SAFE: 0.1,
            RiskLevel.SENSITIVE: 0.5,
            RiskLevel.DANGEROUS: 0.9
        }
        return risk_mapping.get(tool_call.risk_level, 0.5)


class UserHistoryRule(PolicyRule):
    """Rule that considers user's historical behavior."""
    
    @property
    def name(self) -> str:
        return "user_history"
    
    @property
    def weight(self) -> float:
        return 0.2  # 20% weight for user history
    
    def evaluate(self, tool_call: ToolCall, context: Dict[str, Any]) -> float:
        user_context = context.get("user_context", {})
        
        # Get user statistics
        error_count = user_context.get("error_count", 0)
        total_tools_executed = user_context.get("total_tools_executed", 0)
        session_age_days = user_context.get("session_age_days", 0)
        
        risk_adjustment = 0.0
        
        # Increase risk for users with many errors
        if total_tools_executed > 0:
            error_rate = error_count / total_tools_executed
            if error_rate > 0.2:  # More than 20% error rate
                risk_adjustment += 0.3
            elif error_rate > 0.1:  # More than 10% error rate
                risk_adjustment += 0.1
        
        # Decrease risk for experienced users
        if total_tools_executed > 50:
            risk_adjustment -= 0.1
        
        # Slight decrease for long-term users
        if session_age_days > 30:
            risk_adjustment -= 0.05
        
        return max(0.0, min(1.0, risk_adjustment + 0.5))  # Base + adjustment


class ArgumentSensitivityRule(PolicyRule):
    """Rule that analyzes tool arguments for sensitive data."""
    
    @property
    def name(self) -> str:
        return "argument_sensitivity"
    
    @property
    def weight(self) -> float:
        return 0.2  # 20% weight for argument analysis
    
    def evaluate(self, tool_call: ToolCall, context: Dict[str, Any]) -> float:
        arguments_str = str(tool_call.arguments).lower()
        
        # Check for sensitive patterns
        sensitive_patterns = [
            "password", "token", "key", "secret", "credential",
            "api_key", "auth", "login", "private"
        ]
        
        external_patterns = [
            "http://", "https://", "ftp://", "@", "://",
            "delete", "remove", "rm", "destroy", "wipe"
        ]
        
        risk_score = 0.0
        
        # Check for sensitive data
        for pattern in sensitive_patterns:
            if pattern in arguments_str:
                risk_score += 0.3
        
        # Check for external connections or destructive operations
        for pattern in external_patterns:
            if pattern in arguments_str:
                risk_score += 0.2
        
        return min(1.0, risk_score)


class TimeBasedRule(PolicyRule):
    """Rule that considers time-based factors."""
    
    @property
    def name(self) -> str:
        return "time_based"
    
    @property
    def weight(self) -> float:
        return 0.1  # 10% weight for time factors
    
    def evaluate(self, tool_call: ToolCall, context: Dict[str, Any]) -> float:
        current_hour = datetime.utcnow().hour
        
        # Increase risk during off-hours (night time)
        if current_hour < 6 or current_hour > 22:
            return 0.7
        elif current_hour < 8 or current_hour > 20:
            return 0.4
        else:
            return 0.2  # Normal business hours


class SessionContextRule(PolicyRule):
    """Rule that considers current session context."""
    
    @property
    def name(self) -> str:
        return "session_context"
    
    @property
    def weight(self) -> float:
        return 0.1  # 10% weight for session context
    
    def evaluate(self, tool_call: ToolCall, context: Dict[str, Any]) -> float:
        session_context = context.get("session_context", {})
        
        # Get session statistics
        session_age_hours = session_context.get("session_age_hours", 0)
        recent_errors = session_context.get("recent_errors", 0)
        rapid_requests = session_context.get("rapid_requests", False)
        
        risk_adjustment = 0.0
        
        # Long sessions might be suspicious
        if session_age_hours > 24:
            risk_adjustment += 0.3
        elif session_age_hours > 8:
            risk_adjustment += 0.1
        
        # Recent errors increase risk
        if recent_errors > 3:
            risk_adjustment += 0.4
        elif recent_errors > 1:
            risk_adjustment += 0.2
        
        # Rapid requests might indicate automation
        if rapid_requests:
            risk_adjustment += 0.3
        
        return max(0.0, min(1.0, risk_adjustment + 0.2))  # Base + adjustment


class RiskAssessment:
    """Comprehensive risk assessment using multiple rules."""
    
    def __init__(self, rules: List[PolicyRule] | None = None):
        """Initialize risk assessment with policy rules.
        
        Args:
            rules: List of policy rules to apply. If None, uses default rules.
        """
        self.rules = rules or self._get_default_rules()
    
    def _get_default_rules(self) -> List[PolicyRule]:
        """Get the default set of policy rules."""
        return [
            BaseRiskLevelRule(),
            UserHistoryRule(),
            ArgumentSensitivityRule(),
            TimeBasedRule(),
            SessionContextRule()
        ]
    
    def assess_risk(self, tool_call: ToolCall, context: Dict[str, Any]) -> tuple[float, List[str]]:
        """Assess overall risk for a tool call.
        
        Args:
            tool_call: Tool call to assess
            context: Context information for assessment
            
        Returns:
            Tuple of (risk_score, applied_rules)
        """
        total_score = 0.0
        total_weight = 0.0
        applied_rules = []
        
        for rule in self.rules:
            rule_score = rule.evaluate(tool_call, context)
            weighted_score = rule_score * rule.weight
            
            total_score += weighted_score
            total_weight += rule.weight
            applied_rules.append(rule.name)
        
        # Normalize score by total weight
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.5  # Default to medium risk
        
        return min(1.0, max(0.0, final_score)), applied_rules
    
    def make_decision(self, risk_score: float, tool_call: ToolCall) -> DecisionType:
        """Make policy decision based on risk score.
        
        Args:
            risk_score: Calculated risk score (0.0 to 1.0)
            tool_call: Tool call being evaluated
            
        Returns:
            Policy decision type
        """
        # Configurable thresholds
        deny_threshold = 0.8
        confirmation_threshold = 0.4
        
        # Adjust thresholds based on tool's base risk level
        if tool_call.risk_level == RiskLevel.DANGEROUS:
            confirmation_threshold = 0.3  # Lower threshold for dangerous tools
        elif tool_call.risk_level == RiskLevel.SAFE:
            confirmation_threshold = 0.7  # Higher threshold for safe tools
        
        if risk_score >= deny_threshold:
            return DecisionType.DENY
        elif risk_score >= confirmation_threshold:
            return DecisionType.REQUIRE_CONFIRMATION
        else:
            return DecisionType.ALLOW