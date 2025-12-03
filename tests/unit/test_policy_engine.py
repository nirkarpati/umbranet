"""Unit tests for the enhanced policy engine."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from src.core.domain.state import GovernorState, StateNode
from src.core.domain.tools import RiskLevel, ToolCall, DecisionType
from src.action_plane.policy_engine import PolicyEngine, PolicyRule, RiskAssessment
from src.action_plane.policy_engine.rules import (
    BaseRiskLevelRule, UserHistoryRule, ArgumentSensitivityRule,
    TimeBasedRule, SessionContextRule
)


class TestPolicyRules:
    """Test cases for individual policy rules."""
    
    def test_base_risk_level_rule(self):
        """Test base risk level rule evaluation."""
        rule = BaseRiskLevelRule()
        
        assert rule.name == "base_risk_level"
        assert rule.weight == 0.4
        
        # Test different risk levels
        safe_call = self._create_tool_call("safe_tool", RiskLevel.SAFE)
        sensitive_call = self._create_tool_call("sensitive_tool", RiskLevel.SENSITIVE)
        dangerous_call = self._create_tool_call("dangerous_tool", RiskLevel.DANGEROUS)
        
        context = {}
        
        assert rule.evaluate(safe_call, context) == 0.1
        assert rule.evaluate(sensitive_call, context) == 0.5
        assert rule.evaluate(dangerous_call, context) == 0.9
    
    def test_user_history_rule(self):
        """Test user history rule evaluation."""
        rule = UserHistoryRule()
        
        assert rule.name == "user_history"
        assert rule.weight == 0.2
        
        tool_call = self._create_tool_call("test_tool", RiskLevel.SAFE)
        
        # New user with no history
        context = {"user_context": {"error_count": 0, "total_tools_executed": 0, "session_age_days": 0}}
        score = rule.evaluate(tool_call, context)
        assert score == 0.5  # Base score
        
        # Experienced user with good track record
        context = {"user_context": {"error_count": 2, "total_tools_executed": 100, "session_age_days": 60}}
        score = rule.evaluate(tool_call, context)
        assert score < 0.5  # Should be lower due to experience and low error rate
        
        # User with high error rate
        context = {"user_context": {"error_count": 30, "total_tools_executed": 100, "session_age_days": 10}}
        score = rule.evaluate(tool_call, context)
        assert score > 0.5  # Should be higher due to high error rate
    
    def test_argument_sensitivity_rule(self):
        """Test argument sensitivity rule evaluation."""
        rule = ArgumentSensitivityRule()
        
        assert rule.name == "argument_sensitivity"
        assert rule.weight == 0.2
        
        # Safe arguments
        safe_call = self._create_tool_call("test_tool", RiskLevel.SAFE, {"query": "weather"})
        score = rule.evaluate(safe_call, {})
        assert score == 0.0
        
        # Arguments with sensitive data
        sensitive_call = self._create_tool_call("test_tool", RiskLevel.SAFE, {"password": "secret123"})
        score = rule.evaluate(sensitive_call, {})
        assert score > 0.0
        
        # Arguments with external URLs
        external_call = self._create_tool_call("test_tool", RiskLevel.SAFE, {"url": "https://external.com"})
        score = rule.evaluate(external_call, {})
        assert score > 0.0
        
        # Arguments with destructive operations
        destructive_call = self._create_tool_call("test_tool", RiskLevel.SAFE, {"action": "delete all files"})
        score = rule.evaluate(destructive_call, {})
        assert score > 0.0
    
    @patch('src.action_plane.policy_engine.rules.datetime')
    def test_time_based_rule(self, mock_datetime):
        """Test time-based rule evaluation."""
        rule = TimeBasedRule()
        
        assert rule.name == "time_based"
        assert rule.weight == 0.1
        
        tool_call = self._create_tool_call("test_tool", RiskLevel.SAFE)
        
        # Business hours (10 AM)
        mock_datetime.utcnow.return_value = datetime(2024, 1, 1, 10, 0)
        score = rule.evaluate(tool_call, {})
        assert score == 0.2
        
        # Evening (9 PM)
        mock_datetime.utcnow.return_value = datetime(2024, 1, 1, 21, 0)
        score = rule.evaluate(tool_call, {})
        assert score == 0.4
        
        # Night time (2 AM)
        mock_datetime.utcnow.return_value = datetime(2024, 1, 1, 2, 0)
        score = rule.evaluate(tool_call, {})
        assert score == 0.7
    
    def test_session_context_rule(self):
        """Test session context rule evaluation."""
        rule = SessionContextRule()
        
        assert rule.name == "session_context"
        assert rule.weight == 0.1
        
        tool_call = self._create_tool_call("test_tool", RiskLevel.SAFE)
        
        # Normal session
        context = {"session_context": {"session_age_hours": 2, "recent_errors": 0, "rapid_requests": False}}
        score = rule.evaluate(tool_call, context)
        assert score == 0.2  # Base score
        
        # Long session with errors
        context = {"session_context": {"session_age_hours": 30, "recent_errors": 5, "rapid_requests": True}}
        score = rule.evaluate(tool_call, context)
        assert score > 0.5  # Should be high due to multiple risk factors
    
    def _create_tool_call(self, name: str, risk_level: RiskLevel, arguments: dict = None) -> ToolCall:
        """Helper to create tool calls for testing."""
        return ToolCall(
            tool_name=name,
            arguments=arguments or {},
            risk_level=risk_level,
            execution_id="exec_123",
            user_id="user_123",
            session_id="session_456"
        )


class TestRiskAssessment:
    """Test cases for the RiskAssessment class."""
    
    def test_default_rules_initialization(self):
        """Test that default rules are properly initialized."""
        assessment = RiskAssessment()
        
        assert len(assessment.rules) == 5
        rule_names = [rule.name for rule in assessment.rules]
        
        expected_rules = ["base_risk_level", "user_history", "argument_sensitivity", "time_based", "session_context"]
        for expected in expected_rules:
            assert expected in rule_names
    
    def test_custom_rules_initialization(self):
        """Test initialization with custom rules."""
        custom_rule = BaseRiskLevelRule()
        assessment = RiskAssessment([custom_rule])
        
        assert len(assessment.rules) == 1
        assert assessment.rules[0].name == "base_risk_level"
    
    def test_risk_assessment_calculation(self):
        """Test overall risk assessment calculation."""
        assessment = RiskAssessment()
        
        tool_call = ToolCall(
            tool_name="test_tool",
            arguments={},
            risk_level=RiskLevel.SAFE,
            execution_id="exec_123",
            user_id="user_123",
            session_id="session_456"
        )
        
        context = {
            "user_context": {"error_count": 0, "total_tools_executed": 10, "session_age_days": 5},
            "session_context": {"session_age_hours": 2, "recent_errors": 0, "rapid_requests": False}
        }
        
        risk_score, applied_rules = assessment.assess_risk(tool_call, context)
        
        assert 0.0 <= risk_score <= 1.0
        assert len(applied_rules) == 5  # All default rules applied
    
    def test_decision_making(self):
        """Test policy decision making based on risk scores."""
        assessment = RiskAssessment()
        
        safe_call = ToolCall(
            tool_name="safe_tool", arguments={}, risk_level=RiskLevel.SAFE,
            execution_id="exec_1", user_id="user_1", session_id="session_1"
        )
        
        dangerous_call = ToolCall(
            tool_name="dangerous_tool", arguments={}, risk_level=RiskLevel.DANGEROUS,
            execution_id="exec_2", user_id="user_1", session_id="session_1"
        )
        
        # Low risk score should allow
        decision = assessment.make_decision(0.2, safe_call)
        assert decision == DecisionType.ALLOW
        
        # Medium risk score should require confirmation (for safe tools, threshold is 0.7)
        decision = assessment.make_decision(0.75, safe_call)
        assert decision == DecisionType.REQUIRE_CONFIRMATION
        
        # High risk score should deny
        decision = assessment.make_decision(0.9, safe_call)
        assert decision == DecisionType.DENY
        
        # Dangerous tools have lower confirmation threshold
        decision = assessment.make_decision(0.35, dangerous_call)
        assert decision == DecisionType.REQUIRE_CONFIRMATION


class TestPolicyEngine:
    """Test cases for the PolicyEngine class."""
    
    def test_policy_engine_initialization(self):
        """Test policy engine initialization."""
        engine = PolicyEngine()
        
        assert engine.policy_version == "v1.1"
        assert isinstance(engine.risk_assessment, RiskAssessment)
    
    def test_tool_call_evaluation(self):
        """Test complete tool call evaluation."""
        engine = PolicyEngine()
        
        # Create a test state
        state = GovernorState(
            user_id="user_123",
            session_id="session_456",
            current_node=StateNode.POLICY_CHECK
        )
        state.total_tools_executed = 25
        state.error_count = 2
        
        # Create tool call
        tool_call = ToolCall(
            tool_name="test_tool",
            arguments={"param": "value"},
            risk_level=RiskLevel.SENSITIVE,
            execution_id="exec_123",
            user_id="user_123",
            session_id="session_456"
        )
        
        # Evaluate
        decision = engine.evaluate_tool_call(tool_call, state)
        
        assert isinstance(decision.risk_score, float)
        assert 0.0 <= decision.risk_score <= 1.0
        assert decision.decision in [DecisionType.ALLOW, DecisionType.DENY, DecisionType.REQUIRE_CONFIRMATION]
        assert decision.tool_call == tool_call
        assert decision.policy_version == "v1.1"
        assert len(decision.applied_rules) > 0
        assert decision.reasoning is not None
        assert decision.evaluation_time_ms is not None
        assert decision.user_permissions is not None
    
    def test_context_building(self):
        """Test evaluation context building."""
        engine = PolicyEngine()
        
        # Create state with some history
        state = GovernorState(user_id="user_123", session_id="session_456")
        state.total_tools_executed = 50
        state.error_count = 3
        state.total_turns = 25
        
        # Set creation time to 2 hours ago
        state.created_at = datetime.utcnow() - timedelta(hours=2)
        
        tool_call = ToolCall(
            tool_name="context_tool",
            arguments={"test": "value"},
            risk_level=RiskLevel.SAFE,
            execution_id="exec_123",
            user_id="user_123",
            session_id="session_456"
        )
        
        context = engine._build_evaluation_context(tool_call, state)
        
        assert "user_context" in context
        assert "session_context" in context
        assert "tool_context" in context
        
        user_ctx = context["user_context"]
        assert user_ctx["user_id"] == "user_123"
        assert user_ctx["total_tools_executed"] == 50
        assert user_ctx["error_count"] == 3
        assert 0.0 <= user_ctx["success_rate"] <= 1.0
        
        session_ctx = context["session_context"]
        assert session_ctx["session_id"] == "session_456"
        assert session_ctx["session_age_hours"] >= 2.0
        assert session_ctx["conversation_turns"] == 25
        
        tool_ctx = context["tool_context"]
        assert tool_ctx["tool_name"] == "context_tool"
        assert tool_ctx["risk_level"] == "safe"
    
    def test_permission_level_determination(self):
        """Test user permission level determination."""
        engine = PolicyEngine()
        
        # New user
        new_user_state = GovernorState(user_id="new_user", session_id="session_1")
        permission_level = engine._determine_permission_level(new_user_state)
        assert permission_level == "standard"
        
        # Experienced user with good record
        experienced_state = GovernorState(user_id="experienced", session_id="session_2")
        experienced_state.total_tools_executed = 60
        experienced_state.error_count = 2
        permission_level = engine._determine_permission_level(experienced_state)
        assert permission_level == "elevated"
        
        # User with many errors
        error_prone_state = GovernorState(user_id="error_user", session_id="session_3")
        error_prone_state.total_tools_executed = 20
        error_prone_state.error_count = 10  # 50% error rate
        permission_level = engine._determine_permission_level(error_prone_state)
        assert permission_level == "restricted"
    
    def test_rapid_request_detection(self):
        """Test rapid request detection."""
        engine = PolicyEngine()
        
        # Normal usage
        normal_state = GovernorState(user_id="normal", session_id="session_1")
        normal_state.total_turns = 5
        normal_state.created_at = datetime.utcnow() - timedelta(minutes=30)
        assert engine._detect_rapid_requests(normal_state) is False
        
        # Rapid requests - many turns in short time
        rapid_state = GovernorState(user_id="rapid", session_id="session_2")
        rapid_state.total_turns = 15
        rapid_state.created_at = datetime.utcnow() - timedelta(minutes=3)
        assert engine._detect_rapid_requests(rapid_state) is True
        
        # Very high turn rate
        high_rate_state = GovernorState(user_id="high_rate", session_id="session_3")
        high_rate_state.total_turns = 60
        high_rate_state.created_at = datetime.utcnow() - timedelta(minutes=10)
        assert engine._detect_rapid_requests(high_rate_state) is True
    
    def test_policy_stats(self):
        """Test policy engine statistics."""
        engine = PolicyEngine()
        
        stats = engine.get_policy_stats()
        
        assert stats["policy_version"] == "v1.1"
        assert stats["active_rules"] == 5  # Default rules
        assert len(stats["rule_names"]) == 5
        assert len(stats["rule_weights"]) == 5
        assert all(isinstance(weight, float) for weight in stats["rule_weights"].values())
    
    def test_custom_rule_management(self):
        """Test adding and removing custom rules."""
        engine = PolicyEngine()
        
        initial_rule_count = len(engine.risk_assessment.rules)
        
        # Add custom rule
        custom_rule = BaseRiskLevelRule()  # Use existing rule as custom
        engine.add_custom_rule(custom_rule)
        
        assert len(engine.risk_assessment.rules) == initial_rule_count + 1
        
        # Remove rule (there are now 2 base_risk_level rules, so count should go down by 2)
        removed = engine.remove_rule("base_risk_level")
        assert removed is True
        assert len(engine.risk_assessment.rules) == initial_rule_count - 1
        
        # Try to remove non-existent rule
        removed = engine.remove_rule("non_existent_rule")
        assert removed is False