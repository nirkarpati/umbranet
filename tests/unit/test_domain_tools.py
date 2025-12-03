"""Unit tests for domain tool models."""

from datetime import datetime
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.core.domain.tools import (
    DecisionType,
    PolicyDecision,
    RiskLevel,
    ToolCall,
    ToolDefinition,
    ToolStatus,
)


class TestRiskLevel:
    """Test cases for RiskLevel enum."""

    def test_all_risk_levels(self) -> None:
        """Test that all expected risk levels exist."""
        expected_levels = {"safe", "sensitive", "dangerous"}
        actual_levels = {level.value for level in RiskLevel}
        assert actual_levels == expected_levels


class TestDecisionType:
    """Test cases for DecisionType enum."""

    def test_all_decision_types(self) -> None:
        """Test that all expected decision types exist."""
        expected_types = {"allow", "deny", "require_confirmation"}
        actual_types = {decision.value for decision in DecisionType}
        assert actual_types == expected_types


class TestToolStatus:
    """Test cases for ToolStatus enum."""

    def test_all_tool_statuses(self) -> None:
        """Test that all expected tool statuses exist."""
        expected_statuses = {
            "pending", "executing", "completed", "failed", "cancelled", "timeout"
        }
        actual_statuses = {status.value for status in ToolStatus}
        assert actual_statuses == expected_statuses


class TestToolCall:
    """Test cases for ToolCall model."""

    def test_valid_tool_call(self) -> None:
        """Test creating a valid ToolCall."""
        tool_call = ToolCall(
            tool_name="get_weather",
            arguments={"location": "Seattle, WA"},
            risk_level=RiskLevel.SAFE,
            execution_id="exec_123",
            user_id="user_123",
            session_id="session_456"
        )
        
        assert tool_call.tool_name == "get_weather"
        assert tool_call.arguments == {"location": "Seattle, WA"}
        assert tool_call.risk_level == RiskLevel.SAFE
        assert tool_call.execution_id == "exec_123"
        assert tool_call.user_id == "user_123"
        assert tool_call.session_id == "session_456"
        assert tool_call.status == ToolStatus.PENDING  # Default
        assert tool_call.requires_confirmation is False  # Default
        assert tool_call.retry_count == 0
        assert tool_call.max_retries == 3
        assert tool_call.timeout_seconds == 30.0

    def test_tool_call_validation_invalid_name(self) -> None:
        """Test that invalid tool names are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ToolCall(
                tool_name="invalid-tool!",  # Invalid characters
                arguments={},
                risk_level=RiskLevel.SAFE,
                execution_id="exec_123",
                user_id="user_123",
                session_id="session_456"
            )
        
        assert "alphanumeric with underscores" in str(exc_info.value)

    def test_tool_call_validation_invalid_timeout(self) -> None:
        """Test that invalid timeouts are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ToolCall(
                tool_name="test_tool",
                arguments={},
                risk_level=RiskLevel.SAFE,
                execution_id="exec_123",
                user_id="user_123",
                session_id="session_456",
                timeout_seconds=500.0  # Too high
            )
        
        assert "between 0 and 300 seconds" in str(exc_info.value)

    def test_tool_call_execution_lifecycle(self) -> None:
        """Test the tool execution lifecycle methods."""
        tool_call = ToolCall(
            tool_name="test_tool",
            arguments={},
            risk_level=RiskLevel.SAFE,
            execution_id="exec_123",
            user_id="user_123",
            session_id="session_456"
        )
        
        assert tool_call.status == ToolStatus.PENDING
        assert tool_call.started_at is None
        assert tool_call.completed_at is None
        assert tool_call.result is None
        assert tool_call.error_message is None
        assert tool_call.execution_time_ms is None
        
        # Start execution
        tool_call.start_execution()
        assert tool_call.status == ToolStatus.EXECUTING
        assert isinstance(tool_call.started_at, datetime)
        
        # Complete execution successfully
        result = {"output": "success"}
        tool_call.complete_execution(result)
        assert tool_call.status == ToolStatus.COMPLETED
        assert tool_call.result == result
        assert isinstance(tool_call.completed_at, datetime)
        assert isinstance(tool_call.execution_time_ms, float)
        assert tool_call.execution_time_ms >= 0

    def test_tool_call_execution_failure(self) -> None:
        """Test tool execution failure."""
        tool_call = ToolCall(
            tool_name="test_tool",
            arguments={},
            risk_level=RiskLevel.SAFE,
            execution_id="exec_123",
            user_id="user_123",
            session_id="session_456"
        )
        
        tool_call.start_execution()
        tool_call.fail_execution("Connection error")
        
        assert tool_call.status == ToolStatus.FAILED
        assert tool_call.error_message == "Connection error"
        assert isinstance(tool_call.completed_at, datetime)
        assert isinstance(tool_call.execution_time_ms, float)

    def test_tool_call_cancellation(self) -> None:
        """Test tool execution cancellation."""
        tool_call = ToolCall(
            tool_name="test_tool",
            arguments={},
            risk_level=RiskLevel.SAFE,
            execution_id="exec_123",
            user_id="user_123",
            session_id="session_456"
        )
        
        tool_call.cancel_execution()
        
        assert tool_call.status == ToolStatus.CANCELLED
        assert isinstance(tool_call.completed_at, datetime)

    def test_tool_call_retry_mechanism(self) -> None:
        """Test retry mechanism."""
        tool_call = ToolCall(
            tool_name="test_tool",
            arguments={},
            risk_level=RiskLevel.SAFE,
            execution_id="exec_123",
            user_id="user_123",
            session_id="session_456",
            max_retries=2
        )
        
        assert tool_call.retry_count == 0
        
        # First retry
        can_retry = tool_call.increment_retry()
        assert can_retry is True
        assert tool_call.retry_count == 1
        
        # Second retry
        can_retry = tool_call.increment_retry()
        assert can_retry is True
        assert tool_call.retry_count == 2
        
        # Third retry (should fail)
        can_retry = tool_call.increment_retry()
        assert can_retry is False
        assert tool_call.retry_count == 3


class TestPolicyDecision:
    """Test cases for PolicyDecision model."""

    def test_valid_policy_decision_allow(self) -> None:
        """Test creating a valid ALLOW policy decision."""
        tool_call = ToolCall(
            tool_name="get_weather",
            arguments={"location": "Seattle"},
            risk_level=RiskLevel.SAFE,
            execution_id="exec_123",
            user_id="user_123",
            session_id="session_456"
        )
        
        decision = PolicyDecision(
            decision=DecisionType.ALLOW,
            risk_score=0.2,
            reasoning="Safe tool with valid arguments",
            tool_call=tool_call,
            applied_rules=["safe_tool_rule"],
            user_permissions={"weather_access": True}
        )
        
        assert decision.decision == DecisionType.ALLOW
        assert decision.risk_score == 0.2
        assert decision.reasoning == "Safe tool with valid arguments"
        assert decision.tool_call == tool_call
        assert decision.policy_version == "v1.0"  # Default
        assert decision.applied_rules == ["safe_tool_rule"]
        assert decision.user_permissions == {"weather_access": True}
        assert isinstance(decision.evaluated_at, datetime)

    def test_valid_policy_decision_require_confirmation(self) -> None:
        """Test creating a valid REQUIRE_CONFIRMATION policy decision."""
        tool_call = ToolCall(
            tool_name="send_email",
            arguments={"recipient": "test@example.com"},
            risk_level=RiskLevel.DANGEROUS,
            execution_id="exec_123",
            user_id="user_123",
            session_id="session_456"
        )
        
        decision = PolicyDecision(
            decision=DecisionType.REQUIRE_CONFIRMATION,
            risk_score=0.9,
            reasoning="High-risk tool requires user confirmation",
            tool_call=tool_call
        )
        
        assert decision.decision == DecisionType.REQUIRE_CONFIRMATION
        assert decision.risk_score == 0.9

    def test_policy_decision_risk_score_validation(self) -> None:
        """Test risk score validation logic."""
        tool_call = ToolCall(
            tool_name="test_tool",
            arguments={},
            risk_level=RiskLevel.SAFE,
            execution_id="exec_123",
            user_id="user_123",
            session_id="session_456"
        )
        
        # High risk score with ALLOW should fail validation
        with pytest.raises(ValidationError) as exc_info:
            PolicyDecision(
                decision=DecisionType.ALLOW,
                risk_score=0.9,  # Too high for ALLOW
                reasoning="Test reasoning",
                tool_call=tool_call
            )
        
        assert "High risk scores should not result in ALLOW decisions" in str(exc_info.value)
        
        # Low risk score with DENY should fail validation
        with pytest.raises(ValidationError) as exc_info:
            PolicyDecision(
                decision=DecisionType.DENY,
                risk_score=0.1,  # Too low for DENY
                reasoning="Test reasoning",
                tool_call=tool_call
            )
        
        assert "Low risk scores should not result in DENY decisions" in str(exc_info.value)


class TestToolDefinition:
    """Test cases for ToolDefinition model."""

    def test_valid_tool_definition(self) -> None:
        """Test creating a valid ToolDefinition."""
        args_schema = {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state/country"
                }
            },
            "required": ["location"]
        }
        
        tool_def = ToolDefinition(
            name="get_weather",
            description="Get current weather conditions",
            risk_level=RiskLevel.SAFE,
            args_schema=args_schema,
            category="data",
            tags=["weather", "information"],
            timeout_seconds=10.0
        )
        
        assert tool_def.name == "get_weather"
        assert tool_def.description == "Get current weather conditions"
        assert tool_def.risk_level == RiskLevel.SAFE
        assert tool_def.args_schema == args_schema
        assert tool_def.category == "data"
        assert tool_def.tags == ["weather", "information"]
        assert tool_def.timeout_seconds == 10.0
        assert tool_def.max_retries == 3  # Default
        assert tool_def.requires_auth is False  # Default
        assert tool_def.total_calls == 0
        assert tool_def.success_rate == 1.0
        assert tool_def.version == "1.0.0"  # Default

    def test_tool_definition_validation_invalid_name(self) -> None:
        """Test that invalid tool names are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ToolDefinition(
                name="invalid-name!",  # Invalid characters
                description="Test description",
                risk_level=RiskLevel.SAFE,
                args_schema={}
            )
        
        assert "alphanumeric with underscores" in str(exc_info.value)