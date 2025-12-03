"""Tool models for the Governor system.

These models define the data structures for tool execution,
risk assessment, and security policy decisions.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, validator


class RiskLevel(str, Enum):
    """Risk levels for tool execution.
    
    These levels determine the security policy that applies
    to each tool call in the system.
    """
    
    SAFE = "safe"           # Auto-approve, minimal logging
    SENSITIVE = "sensitive" # Auto-approve, enhanced logging
    DANGEROUS = "dangerous" # Require user confirmation


class DecisionType(str, Enum):
    """Policy engine decision types."""
    
    ALLOW = "allow"                    # Tool execution approved
    DENY = "deny"                     # Tool execution denied
    REQUIRE_CONFIRMATION = "require_confirmation"  # User confirmation needed


class ToolStatus(str, Enum):
    """Status of tool execution."""
    
    PENDING = "pending"       # Queued for execution
    EXECUTING = "executing"   # Currently running
    COMPLETED = "completed"   # Successfully completed
    FAILED = "failed"        # Execution failed
    CANCELLED = "cancelled"   # Cancelled by user or system
    TIMEOUT = "timeout"      # Execution timed out


class ToolCall(BaseModel):
    """Structured tool execution request.
    
    This represents a request to execute a specific tool with
    given arguments, along with security and execution metadata.
    """
    
    # Core tool information
    tool_name: str = Field(
        ...,
        description="Name of the tool to execute",
        min_length=1,
        max_length=100
    )
    
    arguments: dict[str, Any] = Field(
        ...,
        description="Arguments to pass to the tool"
    )
    
    # Security and policy
    risk_level: RiskLevel = Field(
        ...,
        description="Risk level assessment for this tool call"
    )
    
    requires_confirmation: bool = Field(
        default=False,
        description="Whether this tool requires user confirmation"
    )
    
    # Execution metadata
    execution_id: str = Field(
        ...,
        description="Unique identifier for this tool execution",
        min_length=1
    )
    
    status: ToolStatus = Field(
        default=ToolStatus.PENDING,
        description="Current execution status"
    )
    
    # Context and tracking
    user_id: str = Field(
        ...,
        description="User who requested this tool execution"
    )
    
    session_id: str = Field(
        ...,
        description="Session context for this tool call"
    )
    
    # Timing information
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this tool call was created"
    )
    
    started_at: datetime | None = Field(
        None,
        description="When tool execution began"
    )
    
    completed_at: datetime | None = Field(
        None,
        description="When tool execution finished"
    )
    
    # Results and errors
    result: dict[str, Any] | None = Field(
        None,
        description="Tool execution result"
    )
    
    error_message: str | None = Field(
        None,
        description="Error message if execution failed"
    )
    
    execution_time_ms: float | None = Field(
        None,
        description="How long the tool took to execute"
    )
    
    # Additional metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional tool-specific metadata"
    )
    
    retry_count: int = Field(
        default=0,
        description="Number of retry attempts"
    )
    
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts"
    )
    
    timeout_seconds: float = Field(
        default=30.0,
        description="Maximum execution time in seconds"
    )
    
    @validator('tool_name')
    def validate_tool_name(cls, v: str) -> str:
        """Validate tool name format."""
        if not v.replace('_', '').isalnum():
            raise ValueError("Tool name must be alphanumeric with underscores")
        return v
    
    @validator('timeout_seconds')
    def validate_timeout(cls, v: float) -> float:
        """Validate timeout is reasonable."""
        if v <= 0 or v > 300:  # Max 5 minutes
            raise ValueError("Timeout must be between 0 and 300 seconds")
        return v
    
    def start_execution(self) -> None:
        """Mark tool execution as started."""
        self.status = ToolStatus.EXECUTING
        self.started_at = datetime.utcnow()
    
    def complete_execution(self, result: dict[str, Any]) -> None:
        """Mark tool execution as completed successfully."""
        self.status = ToolStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.result = result
        
        if self.started_at:
            self.execution_time_ms = (
                (self.completed_at - self.started_at).total_seconds() * 1000
            )
    
    def fail_execution(self, error_message: str) -> None:
        """Mark tool execution as failed."""
        self.status = ToolStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        
        if self.started_at:
            self.execution_time_ms = (
                (self.completed_at - self.started_at).total_seconds() * 1000
            )
    
    def cancel_execution(self) -> None:
        """Cancel tool execution."""
        self.status = ToolStatus.CANCELLED
        self.completed_at = datetime.utcnow()
    
    def increment_retry(self) -> bool:
        """Increment retry count and return whether retry is allowed."""
        self.retry_count += 1
        return self.retry_count <= self.max_retries
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }
        json_schema_extra = {
            "example": {
                "tool_name": "get_weather",
                "arguments": {"location": "Seattle, WA"},
                "risk_level": "safe",
                "requires_confirmation": False,
                "execution_id": "tool_exec_123456",
                "status": "pending",
                "user_id": "user_123",
                "session_id": "telegram_user_123_20241127",
                "timeout_seconds": 10.0,
                "max_retries": 3
            }
        }


class PolicyDecision(BaseModel):
    """Result of policy engine assessment.
    
    This represents the security decision made by the policy engine
    about whether a tool call should be allowed to execute.
    """
    
    # Core decision
    decision: DecisionType = Field(
        ...,
        description="The policy decision (allow/deny/require_confirmation)"
    )
    
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Computed risk score (0.0 = no risk, 1.0 = maximum risk)"
    )
    
    reasoning: str = Field(
        ...,
        description="Human-readable explanation of the decision",
        min_length=1,
        max_length=1000
    )
    
    # Context
    tool_call: ToolCall = Field(
        ...,
        description="The tool call being evaluated"
    )
    
    policy_version: str = Field(
        default="v1.0",
        description="Version of the policy engine that made this decision"
    )
    
    # Timing
    evaluated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this decision was made"
    )
    
    evaluation_time_ms: float | None = Field(
        None,
        description="How long the evaluation took"
    )
    
    # Additional context
    applied_rules: list[str] = Field(
        default_factory=list,
        description="List of policy rules that influenced this decision"
    )
    
    user_permissions: dict[str, Any] = Field(
        default_factory=dict,
        description="Relevant user permissions considered"
    )
    
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional policy-specific metadata"
    )
    
    @validator('risk_score')
    def validate_risk_score(cls, v: float, values: dict[str, Any]) -> float:
        """Ensure risk score aligns with decision type."""
        decision = values.get('decision')
        
        if decision == DecisionType.ALLOW and v > 0.7:
            raise ValueError("High risk scores should not result in ALLOW decisions")
        elif decision == DecisionType.DENY and v < 0.3:
            raise ValueError("Low risk scores should not result in DENY decisions")
        
        return v
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }
        json_schema_extra = {
            "example": {
                "decision": "allow",
                "risk_score": 0.2,
                "reasoning": "Safe tool with validated arguments and user permissions",
                "policy_version": "v1.0",
                "applied_rules": ["safe_tool_auto_approve", "argument_validation_passed"],
                "user_permissions": {"weather_access": True},
                "evaluation_time_ms": 15.7
            }
        }


class ToolDefinition(BaseModel):
    """Definition of a tool that can be registered with the system.
    
    This represents the metadata and schema for a tool that can be
    executed by the Governor system.
    """
    
    name: str = Field(
        ...,
        description="Unique name for this tool",
        min_length=1,
        max_length=100
    )
    
    description: str = Field(
        ...,
        description="Human-readable description for the LLM",
        min_length=1,
        max_length=500
    )
    
    risk_level: RiskLevel = Field(
        ...,
        description="Default risk level for this tool"
    )
    
    args_schema: dict[str, Any] = Field(
        ...,
        description="Pydantic schema for tool arguments"
    )
    
    # Optional metadata
    category: str | None = Field(
        None,
        description="Tool category (communication, data, automation, etc)"
    )
    
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for tool discovery and organization"
    )
    
    timeout_seconds: float = Field(
        default=30.0,
        description="Default timeout for this tool"
    )
    
    max_retries: int = Field(
        default=3,
        description="Default max retries for this tool"
    )
    
    requires_auth: bool = Field(
        default=False,
        description="Whether this tool requires special authentication"
    )
    
    # Usage statistics
    total_calls: int = Field(
        default=0,
        description="Total number of times this tool has been called"
    )
    
    success_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Success rate for this tool"
    )
    
    average_execution_time_ms: float | None = Field(
        None,
        description="Average execution time for this tool"
    )
    
    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this tool was registered"
    )
    
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this tool was last updated"
    )
    
    version: str = Field(
        default="1.0.0",
        description="Tool version"
    )
    
    @validator('name')
    def validate_name(cls, v: str) -> str:
        """Validate tool name format."""
        if not v.replace('_', '').isalnum():
            raise ValueError("Tool name must be alphanumeric with underscores")
        return v
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }
        json_schema_extra = {
            "example": {
                "name": "get_weather",
                "description": "Get current weather conditions for a specified location",
                "risk_level": "safe",
                "args_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and state/country"
                        }
                    },
                    "required": ["location"]
                },
                "category": "data",
                "tags": ["weather", "information", "api"],
                "timeout_seconds": 10.0,
                "max_retries": 2,
                "version": "1.0.0"
            }
        }