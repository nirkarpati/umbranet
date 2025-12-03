"""Unit tests for the tool registry and decorator system."""

import pytest
from typing import Dict, Any

from src.core.domain.tools import RiskLevel, ToolCall, ToolDefinition
from src.action_plane.tool_registry import governor_tool, ToolRegistry
from src.action_plane.tool_registry.decorator import get_tool_metadata, is_governor_tool


class TestGovernorToolDecorator:
    """Test cases for the @governor_tool decorator."""
    
    def setup_method(self):
        """Clear registry before each test."""
        ToolRegistry.clear_registry()
    
    def test_simple_tool_registration(self):
        """Test basic tool registration with decorator."""
        @governor_tool(
            name="test_tool",
            description="A test tool",
            risk_level=RiskLevel.SAFE
        )
        def test_function(param: str) -> str:
            return f"Hello {param}"
        
        # Check that tool is registered
        assert "test_tool" in ToolRegistry.list_tools()
        
        # Check tool metadata
        tool_def = ToolRegistry.get_tool_definition("test_tool")
        assert tool_def is not None
        assert tool_def.name == "test_tool"
        assert tool_def.description == "A test tool"
        assert tool_def.risk_level == RiskLevel.SAFE
        
        # Check function metadata
        assert is_governor_tool(test_function)
        metadata = get_tool_metadata(test_function)
        assert metadata is not None
        assert metadata.name == "test_tool"
    
    def test_tool_with_pydantic_schema(self):
        """Test tool registration with Pydantic schema."""
        from pydantic import BaseModel, Field
        
        class TestSchema(BaseModel):
            name: str = Field(..., description="Name parameter")
            age: int = Field(default=0, description="Age parameter")
        
        @governor_tool(
            name="schema_tool",
            description="Tool with schema",
            risk_level=RiskLevel.SENSITIVE,
            args_schema=TestSchema,
            category="test",
            tags=["testing", "schema"]
        )
        def schema_function(name: str, age: int = 0) -> dict:
            return {"name": name, "age": age}
        
        tool_def = ToolRegistry.get_tool_definition("schema_tool")
        assert tool_def is not None
        assert tool_def.category == "test"
        assert tool_def.tags == ["testing", "schema"]
        assert tool_def.risk_level == RiskLevel.SENSITIVE
        
        # Check that schema was set
        assert tool_def.args_schema is not None
    
    def test_auto_generated_schema(self):
        """Test automatic schema generation from function signature."""
        @governor_tool(
            name="auto_schema",
            description="Tool with auto-generated schema",
            risk_level=RiskLevel.SAFE
        )
        def auto_function(text: str, count: int, enabled: bool = True) -> str:
            return f"{text} {count} {enabled}"
        
        tool_def = ToolRegistry.get_tool_definition("auto_schema")
        assert tool_def is not None
        
        schema = tool_def.args_schema
        assert schema["type"] == "object"
        assert "text" in schema["properties"]
        assert "count" in schema["properties"]
        assert "enabled" in schema["properties"]
        
        # Check required fields (those without defaults)
        assert "text" in schema["required"]
        assert "count" in schema["required"]
        assert "enabled" not in schema["required"]  # Has default value
    
    def test_dangerous_tool_registration(self):
        """Test registration of dangerous tool with authentication."""
        @governor_tool(
            name="dangerous_tool",
            description="A dangerous operation",
            risk_level=RiskLevel.DANGEROUS,
            requires_auth=True,
            timeout_seconds=60.0,
            max_retries=1
        )
        def dangerous_function(action: str) -> dict:
            return {"action": action, "executed": True}
        
        tool_def = ToolRegistry.get_tool_definition("dangerous_tool")
        assert tool_def is not None
        assert tool_def.risk_level == RiskLevel.DANGEROUS
        assert tool_def.requires_auth is True
        assert tool_def.timeout_seconds == 60.0
        assert tool_def.max_retries == 1


class TestToolRegistry:
    """Test cases for the ToolRegistry class."""
    
    def setup_method(self):
        """Clear registry before each test."""
        ToolRegistry.clear_registry()
    
    def test_manual_tool_registration(self):
        """Test manual tool registration."""
        def test_func(x: int) -> int:
            return x * 2
        
        tool_def = ToolDefinition(
            name="manual_tool",
            description="Manually registered tool",
            risk_level=RiskLevel.SAFE,
            args_schema={"type": "object", "properties": {"x": {"type": "integer"}}}
        )
        
        ToolRegistry.register_tool("manual_tool", tool_def, test_func)
        
        assert "manual_tool" in ToolRegistry.list_tools()
        
        retrieved_def, retrieved_func = ToolRegistry.get_tool("manual_tool")
        assert retrieved_def == tool_def
        assert retrieved_func == test_func
    
    def test_tool_listing_and_filtering(self):
        """Test tool listing with various filters."""
        # Register multiple tools
        @governor_tool(name="safe1", description="Safe tool 1", risk_level=RiskLevel.SAFE, category="data", tags=["tag1"])
        def safe_tool1(): pass
        
        @governor_tool(name="safe2", description="Safe tool 2", risk_level=RiskLevel.SAFE, category="data", tags=["tag2"])  
        def safe_tool2(): pass
        
        @governor_tool(name="sensitive1", description="Sensitive tool", risk_level=RiskLevel.SENSITIVE, category="file", tags=["tag1"])
        def sensitive_tool(): pass
        
        @governor_tool(name="dangerous1", description="Dangerous tool", risk_level=RiskLevel.DANGEROUS, category="comm", tags=["tag3"])
        def dangerous_tool(): pass
        
        # Test listing all tools
        all_tools = ToolRegistry.list_tools()
        assert len(all_tools) == 4
        assert "safe1" in all_tools
        assert "dangerous1" in all_tools
        
        # Test filtering by risk level
        safe_tools = ToolRegistry.list_tools(risk_level=RiskLevel.SAFE)
        assert len(safe_tools) == 2
        assert "safe1" in safe_tools
        assert "safe2" in safe_tools
        
        # Test filtering by category
        data_tools = ToolRegistry.list_tools(category="data")
        assert len(data_tools) == 2
        assert "safe1" in data_tools
        assert "safe2" in data_tools
        
        # Test filtering by tag
        tag1_tools = ToolRegistry.list_tools(tag="tag1")
        assert len(tag1_tools) == 2
        assert "safe1" in tag1_tools
        assert "sensitive1" in tag1_tools
    
    def test_tool_validation(self):
        """Test tool call validation."""
        @governor_tool(
            name="validation_tool",
            description="Tool for validation testing",
            risk_level=RiskLevel.SAFE
        )
        def validation_func(required_param: str, optional_param: int = 10) -> dict:
            return {"required": required_param, "optional": optional_param}
        
        # Valid tool call
        valid_call = ToolCall(
            tool_name="validation_tool",
            arguments={"required_param": "test"},
            risk_level=RiskLevel.SAFE,
            execution_id="exec_123",
            user_id="user_123",
            session_id="session_456"
        )
        
        is_valid, error_msg = ToolRegistry.validate_tool_call(valid_call)
        assert is_valid is True
        assert error_msg == ""
        
        # Invalid tool call - missing required parameter
        invalid_call = ToolCall(
            tool_name="validation_tool",
            arguments={"optional_param": 5},  # Missing required_param
            risk_level=RiskLevel.SAFE,
            execution_id="exec_124",
            user_id="user_123",
            session_id="session_456"
        )
        
        is_valid, error_msg = ToolRegistry.validate_tool_call(invalid_call)
        assert is_valid is False
        assert "Missing required arguments" in error_msg
        
        # Tool not found
        unknown_call = ToolCall(
            tool_name="unknown_tool",
            arguments={},
            risk_level=RiskLevel.SAFE,
            execution_id="exec_125",
            user_id="user_123",
            session_id="session_456"
        )
        
        is_valid, error_msg = ToolRegistry.validate_tool_call(unknown_call)
        assert is_valid is False
        assert "not found" in error_msg
    
    def test_tool_execution(self):
        """Test tool execution through registry."""
        @governor_tool(
            name="execution_tool",
            description="Tool for execution testing",
            risk_level=RiskLevel.SAFE
        )
        def execution_func(x: int, y: int = 5) -> dict:
            return {"result": x + y, "inputs": [x, y]}
        
        tool_call = ToolCall(
            tool_name="execution_tool",
            arguments={"x": 10, "y": 3},
            risk_level=RiskLevel.SAFE,
            execution_id="exec_123",
            user_id="user_123",
            session_id="session_456"
        )
        
        result = ToolRegistry.execute_tool(tool_call)
        
        assert result["result"] == 13
        assert result["inputs"] == [10, 3]
    
    def test_tool_execution_errors(self):
        """Test tool execution error handling."""
        @governor_tool(
            name="error_tool",
            description="Tool that causes errors",
            risk_level=RiskLevel.SAFE
        )
        def error_func(should_fail: bool) -> dict:
            if should_fail:
                raise ValueError("Intentional error")
            return {"success": True}
        
        # Test successful execution
        success_call = ToolCall(
            tool_name="error_tool",
            arguments={"should_fail": False},
            risk_level=RiskLevel.SAFE,
            execution_id="exec_123",
            user_id="user_123",
            session_id="session_456"
        )
        
        result = ToolRegistry.execute_tool(success_call)
        assert result["success"] is True
        
        # Test failed execution
        error_call = ToolCall(
            tool_name="error_tool",
            arguments={"should_fail": True},
            risk_level=RiskLevel.SAFE,
            execution_id="exec_124",
            user_id="user_123",
            session_id="session_456"
        )
        
        with pytest.raises(ValueError, match="Intentional error"):
            ToolRegistry.execute_tool(error_call)
    
    def test_registry_statistics(self):
        """Test registry statistics collection."""
        # Register tools with different risk levels
        @governor_tool(name="stat_safe", description="Safe test tool", risk_level=RiskLevel.SAFE, category="data")
        def safe(): pass
        
        @governor_tool(name="stat_sensitive", description="Sensitive test tool", risk_level=RiskLevel.SENSITIVE, category="file")
        def sensitive(): pass
        
        @governor_tool(name="stat_dangerous", description="Dangerous test tool", risk_level=RiskLevel.DANGEROUS, category="data")
        def dangerous(): pass
        
        stats = ToolRegistry.get_registry_stats()
        
        assert stats["total_tools"] == 3
        assert stats["categories"] == 2  # "data" and "file"
        assert stats["risk_level_distribution"]["safe"] == 1
        assert stats["risk_level_distribution"]["sensitive"] == 1 
        assert stats["risk_level_distribution"]["dangerous"] == 1
        assert stats["tools_by_category"]["data"] == 2
        assert stats["tools_by_category"]["file"] == 1
    
    def test_get_tools_by_risk_level(self):
        """Test getting tools filtered by risk level."""
        @governor_tool(name="risk_safe", description="Safe risk test tool", risk_level=RiskLevel.SAFE)
        def safe(): pass
        
        @governor_tool(name="risk_dangerous", description="Dangerous risk test tool", risk_level=RiskLevel.DANGEROUS)
        def dangerous(): pass
        
        safe_tools = ToolRegistry.get_tools_by_risk_level(RiskLevel.SAFE)
        dangerous_tools = ToolRegistry.get_tools_by_risk_level(RiskLevel.DANGEROUS)
        sensitive_tools = ToolRegistry.get_tools_by_risk_level(RiskLevel.SENSITIVE)
        
        assert len(safe_tools) == 1
        assert len(dangerous_tools) == 1
        assert len(sensitive_tools) == 0
        
        assert safe_tools[0].name == "risk_safe"
        assert dangerous_tools[0].name == "risk_dangerous"
    
    def test_clear_registry(self):
        """Test registry clearing functionality."""
        @governor_tool(name="clear_test", description="Tool for clear testing", risk_level=RiskLevel.SAFE)
        def test_tool(): pass
        
        assert len(ToolRegistry.list_tools()) == 1
        
        ToolRegistry.clear_registry()
        
        assert len(ToolRegistry.list_tools()) == 0
        assert len(ToolRegistry.get_categories()) == 0
        assert len(ToolRegistry.get_tags()) == 0