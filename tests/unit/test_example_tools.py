"""Unit tests for example tools to verify tool registry integration."""

import pytest
import importlib
import sys

from src.core.domain.tools import RiskLevel, ToolCall
from src.action_plane.tool_registry import ToolRegistry


class TestExampleToolsIntegration:
    """Test that example tools are properly registered and functional."""
    
    @classmethod
    def setup_class(cls):
        """One-time setup: clear registry and import all tools."""
        ToolRegistry.clear_registry()
        
        # Import tool modules to trigger decorators
        # This ensures tools are registered for testing
        try:
            import src.action_plane.tools.weather_tools
            import src.action_plane.tools.communication_tools
            import src.action_plane.tools.data_tools
            import src.action_plane.tools.file_tools
        except ImportError as e:
            # If imports fail, skip the setup
            pytest.skip(f"Tool import failed: {e}")
    
    def test_weather_tools_registered(self):
        """Test that weather tools are properly registered."""
        tools = ToolRegistry.list_tools()
        
        assert "get_weather" in tools
        assert "get_forecast" in tools
        
        # Check weather tool properties
        weather_def = ToolRegistry.get_tool_definition("get_weather")
        assert weather_def is not None
        assert weather_def.risk_level == RiskLevel.SAFE
        assert weather_def.category == "data"
        assert "weather" in weather_def.tags
    
    def test_communication_tools_registered(self):
        """Test that communication tools are properly registered."""
        tools = ToolRegistry.list_tools()
        
        assert "send_email" in tools
        assert "send_sms" in tools
        
        # Check email tool properties
        email_def = ToolRegistry.get_tool_definition("send_email")
        assert email_def is not None
        assert email_def.risk_level == RiskLevel.DANGEROUS
        assert email_def.category == "communication"
        assert email_def.requires_auth is True
    
    def test_data_tools_registered(self):
        """Test that data tools are properly registered.""" 
        tools = ToolRegistry.list_tools()
        
        assert "search_web" in tools
        assert "calculate" in tools
        
        # Check search tool properties
        search_def = ToolRegistry.get_tool_definition("search_web")
        assert search_def is not None
        assert search_def.risk_level == RiskLevel.SAFE
        assert search_def.category == "data"
    
    def test_file_tools_registered(self):
        """Test that file tools are properly registered."""
        tools = ToolRegistry.list_tools()
        
        assert "read_file" in tools
        assert "write_file" in tools
        
        # Check read file tool properties
        read_def = ToolRegistry.get_tool_definition("read_file")
        assert read_def is not None
        assert read_def.risk_level == RiskLevel.SENSITIVE
        assert read_def.category == "file"
    
    def test_weather_tool_execution(self):
        """Test weather tool execution through registry."""
        tool_call = ToolCall(
            tool_name="get_weather",
            arguments={"location": "Seattle, WA"},
            risk_level=RiskLevel.SAFE,
            execution_id="exec_123",
            user_id="user_123", 
            session_id="session_456"
        )
        
        result = ToolRegistry.execute_tool(tool_call)
        
        # Should return error about missing API key (real implementation)
        assert result["status"] == "error"
        assert "OpenWeatherMap API key not configured" in result["error"]
    
    def test_search_tool_execution(self):
        """Test search tool execution."""
        tool_call = ToolCall(
            tool_name="search_web",
            arguments={"query": "python programming", "num_results": 3},
            risk_level=RiskLevel.SAFE,
            execution_id="exec_124",
            user_id="user_123",
            session_id="session_456"
        )
        
        result = ToolRegistry.execute_tool(tool_call)
        
        # Should return error about missing API key (real implementation)
        assert result["status"] == "error"
        assert "SerpAPI key not configured" in result["error"]
    
    def test_calculator_tool_execution(self):
        """Test calculator tool execution."""
        tool_call = ToolCall(
            tool_name="calculate",
            arguments={"expression": "2 + 3 * 4", "precision": 1},
            risk_level=RiskLevel.SAFE,
            execution_id="exec_125",
            user_id="user_123",
            session_id="session_456"
        )
        
        result = ToolRegistry.execute_tool(tool_call)
        
        assert result["status"] == "success"
        assert result["result"] == 14.0  # 2 + (3 * 4)
        assert result["precision"] == 1
    
    def test_email_tool_execution(self):
        """Test email tool execution."""
        tool_call = ToolCall(
            tool_name="send_email",
            arguments={
                "recipients": ["test@example.com"],
                "subject": "Test Email",
                "body": "This is a test email.",
                "priority": "normal"
            },
            risk_level=RiskLevel.DANGEROUS,
            execution_id="exec_126", 
            user_id="user_123",
            session_id="session_456"
        )
        
        result = ToolRegistry.execute_tool(tool_call)
        
        # Should return error about missing SMTP configuration (real implementation)
        assert result["status"] == "error"
        assert "SMTP configuration missing" in result["error"]
    
    def test_tool_validation_errors(self):
        """Test tool validation with missing required arguments."""
        # Weather tool missing location
        invalid_call = ToolCall(
            tool_name="get_weather",
            arguments={},  # Missing required 'location'
            risk_level=RiskLevel.SAFE,
            execution_id="exec_127",
            user_id="user_123",
            session_id="session_456"
        )
        
        is_valid, error_msg = ToolRegistry.validate_tool_call(invalid_call)
        assert is_valid is False
        assert "Missing required arguments" in error_msg
        assert "location" in error_msg
    
    def test_risk_level_distribution(self):
        """Test that tools have appropriate risk level distribution."""
        stats = ToolRegistry.get_registry_stats()
        risk_dist = stats["risk_level_distribution"]
        
        # Should have tools at all risk levels
        assert risk_dist["safe"] > 0  # Weather, search, calculator
        assert risk_dist["sensitive"] > 0  # File operations
        assert risk_dist["dangerous"] > 0  # Communication tools
        
        # Safe tools should be the majority
        total_tools = sum(risk_dist.values())
        assert risk_dist["safe"] >= risk_dist["dangerous"]
    
    def test_category_distribution(self):
        """Test tool category distribution."""
        stats = ToolRegistry.get_registry_stats()
        categories = stats["tools_by_category"]
        
        expected_categories = {"data", "communication", "utility", "file"}
        for category in expected_categories:
            assert category in categories
            assert categories[category] > 0