"""Tool Registry implementation for managing Governor tools.

This module provides the central registry for all tools available
to the Governor system with discovery and validation capabilities.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from pydantic import ValidationError

from ...core.domain.tools import RiskLevel, ToolDefinition, ToolCall

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for all Governor tools.
    
    This class manages tool registration, discovery, and validation
    for the Governor system. It provides a secure interface for
    tool execution with built-in argument validation.
    """
    
    _tools: Dict[str, ToolDefinition] = {}
    _functions: Dict[str, Callable[..., Any]] = {}
    _categories: Dict[str, List[str]] = {}
    _tags: Dict[str, List[str]] = {}
    
    @classmethod
    def register_tool(
        self,
        name: str,
        definition: ToolDefinition,
        function: Callable[..., Any]
    ) -> None:
        """Register a tool with the registry.
        
        Args:
            name: Unique tool name
            definition: Tool definition with metadata
            function: Actual function to execute
            
        Raises:
            ValueError: If tool name already exists
        """
        if name in self._tools:
            logger.warning(f"Tool '{name}' is being re-registered")
        
        self._tools[name] = definition
        self._functions[name] = function
        
        # Index by category
        if definition.category:
            if definition.category not in self._categories:
                self._categories[definition.category] = []
            self._categories[definition.category].append(name)
        
        # Index by tags
        for tag in definition.tags:
            if tag not in self._tags:
                self._tags[tag] = []
            self._tags[tag].append(name)
        
        logger.info(f"Registered tool '{name}' with risk level {definition.risk_level.value}")
    
    @classmethod
    def get_tool(self, name: str) -> tuple[ToolDefinition, Callable[..., Any]] | None:
        """Get tool definition and function by name.
        
        Args:
            name: Tool name to retrieve
            
        Returns:
            Tuple of (definition, function) or None if not found
        """
        if name not in self._tools:
            return None
        
        return self._tools[name], self._functions[name]
    
    @classmethod
    def get_tool_definition(self, name: str) -> ToolDefinition | None:
        """Get tool definition by name.
        
        Args:
            name: Tool name to retrieve
            
        Returns:
            Tool definition or None if not found
        """
        return self._tools.get(name)
    
    @classmethod
    def get_tool_function(self, name: str) -> Callable[..., Any] | None:
        """Get tool function by name.
        
        Args:
            name: Tool name to retrieve
            
        Returns:
            Tool function or None if not found
        """
        return self._functions.get(name)
    
    @classmethod
    def list_tools(self, 
                   category: str | None = None,
                   tag: str | None = None,
                   risk_level: RiskLevel | None = None) -> List[str]:
        """List available tools with optional filtering.
        
        Args:
            category: Filter by category
            tag: Filter by tag
            risk_level: Filter by risk level
            
        Returns:
            List of tool names matching filters
        """
        tools = list(self._tools.keys())
        
        if category:
            tools = [name for name in tools if name in self._categories.get(category, [])]
        
        if tag:
            tools = [name for name in tools if name in self._tags.get(tag, [])]
        
        if risk_level:
            tools = [
                name for name in tools 
                if self._tools[name].risk_level == risk_level
            ]
        
        return sorted(tools)
    
    @classmethod
    def get_categories(self) -> List[str]:
        """Get list of all tool categories.
        
        Returns:
            List of category names
        """
        return sorted(self._categories.keys())
    
    @classmethod
    def get_tags(self) -> List[str]:
        """Get list of all tool tags.
        
        Returns:
            List of tag names
        """
        return sorted(self._tags.keys())
    
    @classmethod
    def validate_tool_call(self, tool_call: ToolCall) -> tuple[bool, str]:
        """Validate a tool call against registered tool schema.
        
        Args:
            tool_call: Tool call to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        tool_def = self.get_tool_definition(tool_call.tool_name)
        if not tool_def:
            return False, f"Tool '{tool_call.tool_name}' not found"
        
        # Validate arguments against schema
        try:
            # For now, basic validation - could be enhanced with jsonschema
            required_fields = tool_def.args_schema.get("required", [])
            provided_fields = set(tool_call.arguments.keys())
            required_set = set(required_fields)
            
            missing_fields = required_set - provided_fields
            if missing_fields:
                return False, f"Missing required arguments: {', '.join(missing_fields)}"
            
            return True, ""
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    @classmethod
    def execute_tool(self, tool_call: ToolCall) -> dict[str, Any]:
        """Execute a validated tool call.
        
        Args:
            tool_call: Tool call to execute
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If tool not found or validation fails
            Exception: If tool execution fails
        """
        # Get tool function
        function = self.get_tool_function(tool_call.tool_name)
        if not function:
            raise ValueError(f"Tool '{tool_call.tool_name}' not found")
        
        # Validate arguments
        is_valid, error_msg = self.validate_tool_call(tool_call)
        if not is_valid:
            raise ValueError(f"Tool validation failed: {error_msg}")
        
        try:
            # Execute the tool function
            result = function(**tool_call.arguments)
            
            # Ensure result is serializable
            if not isinstance(result, dict):
                result = {"result": result}
            
            return result
            
        except Exception as e:
            logger.error(f"Tool execution failed for '{tool_call.tool_name}': {str(e)}")
            raise
    
    @classmethod
    def get_tools_by_risk_level(self, risk_level: RiskLevel) -> List[ToolDefinition]:
        """Get all tools with specified risk level.
        
        Args:
            risk_level: Risk level to filter by
            
        Returns:
            List of tool definitions with matching risk level
        """
        return [
            tool_def for tool_def in self._tools.values()
            if tool_def.risk_level == risk_level
        ]
    
    @classmethod
    def get_registry_stats(self) -> dict[str, Any]:
        """Get registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        risk_counts = {}
        for risk_level in RiskLevel:
            risk_counts[risk_level.value] = len(self.get_tools_by_risk_level(risk_level))
        
        return {
            "total_tools": len(self._tools),
            "categories": len(self._categories),
            "tags": len(self._tags),
            "risk_level_distribution": risk_counts,
            "tools_by_category": {
                cat: len(tools) for cat, tools in self._categories.items()
            }
        }
    
    @classmethod
    def clear_registry(self) -> None:
        """Clear all registered tools.
        
        Warning: This is primarily for testing purposes.
        """
        self._tools.clear()
        self._functions.clear()
        self._categories.clear()
        self._tags.clear()
        logger.info("Tool registry cleared")
    
    @classmethod
    def discover_tools_from_module(self, module: Any) -> List[str]:
        """Discover and register tools from a module.
        
        Args:
            module: Python module to scan for tools
            
        Returns:
            List of discovered tool names
        """
        from .decorator import is_governor_tool, get_tool_metadata
        
        discovered = []
        
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if callable(attr) and is_governor_tool(attr):
                tool_def = get_tool_metadata(attr)
                if tool_def:
                    self.register_tool(tool_def.name, tool_def, attr)
                    discovered.append(tool_def.name)
        
        logger.info(f"Discovered {len(discovered)} tools from module {module.__name__}")
        return discovered