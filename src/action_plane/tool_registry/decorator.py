"""Governor tool decorator for registering tools with the system.

This module provides the @governor_tool decorator that enables secure
tool registration with automatic validation and risk assessment.
"""

import functools
import inspect
from typing import Any, Callable, Type, TypeVar

from pydantic import BaseModel, create_model

from ...core.domain.tools import RiskLevel, ToolDefinition

F = TypeVar('F', bound=Callable[..., Any])


def governor_tool(
    name: str,
    description: str,
    risk_level: RiskLevel,
    args_schema: Type[BaseModel] | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
    timeout_seconds: float = 30.0,
    max_retries: int = 3,
    requires_auth: bool = False
) -> Callable[[F], F]:
    """Decorator to register a function as a Governor tool.
    
    This decorator registers a function with the tool registry, making it
    available for execution through the Governor state machine with 
    automatic risk assessment and validation.
    
    Args:
        name: Unique tool name for identification
        description: Human-readable description for LLM context
        risk_level: Security risk level (SAFE, SENSITIVE, DANGEROUS)
        args_schema: Optional Pydantic schema for argument validation
        category: Tool category for organization
        tags: Tags for discovery and filtering
        timeout_seconds: Maximum execution time
        max_retries: Maximum retry attempts on failure
        requires_auth: Whether tool requires special authentication
        
    Returns:
        Decorated function registered with the tool registry
        
    Example:
        @governor_tool(
            name="get_weather",
            description="Get current weather for a location",
            risk_level=RiskLevel.SAFE,
            args_schema=WeatherSchema,
            category="data",
            tags=["weather", "information"]
        )
        def get_weather(location: str) -> dict:
            return {"weather": "sunny", "temp": 72, "location": location}
    """
    def decorator(func: F) -> F:
        # Import here to avoid circular imports
        from .registry import ToolRegistry
        
        # Generate schema from function signature if not provided
        if args_schema is None:
            generated_schema = _generate_schema_from_function(func)
        else:
            generated_schema = args_schema.model_json_schema()
        
        # Create tool definition
        tool_def = ToolDefinition(
            name=name,
            description=description,
            risk_level=risk_level,
            args_schema=generated_schema,
            category=category,
            tags=tags or [],
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            requires_auth=requires_auth
        )
        
        # Create wrapper function with metadata
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)
        
        # Attach metadata to the function
        wrapper._governor_tool_definition = tool_def  # type: ignore
        wrapper._governor_tool_function = func  # type: ignore
        wrapper._governor_tool_name = name  # type: ignore
        
        # Register with global registry
        ToolRegistry.register_tool(name, tool_def, wrapper)
        
        return wrapper  # type: ignore
    
    return decorator


def _generate_schema_from_function(func: Callable[..., Any]) -> dict[str, Any]:
    """Generate Pydantic schema from function signature.
    
    Args:
        func: Function to analyze
        
    Returns:
        JSON schema dictionary for function arguments
    """
    sig = inspect.signature(func)
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        # Skip self and cls parameters
        if param_name in ('self', 'cls'):
            continue
            
        # Determine parameter type
        param_type = param.annotation
        if param_type == inspect.Parameter.empty:
            param_type = str  # Default to string
        
        # Convert Python types to JSON Schema types
        json_type = _python_type_to_json_type(param_type)
        
        properties[param_name] = {
            "type": json_type,
            "description": f"Parameter {param_name}"
        }
        
        # Mark as required if no default value
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
    
    return {
        "type": "object",
        "properties": properties,
        "required": required
    }


def _python_type_to_json_type(python_type: Type[Any]) -> str:
    """Convert Python type to JSON Schema type.
    
    Args:
        python_type: Python type to convert
        
    Returns:
        JSON Schema type string
    """
    type_mapping = {
        str: "string",
        int: "integer", 
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object"
    }
    
    # Handle typing module types
    if hasattr(python_type, '__origin__'):
        origin = python_type.__origin__
        if origin in type_mapping:
            return type_mapping[origin]
    
    # Handle basic types
    if python_type in type_mapping:
        return type_mapping[python_type]
    
    # Default to string for unknown types
    return "string"


def get_tool_metadata(func: Callable[..., Any]) -> ToolDefinition | None:
    """Get tool metadata from a decorated function.
    
    Args:
        func: Function to check for tool metadata
        
    Returns:
        Tool definition if function is a registered tool, None otherwise
    """
    return getattr(func, '_governor_tool_definition', None)


def is_governor_tool(func: Callable[..., Any]) -> bool:
    """Check if a function is a registered Governor tool.
    
    Args:
        func: Function to check
        
    Returns:
        True if function is a registered tool
    """
    return hasattr(func, '_governor_tool_definition')