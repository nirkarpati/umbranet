"""Tool Registry - Secure tool registration and discovery.

This module provides the @governor_tool decorator for registering tools
with the Governor system and handles tool discovery and validation.
"""

from .decorator import governor_tool
from .registry import ToolRegistry

__all__ = ["governor_tool", "ToolRegistry"]