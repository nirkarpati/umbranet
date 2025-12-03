"""Context Management System for the Governor.

This module provides dynamic context assembly capabilities for the Headless Governor,
enabling intelligent prompt construction from persona, environment, memory, and task data.
"""

from .assembler import ContextAssembler, ContextData

__all__ = ["ContextAssembler", "ContextData"]