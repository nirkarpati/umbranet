"""Domain models for the Governor system.

This module contains all the core data structures and domain models
that represent the fundamental concepts in the Governor system.
"""

# Event models - Input/Output data structures
from .events import (
    ChannelType,
    EventMetadata,
    GovernorEvent,
    GovernorResponse,
    MessageType,
    ResponseMetadata,
    ResponseType,
)

# State models - Conversation flow and state management
from .state import (
    ActiveTask,
    ConversationTurn,
    GovernorState,
    StateNode,
)

# Tool models - Tool execution and security
from .tools import (
    DecisionType,
    PolicyDecision,
    RiskLevel,
    ToolCall,
    ToolDefinition,
    ToolStatus,
)

# Export all domain models for easy importing
__all__ = [
    # Event models
    "GovernorEvent",
    "GovernorResponse", 
    "ChannelType",
    "MessageType",
    "ResponseType",
    "EventMetadata",
    "ResponseMetadata",
    
    # State models
    "GovernorState",
    "StateNode",
    "ConversationTurn",
    "ActiveTask",
    
    # Tool models
    "ToolCall",
    "ToolDefinition",
    "PolicyDecision",
    "RiskLevel",
    "DecisionType",
    "ToolStatus",
]

# Version information
__version__ = "1.0.0"
__author__ = "Governor Team"
__description__ = "Domain models for the Headless Governor System"