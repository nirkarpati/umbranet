"""Memory-related domain models for the Governor system."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    """A single turn in a conversation (user message + assistant response)."""
    
    user_message: str = Field(..., description="User's message content")
    assistant_response: str = Field(..., description="Assistant's response content") 
    timestamp: datetime = Field(..., description="When this turn occurred")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for this turn"
    )


class ContextObject(BaseModel):
    """Standardized context representation returned by memory tiers."""
    
    summary: str | None = Field(
        None,
        description=(
            "Compressed summary of conversation history prior to recent messages"
        )
    )
    
    recent_messages: list[ConversationTurn] = Field(
        default_factory=list,
        description="Recent conversation turns within token budget"
    )
    
    token_count: int = Field(
        default=0,
        description="Estimated token count of the context"
    )
    
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this context was last updated"
    )
    
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context metadata"
    )


class MemoryEntry(BaseModel):
    """Base model for entries stored in memory systems."""
    
    user_id: str = Field(..., description="User this entry belongs to")
    content: str = Field(..., description="The memory content")
    timestamp: datetime = Field(..., description="When this was created/updated")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Entry-specific metadata"
    )


class SummarizationRequest(BaseModel):
    """Request for summarizing conversation history."""
    
    existing_summary: str | None = Field(
        None,
        description="Existing summary to merge with new messages"
    )
    
    messages_to_summarize: list[ConversationTurn] = Field(
        ...,
        description="New messages to incorporate into summary"
    )
    
    max_summary_tokens: int = Field(
        default=500,
        description="Maximum tokens for the resulting summary"
    )