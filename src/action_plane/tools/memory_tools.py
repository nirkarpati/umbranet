"""Memory management tools for the Governor system.

This module provides tools for the Governor Agent to actively manage
episodic memory, allowing explicit decisions about what to remember
and how to search past interactions.
"""

import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ...core.domain.tools import RiskLevel
from ...memory.tiers.episodic import EpisodicMemoryStore
from ..tool_registry.decorator import governor_tool

logger = logging.getLogger(__name__)


class SearchMemoryArgs(BaseModel):
    """Arguments for searching episodic memory."""
    
    query: str = Field(
        ...,
        description="The search text to find similar past interactions",
        min_length=1,
        max_length=500,
    )
    
    user_id: str = Field(
        ...,
        description="The ID of the user whose memories to search",
        min_length=1
    )
    
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return"
    )


class SaveMemoryArgs(BaseModel):
    """Arguments for saving episodic memory."""
    
    content: str = Field(
        ...,
        description=(
            "The core content to save "
            "(e.g., 'User mentioned they prefer Python over Java')"
        ),
        min_length=1,
        max_length=2000,
    )
    
    user_id: str = Field(
        ...,
        description="The user ID",
        min_length=1
    )
    
    importance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0.0 to 1.0 rating of memory significance"
    )
    
    tags: list[str] = Field(
        default_factory=list,
        description=(
            "Semantic tags for categorization "
            "(e.g., ['preference', 'coding'])"
        ),
    )
    
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional context"
    )
    
    occurred_at: str | None = Field(
        None,
        description=(
            "ISO 8601 date/time string (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS) "
            "indicating when the event actually happened. REQUIRED if the "
            "memory is about a past event."
        )
    )


@governor_tool(
    name="search_episodic_memory",
    description="Search through past conversation episodes using vector similarity",
    risk_level=RiskLevel.SAFE,
    args_schema=SearchMemoryArgs,
    category="memory",
    tags=["memory", "search", "episodic"],
    timeout_seconds=15.0
)
async def search_episodic_memory(
    query: str,
    user_id: str,
    limit: int = 5
) -> dict[str, Any]:
    """Search through past conversation episodes using vector similarity.
    
    This tool allows the Governor Agent to actively search for relevant
    past interactions to inform current responses.
    
    Args:
        query: The search text to find similar past interactions
        user_id: The ID of the user whose memories to search
        limit: Maximum number of results to return
        
    Returns:
        Dictionary containing search results with content, timestamp, and metadata
        
    Raises:
        Exception: If search fails
    """
    try:
        async with EpisodicMemoryStore() as memory_store:
            memories = await memory_store.recall(
                user_id=user_id,
                query_text=query,
                limit=limit
            )
            
            # Convert to JSON-serializable format
            results = []
            for memory in memories:
                results.append({
                    "content": memory.content,
                    "timestamp": (
                        memory.timestamp.isoformat() 
                        if memory.timestamp else None
                    ),
                    "metadata": memory.metadata,
                    "user_id": memory.user_id
                })
            
            logger.info(
                f"Retrieved {len(results)} memories for user {user_id} "
                f"with query: {query[:50]}..."
            )
            
            return {
                "status": "success",
                "results": results,
                "query": query,
                "total_found": len(results)
            }
            
    except Exception as e:
        logger.error(
            f"Failed to search episodic memory for user {user_id}: {str(e)}"
        )
        return {
            "status": "error",
            "error": str(e),
            "query": query,
            "results": []
        }


@governor_tool(
    name="save_episodic_memory",
    description="Save a significant interaction or observation to episodic memory",
    risk_level=RiskLevel.SENSITIVE,
    args_schema=SaveMemoryArgs,
    category="memory",
    tags=["memory", "save", "episodic"],
    timeout_seconds=10.0
)
async def save_episodic_memory(
    content: str,
    user_id: str,
    importance_score: float,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    occurred_at: str | None = None
) -> dict[str, Any]:
    """Save a significant interaction or observation to episodic memory.
    
    This tool allows the Governor Agent to actively curate what gets
    remembered, providing explicit control over memory formation.
    
    TEMPORAL RESOLUTION - CRITICAL:
    If the user mentions relative time (e.g., 'yesterday', 'last month', 
    'a few days ago'), you MUST calculate the absolute date based on the 
    current time and follow these steps:
    
    1) Rewrite the content to use the absolute date. For example:
       - Change 'went surfing a month ago' to 'went surfing in November 2024'
       - Change 'met John yesterday' to 'met John on December 5, 2024'
    
    2) Pass the calculated date in occurred_at parameter as ISO format (YYYY-MM-DD).
    
    3) The system will automatically enhance the content with temporal context 
       for better vector search and memory grounding.
    
    Examples:
    - User: "I went hiking last weekend"
    - Content: "User went hiking on December 1, 2024" 
    - occurred_at: "2024-12-01"
    
    Args:
        content: The core content to save
        user_id: The user ID
        importance_score: 0.0 to 1.0 rating of memory significance
        tags: Semantic tags for categorization
        metadata: Any additional context
        occurred_at: ISO 8601 date/time string for when the event happened
        
    Returns:
        Dictionary containing save status and episode ID
        
    Raises:
        Exception: If save fails
    """
    try:
        # Parse occurred_at timestamp if provided
        occurrence_timestamp = None
        if occurred_at:
            try:
                # Support both date-only (YYYY-MM-DD) and datetime formats
                if 'T' in occurred_at or ' ' in occurred_at:
                    occurrence_timestamp = datetime.fromisoformat(
                        occurred_at.replace('Z', '+00:00')
                    )
                else:
                    # Date-only format, set to start of day
                    occurrence_timestamp = datetime.fromisoformat(
                        f"{occurred_at}T00:00:00"
                    )
            except ValueError as e:
                logger.warning(
                    f"Invalid occurred_at format '{occurred_at}': {e}. "
                    "Using current time instead."
                )
        
        # Prepare metadata with tags and importance
        full_metadata = {
            "importance_score": importance_score,
            "tags": tags or [],
            "source": "governor_agent",
            **(metadata or {})
        }
        
        async with EpisodicMemoryStore() as memory_store:
            episode_id = await memory_store.log_episode(
                user_id=user_id,
                content=content,
                metadata=full_metadata,
                timestamp=occurrence_timestamp
            )
            
            temporal_info = (
                f" occurred at {occurred_at}" if occurred_at else " (current time)"
            )
            logger.info(
                f"Saved episode {episode_id} for user {user_id} "
                f"with importance {importance_score}{temporal_info}"
            )
            
            return {
                "status": "success",
                "episode_id": episode_id,
                "user_id": user_id,
                "importance_score": importance_score,
                "tags": tags or []
            }
            
    except Exception as e:
        logger.error(
            f"Failed to save episodic memory for user {user_id}: {str(e)}"
        )
        return {
            "status": "error",
            "error": str(e),
            "user_id": user_id
        }