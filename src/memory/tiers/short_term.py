"""Short-term memory implementation using Redis with token-managed rolling summary.

This module implements Tier 1 of the RAG++ memory hierarchy, providing:
- Token-budget managed conversation buffers
- Rolling summary compression when buffer overflows
- Atomic operations with distributed locking
- Session-based context retrieval
"""

import json
import logging
from datetime import datetime
from typing import Any

import redis.asyncio as redis
from redis.asyncio.lock import Lock

from ...core.config import settings
from ...core.domain.memory import ContextObject, ConversationTurn, SummarizationRequest
from ...core.utils.tokens import token_counter
from ..services.summarizer import get_summarizer

logger = logging.getLogger(__name__)


class RedisConnectionError(Exception):
    """Exception raised for Redis connection issues."""
    pass


class ShortTermMemoryClient:
    """Redis-based short-term memory with token-managed rolling summaries.
    
    Key Schema:
        session:{user_id}:v1 - Hash containing:
            - buffer: JSON list of ConversationTurn objects
            - summary: String containing compressed conversation history
            - token_count: Integer cache of current buffer token count
            - last_updated: ISO timestamp
    """
    
    def __init__(self, max_token_budget: int = 2000):
        """Initialize the Redis client.
        
        Args:
            max_token_budget: Maximum tokens allowed in buffer before summarization
        """
        self.max_token_budget = max_token_budget
        self.redis_client: redis.Redis | None = None
        self._lock_timeout = 30  # seconds
    
    async def __aenter__(self) -> "ShortTermMemoryClient":
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self) -> None:
        """Establish Redis connection."""
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Connected to Redis for short-term memory")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise RedisConnectionError(f"Redis connection failed: {str(e)}") from e
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Disconnected from Redis")
    
    def _get_session_key(self, user_id: str) -> str:
        """Get Redis key for user session."""
        return f"session:{user_id}:v1"
    
    def _get_lock_key(self, user_id: str) -> str:
        """Get Redis lock key for user session."""
        return f"lock:{user_id}:session"
    
    def _serialize_turn(self, turn: ConversationTurn) -> dict:
        """Serialize conversation turn for Redis storage."""
        return {
            "user_message": turn.user_message,
            "assistant_response": turn.assistant_response,
            "timestamp": turn.timestamp.isoformat(),
            "metadata": turn.metadata
        }
    
    def _deserialize_turn(self, data: dict) -> ConversationTurn:
        """Deserialize conversation turn from Redis data."""
        return ConversationTurn(
            user_message=data["user_message"],
            assistant_response=data["assistant_response"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )
    
    async def _get_session_data(self, user_id: str) -> dict[str, str]:
        """Get all session data from Redis."""
        if not self.redis_client:
            raise RedisConnectionError("Redis client not connected")
        
        session_key = self._get_session_key(user_id)
        data = await self.redis_client.hgetall(session_key)
        
        if not data:
            # Initialize empty session
            return {
                "buffer": "[]",
                "summary": "",
                "token_count": "0",
                "last_updated": datetime.utcnow().isoformat()
            }
        
        return data
    
    async def _save_session_data(self, user_id: str, data: dict[str, str]) -> None:
        """Save session data to Redis."""
        if not self.redis_client:
            raise RedisConnectionError("Redis client not connected")
        
        session_key = self._get_session_key(user_id)
        data["last_updated"] = datetime.utcnow().isoformat()
        
        await self.redis_client.hset(session_key, mapping=data)
        # Set expiration: 7 days of inactivity
        await self.redis_client.expire(session_key, 7 * 24 * 3600)
    
    async def _trigger_summarization(
        self, 
        existing_summary: str | None, 
        turns_to_summarize: list[ConversationTurn]
    ) -> str:
        """Trigger background summarization of conversation turns."""
        try:
            async with await get_summarizer() as summarizer:
                request = SummarizationRequest(
                    existing_summary=existing_summary,
                    messages_to_summarize=turns_to_summarize,
                    max_summary_tokens=500
                )
                return await summarizer.summarize_conversation(request)
        
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            # Fallback: keep existing summary
            return existing_summary or ""
    
    async def add_turn(
        self, 
        user_id: str, 
        user_message: str, 
        assistant_response: str,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Add a conversation turn to the session buffer.
        
        This method:
        1. Acquires a distributed lock to prevent race conditions
        2. Adds the turn to the buffer
        3. Checks if token budget is exceeded
        4. If exceeded, summarizes older messages and updates buffer
        5. Releases the lock
        
        Args:
            user_id: User identifier
            user_message: User's message content
            assistant_response: Assistant's response content
            metadata: Optional metadata for the turn
        """
        if not self.redis_client:
            raise RedisConnectionError("Redis client not connected")
        
        # Create conversation turn
        turn = ConversationTurn(
            user_message=user_message,
            assistant_response=assistant_response,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        lock_key = self._get_lock_key(user_id)
        
        async with Lock(
            self.redis_client, 
            lock_key, 
            timeout=self._lock_timeout,
            blocking_timeout=5
        ):
            # Get current session data
            session_data = await self._get_session_data(user_id)
            
            # Parse current buffer
            buffer_json = session_data.get("buffer", "[]")
            current_buffer = [
                self._deserialize_turn(data) 
                for data in json.loads(buffer_json)
            ]
            
            # Add new turn
            current_buffer.append(turn)
            
            # Calculate token count
            current_summary = session_data.get("summary", "")
            summary_tokens = token_counter.count_tokens(current_summary)
            
            # Check if we need to summarize
            total_tokens = summary_tokens + sum(
                token_counter.count_conversation_turn_tokens(t) 
                for t in current_buffer
            )
            
            if total_tokens > self.max_token_budget and len(current_buffer) > 1:
                logger.info(
                    f"Token budget exceeded for user {user_id}: {total_tokens} > {self.max_token_budget}"
                )
                
                # Determine how many turns to summarize (keep recent ones in buffer)
                turns_within_budget, _ = token_counter.estimate_turns_within_budget(
                    list(reversed(current_buffer)),  # Most recent first
                    self.max_token_budget // 2,  # Use half budget for buffer
                    summary_tokens
                )
                
                # Keep recent turns that fit in half the budget
                turns_to_keep = list(reversed(turns_within_budget))
                turns_to_summarize = current_buffer[:-len(turns_to_keep)] if turns_to_keep else current_buffer[:-1]
                
                if turns_to_summarize:
                    # Summarize older turns
                    new_summary = await self._trigger_summarization(
                        current_summary if current_summary else None,
                        turns_to_summarize
                    )
                    
                    # Update session with new summary and reduced buffer
                    session_data.update({
                        "summary": new_summary,
                        "buffer": json.dumps([
                            self._serialize_turn(turn) for turn in turns_to_keep
                        ]),
                        "token_count": str(sum(
                            token_counter.count_conversation_turn_tokens(t) 
                            for t in turns_to_keep
                        ))
                    })
                else:
                    # Just update the buffer
                    session_data.update({
                        "buffer": json.dumps([
                            self._serialize_turn(turn) for turn in current_buffer
                        ]),
                        "token_count": str(total_tokens - summary_tokens)
                    })
            else:
                # No summarization needed, just update buffer
                session_data.update({
                    "buffer": json.dumps([
                        self._serialize_turn(turn) for turn in current_buffer
                    ]),
                    "token_count": str(total_tokens - summary_tokens)
                })
            
            # Save updated session data
            await self._save_session_data(user_id, session_data)
    
    async def get_context(self, user_id: str) -> ContextObject:
        """Retrieve conversation context for the user.
        
        Args:
            user_id: User identifier
            
        Returns:
            ContextObject with summary and recent messages
        """
        if not self.redis_client:
            raise RedisConnectionError("Redis client not connected")
        
        session_data = await self._get_session_data(user_id)
        
        # Parse buffer
        buffer_json = session_data.get("buffer", "[]")
        recent_turns = [
            self._deserialize_turn(data) 
            for data in json.loads(buffer_json)
        ]
        
        summary = session_data.get("summary", "")
        
        # Calculate total token count
        total_tokens = (
            token_counter.count_tokens(summary) +
            sum(token_counter.count_conversation_turn_tokens(turn) for turn in recent_turns)
        )
        
        return ContextObject(
            summary=summary if summary else None,
            recent_messages=recent_turns,
            token_count=total_tokens,
            last_updated=datetime.fromisoformat(session_data.get("last_updated", datetime.utcnow().isoformat())),
            metadata={
                "buffer_size": len(recent_turns),
                "summary_length": len(summary) if summary else 0
            }
        )
    
    async def clear_session(self, user_id: str) -> None:
        """Clear all session data for a user.
        
        Args:
            user_id: User identifier
        """
        if not self.redis_client:
            raise RedisConnectionError("Redis client not connected")
        
        session_key = self._get_session_key(user_id)
        await self.redis_client.delete(session_key)
        logger.info(f"Cleared session data for user {user_id}")
    
    async def get_session_stats(self, user_id: str) -> dict[str, Any]:
        """Get statistics about the user's session.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with session statistics
        """
        context = await self.get_context(user_id)
        
        return {
            "buffer_size": len(context.recent_messages),
            "has_summary": context.summary is not None,
            "total_tokens": context.token_count,
            "last_updated": context.last_updated.isoformat(),
            "summary_length": len(context.summary) if context.summary else 0
        }