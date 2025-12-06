"""Episodic memory implementation using PGVector for vector similarity search.

This module implements Tier 2 of the RAG++ memory hierarchy, providing:
- Append-only logging of all interactions
- Vector similarity search for memory recall
- Time-stamped episode storage
- Metadata support for rich context
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any

from ...core.domain.memory import MemoryEntry
from ...core.embeddings.base import EmbeddingProvider
from ...core.embeddings.provider_factory import get_embedding_provider
from ..database.postgres import PostgresConnection, get_postgres_connection

logger = logging.getLogger(__name__)


class EpisodicMemoryError(Exception):
    """Exception raised for episodic memory operations."""
    pass


class EpisodicMemoryStore:
    """PGVector-based episodic memory store with vector similarity search."""
    
    def __init__(self, embedding_provider: EmbeddingProvider | None = None):
        """Initialize episodic memory store.
        
        Args:
            embedding_provider: Embedding provider to use, defaults to auto-selection
        """
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.postgres: PostgresConnection | None = None
    
    async def __aenter__(self) -> "EpisodicMemoryStore":
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self) -> None:
        """Initialize connections to database and embedding provider."""
        try:
            # Initialize database connection
            self.postgres = await get_postgres_connection()
            await self.postgres.initialize_schema()
            
            logger.info("Connected to episodic memory store")
        except Exception as e:
            logger.error(f"Failed to connect to episodic memory store: {str(e)}")
            raise EpisodicMemoryError(
                f"Connection failed: {str(e)}"
            ) from e
    
    async def disconnect(self) -> None:
        """Close connections."""
        # PostgresConnection handles its own lifecycle
        logger.info("Disconnected from episodic memory store")
    
    def _format_interaction_content(
        self, 
        user_message: str, 
        assistant_response: str
    ) -> str:
        """Format interaction for storage and embedding.
        
        Args:
            user_message: User's input message
            assistant_response: Assistant's response
            
        Returns:
            Formatted content string for embedding
        """
        return f"User: {user_message}\nAssistant: {assistant_response}"
    
    async def log_episode(
        self,
        user_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None
    ) -> str:
        """Log a generic episode to episodic memory.
        
        This method allows saving arbitrary content (observations, facts, etc.)
        without forcing it into the user/assistant conversation format.
        
        Args:
            user_id: User identifier
            content: The content to save
            metadata: Optional metadata
            timestamp: Optional custom timestamp for when the event occurred.
                      If None, uses current UTC time for storage time.
            
        Returns:
            UUID of the logged episode
            
        Raises:
            EpisodicMemoryError: If logging fails
        """
        if not self.postgres:
            raise EpisodicMemoryError("Store not connected - use async context manager")
        
        try:
            # Prepare metadata with temporal grounding information
            full_metadata = metadata or {}
            
            # Use provided timestamp or current time
            storage_timestamp = datetime.utcnow()  # Always use current time for storage
            
            # If custom timestamp provided, store the occurrence time in metadata
            if timestamp:
                full_metadata["occurred_at"] = timestamp.isoformat()
                # Enhance content with temporal context for better vector grounding
                date_str = timestamp.strftime("%Y-%m-%d")
                content = f"[Date: {date_str}] {content}"
            
            # Generate embedding from the potentially enhanced content
            async with self.embedding_provider as provider:
                embedding = await provider.embed_text(content)
            
            # Store in database
            episode_id = str(uuid.uuid4())
            
            query = """
                INSERT INTO episodic_logs 
                (id, user_id, timestamp, content, embedding, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
            """
            
            await self.postgres.execute_query(
                query,
                episode_id,
                user_id,
                storage_timestamp,
                content,
                str(embedding),  # Convert list to string representation for pgvector
                json.dumps(full_metadata)
            )
            
            logger.debug(f"Logged episode {episode_id} for user {user_id}")
            return episode_id
            
        except Exception as e:
            logger.error(f"Failed to log episode for user {user_id}: {str(e)}")
            raise EpisodicMemoryError(f"Episode logging failed: {str(e)}") from e

    async def log_interaction(
        self,
        user_id: str,
        user_message: str,
        assistant_response: str,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Log an interaction to episodic memory.
        
        This is an append-only operation that stores the conversation turn
        with its vector embedding for future similarity search.
        
        Args:
            user_id: User identifier
            user_message: User's input message
            assistant_response: Assistant's response message
            metadata: Optional metadata (location, tool_used, etc.)
            
        Returns:
            UUID of the logged episode
            
        Raises:
            EpisodicMemoryError: If logging fails
        """
        # Format content for embedding and delegate to log_episode
        content = self._format_interaction_content(user_message, assistant_response)
        return await self.log_episode(user_id, content, metadata)
    
    async def recall(
        self,
        user_id: str,
        query_text: str,
        limit: int = 5,
        similarity_threshold: float = 0.7
    ) -> list[MemoryEntry]:
        """Recall similar past interactions using vector similarity search.
        
        Args:
            user_id: User identifier
            query_text: Text to find similar interactions for
            limit: Maximum number of results to return
            similarity_threshold: Minimum cosine similarity (0.0 to 1.0)
            
        Returns:
            List of similar memory entries, ordered by similarity
            
        Raises:
            EpisodicMemoryError: If recall fails
        """
        if not self.postgres:
            raise EpisodicMemoryError("Store not connected - use async context manager")
        
        try:
            # Generate query embedding
            async with self.embedding_provider as provider:
                query_embedding = await provider.embed_text(query_text)
            
            # Vector similarity search using cosine distance
            # Note: pgvector uses <=> for cosine distance (lower is more similar)
            query = """
                SELECT 
                    id,
                    user_id,
                    timestamp,
                    content,
                    metadata,
                    1 - (embedding <=> $2) AS similarity
                FROM episodic_logs
                WHERE user_id = $1
                    AND 1 - (embedding <=> $2) >= $3
                ORDER BY embedding <=> $2
                LIMIT $4
            """
            
            results = await self.postgres.execute_query(
                query,
                user_id,
                str(query_embedding),  # Convert list to string for pgvector
                similarity_threshold,
                limit,
                fetch=True
            )
            
            # Convert to MemoryEntry objects
            memories = []
            for row in results or []:
                memories.append(MemoryEntry(
                    user_id=row['user_id'],
                    content=row['content'],
                    timestamp=row['timestamp'],
                    metadata={
                        'episode_id': row['id'],
                        'similarity': float(row['similarity']),
                        **json.loads(row['metadata'])
                    }
                ))
            
            logger.debug(
                f"Recalled {len(memories)} memories for user {user_id} "
                f"with query: {query_text[:50]}..."
            )
            return memories
            
        except Exception as e:
            logger.error(f"Failed to recall memories for user {user_id}: {str(e)}")
            raise EpisodicMemoryError(f"Recall failed: {str(e)}") from e
    
    async def get_recent_interactions(
        self,
        user_id: str,
        limit: int = 10
    ) -> list[MemoryEntry]:
        """Get the most recent interactions for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of results to return
            
        Returns:
            List of recent memory entries, ordered by timestamp (newest first)
        """
        if not self.postgres:
            raise EpisodicMemoryError("Store not connected - use async context manager")
        
        try:
            query = """
                SELECT id, user_id, timestamp, content, metadata
                FROM episodic_logs
                WHERE user_id = $1
                ORDER BY timestamp DESC
                LIMIT $2
            """
            
            results = await self.postgres.execute_query(
                query, user_id, limit, fetch=True
            )
            
            memories = []
            for row in results or []:
                memories.append(MemoryEntry(
                    user_id=row['user_id'],
                    content=row['content'],
                    timestamp=row['timestamp'],
                    metadata={
                        'episode_id': row['id'],
                        **json.loads(row['metadata'])
                    }
                ))
            
            return memories
            
        except Exception as e:
            logger.error(
                f"Failed to get recent interactions for user {user_id}: {str(e)}"
            )
            raise EpisodicMemoryError(f"Recent recall failed: {str(e)}") from e
    
    async def get_interaction_count(self, user_id: str) -> int:
        """Get total number of logged interactions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of logged interactions
        """
        if not self.postgres:
            raise EpisodicMemoryError("Store not connected - use async context manager")
        
        try:
            query = "SELECT COUNT(*) as count FROM episodic_logs WHERE user_id = $1"
            result = await self.postgres.execute_query(query, user_id, fetch=True)
            
            if result and len(result) > 0:
                return int(result[0]['count'])
            return 0
            
        except Exception as e:
            logger.error(
                f"Failed to get interaction count for user {user_id}: {str(e)}"
            )
            return 0
    
    async def get_recent_episodes(
        self, user_id: str, limit: int = 10
    ) -> list[dict]:
        """Get recent episodes for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of episodes to return
            
        Returns:
            List of episode dictionaries
        """
        if not self.postgres:
            raise EpisodicMemoryError("Store not connected - use async context manager")
        
        try:
            query = """
            SELECT content, timestamp, metadata
            FROM episodic_logs 
            WHERE user_id = $1
            ORDER BY timestamp DESC
            LIMIT $2
            """
            
            rows = await self.postgres.execute_query(query, user_id, limit, fetch=True)
            episodes = []
            
            for row in rows or []:  # Add null safety like get_recent_interactions
                episodes.append({
                    "content": row['content'],  # Use dictionary access
                    "timestamp": (
                        row['timestamp'].isoformat() 
                        if row['timestamp'] else ""
                    ),
                    "metadata": (
                        json.loads(row['metadata']) 
                        if row['metadata'] else {}
                    )  # Parse JSON metadata
                })
            
            return episodes
            
        except Exception as e:
            logger.error(f"Failed to get recent episodes for user {user_id}: {str(e)}")
            return []
    
    async def delete_user_data(self, user_id: str) -> int:
        """Delete all episodic data for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of deleted records
        """
        if not self.postgres:
            raise EpisodicMemoryError("Store not connected - use async context manager")
        
        try:
            # First count existing records
            count = await self.get_interaction_count(user_id)
            
            # Delete all user data
            query = "DELETE FROM episodic_logs WHERE user_id = $1"
            await self.postgres.execute_query(query, user_id)
            
            logger.info(f"Deleted {count} episodic records for user {user_id}")
            return count
            
        except Exception as e:
            logger.error(f"Failed to delete data for user {user_id}: {str(e)}")
            raise EpisodicMemoryError(f"Deletion failed: {str(e)}") from e