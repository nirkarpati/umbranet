"""PostgreSQL database connection and utilities for episodic memory."""

import asyncio
import logging
from typing import Any

import asyncpg

from ...core.config import settings

logger = logging.getLogger(__name__)


class PostgresConnection:
    """Async PostgreSQL connection manager for memory storage."""
    
    def __init__(self):
        """Initialize connection manager."""
        self.pool: asyncpg.Pool | None = None
    
    async def __aenter__(self) -> "PostgresConnection":
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self) -> None:
        """Establish connection pool to PostgreSQL."""
        try:
            self.pool = await asyncpg.create_pool(
                settings.postgres_url,
                min_size=2,
                max_size=10,
                command_timeout=30,
                server_settings={
                    # Disable JIT for better performance with short queries
                    "jit": "off"
                }
            )
            
            # Test connection and ensure extensions
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
                
                # Ensure pgvector extension is available
                try:
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    logger.info("Connected to PostgreSQL with pgvector support")
                except Exception as e:
                    logger.warning(f"pgvector extension not available: {e}")
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            raise
    
    async def disconnect(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Disconnected from PostgreSQL")
    
    async def initialize_schema(self) -> None:
        """Initialize database schema for episodic memory."""
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")
        
        # Schema for episodic logs (append-only)
        create_episodic_logs = """
        CREATE TABLE IF NOT EXISTS episodic_logs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id VARCHAR(100) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            content TEXT NOT NULL,
            embedding vector(1536),  -- Default for OpenAI text-embedding-3-small
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
        
        # Indexes for efficient queries
        create_indexes = """
        CREATE INDEX IF NOT EXISTS idx_episodic_logs_user_id ON episodic_logs(user_id);
        CREATE INDEX IF NOT EXISTS idx_episodic_logs_timestamp ON episodic_logs(timestamp);
        CREATE INDEX IF NOT EXISTS idx_episodic_logs_user_timestamp ON episodic_logs(user_id, timestamp DESC);
        """
        
        # Vector similarity index (using HNSW for fast approximate search)
        create_vector_index = """
        CREATE INDEX IF NOT EXISTS idx_episodic_logs_embedding 
        ON episodic_logs USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
        
        async with self.pool.acquire() as conn:
            try:
                await conn.execute(create_episodic_logs)
                await conn.execute(create_indexes)
                
                # Vector index creation might fail if pgvector not available
                try:
                    await conn.execute(create_vector_index)
                    logger.info("Database schema initialized with vector index")
                except Exception as e:
                    logger.warning(f"Could not create vector index: {e}")
                    logger.info("Database schema initialized without vector index")
                    
            except Exception as e:
                logger.error(f"Failed to initialize schema: {e}")
                raise
    
    async def execute_query(
        self, 
        query: str, 
        *args: Any,
        fetch: bool = False
    ) -> list[asyncpg.Record] | None:
        """Execute a query with connection pool.
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            fetch: Whether to fetch results
            
        Returns:
            Query results if fetch=True, None otherwise
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")
        
        async with self.pool.acquire() as conn:
            if fetch:
                return await conn.fetch(query, *args)
            else:
                await conn.execute(query, *args)
                return None
    
    async def get_connection(self) -> asyncpg.Connection:
        """Get a connection from the pool for transaction use.
        
        Note: Caller is responsible for releasing the connection.
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")
        
        return await self.pool.acquire()
    
    def release_connection(self, conn: asyncpg.Connection) -> None:
        """Release a connection back to the pool."""
        if self.pool:
            asyncio.create_task(self.pool.release(conn))


# Global connection instance
_postgres_connection: PostgresConnection | None = None


async def get_postgres_connection() -> PostgresConnection:
    """Get global PostgreSQL connection instance."""
    global _postgres_connection
    
    if _postgres_connection is None:
        _postgres_connection = PostgresConnection()
        await _postgres_connection.connect()
    
    return _postgres_connection