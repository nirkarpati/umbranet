"""Neo4j database connection and utilities for semantic memory."""

import logging
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession
from neo4j.exceptions import AuthError, ServiceUnavailable

from ...core.config import settings

logger = logging.getLogger(__name__)


class Neo4jConnectionError(Exception):
    """Exception raised for Neo4j connection issues."""
    pass


class Neo4jConnection:
    """Async Neo4j connection manager for semantic memory storage."""
    
    def __init__(self):
        """Initialize connection manager."""
        self.driver: AsyncDriver | None = None
    
    async def __aenter__(self) -> "Neo4jConnection":
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self) -> None:
        """Establish connection to Neo4j."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                settings.neo4j_url,
                auth=settings.neo4j_auth,
                max_connection_lifetime=30 * 60,  # 30 minutes
                max_connection_pool_size=50,
                connection_acquisition_timeout=60,  # 60 seconds
                encrypted=False  # Set to True for production with SSL
            )
            
            # Test connection
            await self.driver.verify_connectivity()
            logger.info("Connected to Neo4j database")
            
        except AuthError as e:
            logger.error(f"Neo4j authentication failed: {str(e)}")
            raise Neo4jConnectionError(f"Authentication failed: {str(e)}") from e
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable: {str(e)}")
            raise Neo4jConnectionError(f"Service unavailable: {str(e)}") from e
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise Neo4jConnectionError(f"Connection failed: {str(e)}") from e
    
    async def disconnect(self) -> None:
        """Close Neo4j connection."""
        if self.driver:
            await self.driver.close()
            logger.info("Disconnected from Neo4j")
    
    async def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        database: str = "neo4j"
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return results.
        
        Args:
            query: Cypher query to execute
            parameters: Query parameters
            database: Database name (default: "neo4j")
            
        Returns:
            List of result records as dictionaries
            
        Raises:
            Neo4jConnectionError: If query execution fails
        """
        if not self.driver:
            raise Neo4jConnectionError("Driver not connected - use async context manager")
        
        try:
            async with self.driver.session(database=database) as session:
                result = await session.run(query, parameters or {})
                records = [record.data() async for record in result]
                return records
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise Neo4jConnectionError(f"Query failed: {str(e)}") from e
    
    async def execute_write_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        database: str = "neo4j"
    ) -> list[dict[str, Any]]:
        """Execute a write query in a transaction.
        
        Args:
            query: Cypher query to execute
            parameters: Query parameters
            database: Database name
            
        Returns:
            List of result records
        """
        if not self.driver:
            raise Neo4jConnectionError("Driver not connected - use async context manager")
        
        try:
            async with self.driver.session(database=database) as session:
                result = await session.execute_write(
                    self._execute_query_tx, query, parameters or {}
                )
                return result
        except Exception as e:
            logger.error(f"Write query execution failed: {str(e)}")
            raise Neo4jConnectionError(f"Write query failed: {str(e)}") from e
    
    @staticmethod
    async def _execute_query_tx(tx, query: str, parameters: dict[str, Any]):
        """Execute query within transaction."""
        result = await tx.run(query, parameters)
        return [record.data() async for record in result]
    
    async def get_session(self, database: str = "neo4j") -> AsyncSession:
        """Get a session for complex operations.
        
        Note: Caller is responsible for closing the session.
        """
        if not self.driver:
            raise Neo4jConnectionError("Driver not connected - use async context manager")
        
        return self.driver.session(database=database)
    
    async def initialize_schema(self) -> None:
        """Initialize database schema and constraints for semantic memory."""
        if not self.driver:
            raise Neo4jConnectionError("Driver not connected")
        
        # Create constraints for unique entities per user
        constraints = [
            # Unique constraint on User entities
            "CREATE CONSTRAINT user_entity_unique IF NOT EXISTS "
            "FOR (u:User) REQUIRE (u.user_id, u.entity_id) IS UNIQUE",
            
            # Unique constraint on SystemAgent
            "CREATE CONSTRAINT system_agent_unique IF NOT EXISTS "
            "FOR (s:SystemAgent) REQUIRE s.entity_id IS UNIQUE",
            
            # Index on user_id for fast tenant filtering
            "CREATE INDEX user_id_index IF NOT EXISTS FOR (n:User) ON (n.user_id)",
            
            # Index on entity names for text searches
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (n:Entity) ON (n.name)",
            
            # Index on relationship weights for filtering  
            "CREATE INDEX relationship_weight_index IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.weight)"
        ]
        
        try:
            for constraint in constraints:
                await self.execute_write_query(constraint)
            
            logger.info("Neo4j schema initialized successfully")
        except Exception as e:
            logger.warning(f"Schema initialization partially failed: {e}")
            # Continue anyway - some constraints might already exist


# Global connection instance
_neo4j_connection: Neo4jConnection | None = None


async def get_neo4j_connection() -> Neo4jConnection:
    """Get global Neo4j connection instance."""
    global _neo4j_connection
    
    if _neo4j_connection is None:
        _neo4j_connection = Neo4jConnection()
        await _neo4j_connection.connect()
    
    return _neo4j_connection