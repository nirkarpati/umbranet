"""Procedural memory implementation using PostgreSQL for profiles and vectorized instructions.

This module implements Tier 4 of the RAG++ memory hierarchy, providing:
- Static profile store for hard facts (key-value pairs)
- Vectorized instruction store for behavioral preferences
- Context-dependent rule retrieval via embedding similarity
- Profile and instruction management with metadata tracking
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any

import asyncpg

from ...core.embeddings.base import EmbeddingProvider
from ...core.embeddings.provider_factory import get_embedding_provider
from ...core.domain.procedural import (
    BehavioralInstruction,
    InstructionCategory,
    InstructionQuery,
    ProceduralMemoryStats,
    ProfileCategory,
    ProfileEntry,
    RelevantInstruction,
    UserProfile,
)
from ..database.postgres import get_postgres_connection, PostgresConnection

logger = logging.getLogger(__name__)


class ProceduralMemoryError(Exception):
    """Exception raised for procedural memory operations."""
    pass


class ProceduralMemoryStore:
    """PostgreSQL-based procedural memory with static profiles and vectorized instructions."""
    
    def __init__(self, embedding_provider: EmbeddingProvider | None = None):
        """Initialize procedural memory store.
        
        Args:
            embedding_provider: Embedding provider for instruction vectorization
        """
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.postgres: PostgresConnection | None = None
    
    async def __aenter__(self) -> "ProceduralMemoryStore":
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self) -> None:
        """Initialize connections to database and embedding provider."""
        try:
            self.postgres = await get_postgres_connection()
            await self._initialize_schema()
            
            logger.info("Connected to procedural memory store")
        except Exception as e:
            logger.error(f"Failed to connect to procedural memory store: {str(e)}")
            raise ProceduralMemoryError(f"Connection failed: {str(e)}") from e
    
    async def disconnect(self) -> None:
        """Close connections."""
        logger.info("Disconnected from procedural memory store")
    
    async def _initialize_schema(self) -> None:
        """Initialize database schema for procedural memory."""
        if not self.postgres:
            raise ProceduralMemoryError("Database not connected")
        
        # User profile table (key-value store)
        create_user_profile = """
        CREATE TABLE IF NOT EXISTS user_profile (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id VARCHAR(100) NOT NULL,
            category VARCHAR(50) NOT NULL,
            key VARCHAR(100) NOT NULL,
            value TEXT NOT NULL,
            value_type VARCHAR(20) DEFAULT 'string',
            is_sensitive BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            metadata JSONB DEFAULT '{}'
        );
        """
        
        # Agent instructions table (vectorized rules)
        create_agent_instructions = """
        CREATE TABLE IF NOT EXISTS agent_instructions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id VARCHAR(100) NOT NULL,
            instruction_id VARCHAR(100) NOT NULL,
            category VARCHAR(50) NOT NULL,
            title VARCHAR(200) NOT NULL,
            instruction TEXT NOT NULL,
            confidence FLOAT DEFAULT 0.5 CHECK (confidence >= 0.0 AND confidence <= 1.0),
            priority INTEGER DEFAULT 1 CHECK (priority >= 1 AND priority <= 10),
            is_active BOOLEAN DEFAULT TRUE,
            embedding vector(1536),  -- Default for OpenAI embeddings
            examples JSONB DEFAULT '[]',
            exceptions JSONB DEFAULT '[]',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_used TIMESTAMPTZ,
            usage_count INTEGER DEFAULT 0,
            source VARCHAR(50) DEFAULT 'user_explicit',
            metadata JSONB DEFAULT '{}'
        );
        """
        
        # Indexes for efficient queries
        create_indexes = """
        -- Profile indexes
        CREATE UNIQUE INDEX IF NOT EXISTS idx_user_profile_unique 
        ON user_profile(user_id, category, key);
        
        CREATE INDEX IF NOT EXISTS idx_user_profile_user_id ON user_profile(user_id);
        CREATE INDEX IF NOT EXISTS idx_user_profile_category ON user_profile(category);
        
        -- Instruction indexes  
        CREATE UNIQUE INDEX IF NOT EXISTS idx_instructions_unique
        ON agent_instructions(user_id, instruction_id);
        
        CREATE INDEX IF NOT EXISTS idx_instructions_user_id ON agent_instructions(user_id);
        CREATE INDEX IF NOT EXISTS idx_instructions_category ON agent_instructions(category);
        CREATE INDEX IF NOT EXISTS idx_instructions_active ON agent_instructions(user_id, is_active);
        CREATE INDEX IF NOT EXISTS idx_instructions_confidence ON agent_instructions(confidence);
        CREATE INDEX IF NOT EXISTS idx_instructions_priority ON agent_instructions(priority);
        """
        
        # Vector similarity index for instructions
        create_vector_index = """
        CREATE INDEX IF NOT EXISTS idx_instructions_embedding 
        ON agent_instructions USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
        
        try:
            await self.postgres.execute_query(create_user_profile)
            await self.postgres.execute_query(create_agent_instructions)
            await self.postgres.execute_query(create_indexes)
            
            # Vector index creation might fail if pgvector not available
            try:
                await self.postgres.execute_query(create_vector_index)
                logger.info("Procedural memory schema initialized with vector index")
            except Exception as e:
                logger.warning(f"Could not create vector index: {e}")
                logger.info("Procedural memory schema initialized without vector index")
                
        except Exception as e:
            logger.error(f"Failed to initialize procedural memory schema: {e}")
            raise ProceduralMemoryError(f"Schema initialization failed: {str(e)}") from e
    
    # --- Profile Store Methods ---
    
    async def set_profile_value(
        self,
        user_id: str,
        category: ProfileCategory,
        key: str,
        value: str,
        value_type: str = "string",
        is_sensitive: bool = False,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Set a profile value (upsert operation).
        
        Args:
            user_id: User identifier
            category: Profile category
            key: Profile key
            value: Profile value
            value_type: Type of the value
            is_sensitive: Whether the data is sensitive
            metadata: Additional metadata
        """
        if not self.postgres:
            raise ProceduralMemoryError("Store not connected")
        
        try:
            query = """
            INSERT INTO user_profile 
            (user_id, category, key, value, value_type, is_sensitive, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (user_id, category, key)
            DO UPDATE SET
                value = EXCLUDED.value,
                value_type = EXCLUDED.value_type,
                is_sensitive = EXCLUDED.is_sensitive,
                updated_at = NOW(),
                metadata = EXCLUDED.metadata
            """
            
            await self.postgres.execute_query(
                query,
                user_id,
                category.value,
                key.lower().replace(' ', '_').replace('-', '_'),
                value,
                value_type,
                is_sensitive,
                json.dumps(metadata or {})
            )
            
            logger.debug(f"Set profile value {category.value}.{key} for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to set profile value: {str(e)}")
            raise ProceduralMemoryError(f"Profile update failed: {str(e)}") from e
    
    async def get_profile_value(
        self,
        user_id: str,
        category: ProfileCategory,
        key: str
    ) -> str | None:
        """Get a single profile value.
        
        Args:
            user_id: User identifier
            category: Profile category
            key: Profile key
            
        Returns:
            Profile value or None if not found
        """
        if not self.postgres:
            raise ProceduralMemoryError("Store not connected")
        
        try:
            query = """
            SELECT value FROM user_profile
            WHERE user_id = $1 AND category = $2 AND key = $3
            """
            
            normalized_key = key.lower().replace(' ', '_').replace('-', '_')
            result = await self.postgres.execute_query(
                query, user_id, category.value, normalized_key, fetch=True
            )
            
            return result[0]['value'] if result else None
            
        except Exception as e:
            logger.error(f"Failed to get profile value: {str(e)}")
            return None
    
    async def get_user_profile(self, user_id: str) -> UserProfile:
        """Get complete user profile.
        
        Args:
            user_id: User identifier
            
        Returns:
            Complete user profile
        """
        if not self.postgres:
            raise ProceduralMemoryError("Store not connected")
        
        try:
            query = """
            SELECT category, key, value, value_type, is_sensitive, 
                   created_at, updated_at, metadata
            FROM user_profile
            WHERE user_id = $1
            ORDER BY category, key
            """
            
            results = await self.postgres.execute_query(query, user_id, fetch=True)
            
            profile = UserProfile(user_id=user_id)
            
            for row in results or []:
                entry_key = f"{row['category']}.{row['key']}"
                
                profile.entries[entry_key] = ProfileEntry(
                    user_id=user_id,
                    category=ProfileCategory(row['category']),
                    key=row['key'],
                    value=row['value'],
                    value_type=row['value_type'],
                    is_sensitive=row['is_sensitive'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    metadata=json.loads(row['metadata'])
                )
            
            if results:
                profile.last_updated = max(entry.updated_at for entry in profile.entries.values())
            
            return profile
            
        except Exception as e:
            logger.error(f"Failed to get user profile: {str(e)}")
            raise ProceduralMemoryError(f"Profile retrieval failed: {str(e)}") from e
    
    async def delete_profile_value(
        self,
        user_id: str,
        category: ProfileCategory,
        key: str
    ) -> bool:
        """Delete a profile value.
        
        Args:
            user_id: User identifier
            category: Profile category
            key: Profile key
            
        Returns:
            True if deleted, False if not found
        """
        if not self.postgres:
            raise ProceduralMemoryError("Store not connected")
        
        try:
            query = """
            DELETE FROM user_profile
            WHERE user_id = $1 AND category = $2 AND key = $3
            """
            
            normalized_key = key.lower().replace(' ', '_').replace('-', '_')
            result = await self.postgres.execute_query(
                query, user_id, category.value, normalized_key
            )
            
            # Check if any rows were affected (asyncpg doesn't return rowcount easily)
            # So we'll do a simple existence check before deletion
            existing = await self.get_profile_value(user_id, category, key)
            if existing:
                await self.postgres.execute_query(
                    query, user_id, category.value, normalized_key
                )
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete profile value: {str(e)}")
            return False
    
    # --- Instruction Store Methods ---
    
    async def add_instruction(
        self,
        user_id: str,
        category: InstructionCategory,
        title: str,
        instruction: str,
        confidence: float = 0.5,
        priority: int = 1,
        examples: list[str] | None = None,
        exceptions: list[str] | None = None,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """Add a behavioral instruction.
        
        Args:
            user_id: User identifier
            category: Instruction category
            title: Instruction title
            instruction: Instruction text
            confidence: Confidence level (0.0-1.0)
            priority: Priority level (1-10)
            examples: Example scenarios
            exceptions: Exception conditions
            metadata: Additional metadata
            
        Returns:
            Instruction ID
        """
        if not self.postgres:
            raise ProceduralMemoryError("Store not connected")
        
        try:
            instruction_id = str(uuid.uuid4())
            
            # Generate embedding for the instruction
            async with self.embedding_provider as provider:
                embedding = await provider.embed_text(instruction)
            
            query = """
            INSERT INTO agent_instructions 
            (user_id, instruction_id, category, title, instruction, confidence, 
             priority, embedding, examples, exceptions, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING instruction_id
            """
            
            result = await self.postgres.execute_query(
                query,
                user_id,
                instruction_id,
                category.value,
                title,
                instruction,
                confidence,
                priority,
                embedding,
                json.dumps(examples or []),
                json.dumps(exceptions or []),
                json.dumps(metadata or {}),
                fetch=True
            )
            
            logger.debug(f"Added instruction {instruction_id} for user {user_id}")
            return result[0]['instruction_id']
            
        except Exception as e:
            logger.error(f"Failed to add instruction: {str(e)}")
            raise ProceduralMemoryError(f"Instruction creation failed: {str(e)}") from e
    
    async def update_instruction(
        self,
        user_id: str,
        instruction_id: str,
        **updates: Any
    ) -> bool:
        """Update an existing instruction.
        
        Args:
            user_id: User identifier
            instruction_id: Instruction identifier
            **updates: Fields to update
            
        Returns:
            True if updated, False if not found
        """
        if not self.postgres:
            raise ProceduralMemoryError("Store not connected")
        
        if not updates:
            return False
        
        try:
            # Build dynamic update query
            set_clauses = []
            params = [user_id, instruction_id]
            param_count = 2
            
            for field, value in updates.items():
                if field == 'instruction':
                    # Re-embed if instruction text changed
                    async with self.embedding_provider as provider:
                        embedding = await provider.embed_text(value)
                    param_count += 1
                    set_clauses.append(f"instruction = ${param_count}")
                    params.append(value)
                    param_count += 1
                    set_clauses.append(f"embedding = ${param_count}")
                    params.append(embedding)
                elif field in ['examples', 'exceptions', 'metadata']:
                    param_count += 1
                    set_clauses.append(f"{field} = ${param_count}")
                    params.append(json.dumps(value))
                else:
                    param_count += 1
                    set_clauses.append(f"{field} = ${param_count}")
                    params.append(value)
            
            if not set_clauses:
                return False
            
            set_clauses.append("updated_at = NOW()")
            
            query = f"""
            UPDATE agent_instructions 
            SET {', '.join(set_clauses)}
            WHERE user_id = $1 AND instruction_id = $2
            """
            
            await self.postgres.execute_query(query, *params)
            
            logger.debug(f"Updated instruction {instruction_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update instruction: {str(e)}")
            return False
    
    async def get_relevant_instructions(
        self,
        user_id: str,
        query_text: str,
        categories: list[InstructionCategory] | None = None,
        min_confidence: float = 0.3,
        min_priority: int = 1,
        include_inactive: bool = False,
        limit: int = 10
    ) -> list[RelevantInstruction]:
        """Get behavioral instructions relevant to query text.
        
        Args:
            user_id: User identifier
            query_text: Query text to find relevant instructions
            categories: Specific categories to search (None = all)
            min_confidence: Minimum confidence threshold
            min_priority: Minimum priority level
            include_inactive: Whether to include inactive instructions
            limit: Maximum number of results
            
        Returns:
            List of relevant instructions with similarity scores
        """
        if not self.postgres:
            raise ProceduralMemoryError("Store not connected")
        
        try:
            # Generate query embedding
            async with self.embedding_provider as provider:
                query_embedding = await provider.embed_text(query_text)
            
            # Build query conditions
            conditions = [
                "user_id = $1",
                "confidence >= $3",
                "priority >= $4"
            ]
            params = [user_id, query_embedding, min_confidence, min_priority]
            param_count = 4
            
            if not include_inactive:
                conditions.append("is_active = TRUE")
            
            if categories:
                param_count += 1
                category_values = [cat.value for cat in categories]
                conditions.append(f"category = ANY(${param_count})")
                params.append(category_values)
            
            # Vector similarity search query
            query = f"""
            SELECT 
                instruction_id, category, title, instruction, confidence, priority,
                is_active, examples, exceptions, created_at, updated_at, 
                last_used, usage_count, source, metadata,
                1 - (embedding <=> $2) as similarity
            FROM agent_instructions
            WHERE {' AND '.join(conditions)}
                AND embedding IS NOT NULL
            ORDER BY embedding <=> $2
            LIMIT $5
            """
            
            params.append(limit)
            
            results = await self.postgres.execute_query(query, *params, fetch=True)
            
            # Convert to RelevantInstruction objects
            relevant_instructions = []
            for row in results or []:
                instruction = BehavioralInstruction(
                    user_id=user_id,
                    instruction_id=row['instruction_id'],
                    category=InstructionCategory(row['category']),
                    title=row['title'],
                    instruction=row['instruction'],
                    confidence=float(row['confidence']),
                    priority=int(row['priority']),
                    is_active=row['is_active'],
                    examples=json.loads(row['examples']),
                    exceptions=json.loads(row['exceptions']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    last_used=row['last_used'],
                    usage_count=int(row['usage_count']),
                    source=row['source'],
                    metadata=json.loads(row['metadata'])
                )
                
                relevant_instructions.append(RelevantInstruction(
                    instruction=instruction,
                    relevance_score=float(row['similarity']),
                    match_reason=f"Vector similarity: {row['similarity']:.3f}"
                ))
            
            # Update usage statistics for returned instructions
            if relevant_instructions:
                instruction_ids = [ri.instruction.instruction_id for ri in relevant_instructions]
                update_usage_query = """
                UPDATE agent_instructions 
                SET usage_count = usage_count + 1, last_used = NOW()
                WHERE user_id = $1 AND instruction_id = ANY($2)
                """
                await self.postgres.execute_query(update_usage_query, user_id, instruction_ids)
            
            logger.debug(
                f"Retrieved {len(relevant_instructions)} relevant instructions "
                f"for user {user_id} query: {query_text[:50]}..."
            )
            
            return relevant_instructions
            
        except Exception as e:
            logger.error(f"Failed to get relevant instructions: {str(e)}")
            raise ProceduralMemoryError(f"Instruction retrieval failed: {str(e)}") from e
    
    async def get_all_instructions(
        self,
        user_id: str,
        include_inactive: bool = False
    ) -> list[BehavioralInstruction]:
        """Get all instructions for a user.
        
        Args:
            user_id: User identifier
            include_inactive: Whether to include inactive instructions
            
        Returns:
            List of all user instructions
        """
        if not self.postgres:
            raise ProceduralMemoryError("Store not connected")
        
        try:
            conditions = ["user_id = $1"]
            params = [user_id]
            
            if not include_inactive:
                conditions.append("is_active = TRUE")
            
            query = f"""
            SELECT 
                instruction_id, category, title, instruction, confidence, priority,
                is_active, examples, exceptions, created_at, updated_at,
                last_used, usage_count, source, metadata
            FROM agent_instructions
            WHERE {' AND '.join(conditions)}
            ORDER BY priority DESC, confidence DESC, created_at DESC
            """
            
            results = await self.postgres.execute_query(query, *params, fetch=True)
            
            instructions = []
            for row in results or []:
                instructions.append(BehavioralInstruction(
                    user_id=user_id,
                    instruction_id=row['instruction_id'],
                    category=InstructionCategory(row['category']),
                    title=row['title'],
                    instruction=row['instruction'],
                    confidence=float(row['confidence']),
                    priority=int(row['priority']),
                    is_active=row['is_active'],
                    examples=json.loads(row['examples']),
                    exceptions=json.loads(row['exceptions']),
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    last_used=row['last_used'],
                    usage_count=int(row['usage_count']),
                    source=row['source'],
                    metadata=json.loads(row['metadata'])
                ))
            
            return instructions
            
        except Exception as e:
            logger.error(f"Failed to get all instructions: {str(e)}")
            return []
    
    async def delete_instruction(self, user_id: str, instruction_id: str) -> bool:
        """Delete an instruction.
        
        Args:
            user_id: User identifier
            instruction_id: Instruction identifier
            
        Returns:
            True if deleted, False if not found
        """
        if not self.postgres:
            raise ProceduralMemoryError("Store not connected")
        
        try:
            query = """
            DELETE FROM agent_instructions
            WHERE user_id = $1 AND instruction_id = $2
            """
            
            await self.postgres.execute_query(query, user_id, instruction_id)
            
            logger.debug(f"Deleted instruction {instruction_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete instruction: {str(e)}")
            return False
    
    # --- Statistics and Management ---
    
    async def get_procedural_memory_stats(self, user_id: str) -> ProceduralMemoryStats:
        """Get statistics about user's procedural memory.
        
        Args:
            user_id: User identifier
            
        Returns:
            Procedural memory statistics
        """
        if not self.postgres:
            raise ProceduralMemoryError("Store not connected")
        
        try:
            # Get profile stats
            profile_query = """
            SELECT 
                COUNT(*) as total_entries,
                category,
                COUNT(*) as category_count,
                MAX(updated_at) as last_update
            FROM user_profile
            WHERE user_id = $1
            GROUP BY category
            """
            
            # Get instruction stats
            instruction_query = """
            SELECT 
                COUNT(*) as total_instructions,
                COUNT(CASE WHEN is_active THEN 1 END) as active_instructions,
                AVG(confidence) as avg_confidence,
                category,
                COUNT(*) as category_count,
                MAX(updated_at) as last_update
            FROM agent_instructions
            WHERE user_id = $1
            GROUP BY category
            """
            
            profile_results = await self.postgres.execute_query(
                profile_query, user_id, fetch=True
            )
            instruction_results = await self.postgres.execute_query(
                instruction_query, user_id, fetch=True
            )
            
            # Build statistics
            profile_categories = {}
            total_profile_entries = 0
            last_profile_update = None
            
            for row in profile_results or []:
                profile_categories[row['category']] = row['category_count']
                total_profile_entries += row['category_count']
                if not last_profile_update or row['last_update'] > last_profile_update:
                    last_profile_update = row['last_update']
            
            instruction_categories = {}
            total_instructions = 0
            active_instructions = 0
            avg_confidence = 0.0
            last_instruction_update = None
            
            for row in instruction_results or []:
                instruction_categories[row['category']] = row['category_count']
                total_instructions += row['category_count']
                active_instructions += row.get('active_instructions', 0)
                if row.get('avg_confidence'):
                    avg_confidence = float(row['avg_confidence'])
                if not last_instruction_update or row['last_update'] > last_instruction_update:
                    last_instruction_update = row['last_update']
            
            return ProceduralMemoryStats(
                user_id=user_id,
                total_profile_entries=total_profile_entries,
                total_instructions=total_instructions,
                profile_categories=profile_categories,
                instruction_categories=instruction_categories,
                active_instructions=active_instructions,
                average_instruction_confidence=avg_confidence,
                last_profile_update=last_profile_update,
                last_instruction_update=last_instruction_update
            )
            
        except Exception as e:
            logger.error(f"Failed to get procedural memory stats: {str(e)}")
            return ProceduralMemoryStats(user_id=user_id)
    
    async def get_user_rules(self, user_id: str) -> list[dict]:
        """Get procedural rules and instructions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of rule dictionaries
        """
        if not self.postgres:
            raise ProceduralMemoryError("Store not connected")
        
        try:
            # Get instructions which are the procedural rules
            instructions = await self.get_all_instructions(user_id)
            
            rules = []
            for instruction in instructions:
                rules.append({
                    "title": instruction.title,
                    "instruction": instruction.instruction,
                    "priority": instruction.priority
                })
            
            # Also add profile-based preferences
            profile = await self.get_user_profile(user_id)
            if profile.entries:
                for entry_key, entry in profile.entries.items():
                    # entry_key format is "category.key", value is ProfileEntry object
                    category_key = entry_key.replace(".", ": ")
                    rules.append({
                        "title": f"User Preference: {category_key}",
                        "instruction": f"Remember that user prefers: {entry.value}"
                    })
            
            return rules
            
        except Exception as e:
            logger.error(f"Failed to get user rules for {user_id}: {str(e)}")
            return []
    
    async def delete_all_user_data(self, user_id: str) -> tuple[int, int]:
        """Delete all procedural memory data for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (deleted_profile_entries, deleted_instructions)
        """
        if not self.postgres:
            raise ProceduralMemoryError("Store not connected")
        
        try:
            # Count before deletion
            stats = await self.get_procedural_memory_stats(user_id)
            
            # Delete profile data
            profile_query = "DELETE FROM user_profile WHERE user_id = $1"
            await self.postgres.execute_query(profile_query, user_id)
            
            # Delete instruction data
            instruction_query = "DELETE FROM agent_instructions WHERE user_id = $1"
            await self.postgres.execute_query(instruction_query, user_id)
            
            deleted_profile = stats.total_profile_entries
            deleted_instructions = stats.total_instructions
            
            logger.info(
                f"Deleted {deleted_profile} profile entries and "
                f"{deleted_instructions} instructions for user {user_id}"
            )
            
            return deleted_profile, deleted_instructions
            
        except Exception as e:
            logger.error(f"Failed to delete user data: {str(e)}")
            raise ProceduralMemoryError(f"User data deletion failed: {str(e)}") from e