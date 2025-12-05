"""Semantic memory implementation using Neo4j weighted probabilistic knowledge graph.

This module implements Tier 3 of the RAG++ memory hierarchy, providing:
- Dynamic ontology with schema-on-write
- Weighted probabilistic relationships
- Entity extraction and graph construction
- Semantic fact retrieval via graph traversal
"""

import logging
import uuid
from typing import Any

from ...core.domain.semantic import (
    EntityExtractionResult,
    EntityType,
    GraphStats,
    RelationshipType,
)
from ...core.embeddings.base import EmbeddingProvider
from ...core.embeddings.provider_factory import get_embedding_provider
from ..database.neo4j_client import Neo4jConnection, get_neo4j_connection
from ..services.entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)


class SemanticMemoryError(Exception):
    """Exception raised for semantic memory operations."""
    pass


class SemanticMemoryStore:
    """Neo4j-based semantic memory store with weighted probabilistic relationships."""
    
    def __init__(self, embedding_provider: EmbeddingProvider | None = None):
        """Initialize semantic memory store.
        
        Args:
            embedding_provider: Optional embedding provider for vector search
        """
        self.neo4j: Neo4jConnection | None = None
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.entity_extractor = EntityExtractor(self.embedding_provider)
    
    async def __aenter__(self) -> "SemanticMemoryStore":
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self) -> None:
        """Initialize connections to Neo4j and entity extractor."""
        try:
            self.neo4j = await get_neo4j_connection()
            await self.neo4j.initialize_schema()
            
            # Ensure user core entities exist
            await self._ensure_core_entities()
            
            logger.info("Connected to semantic memory store")
        except Exception as e:
            logger.error(f"Failed to connect to semantic memory store: {str(e)}")
            raise SemanticMemoryError(f"Connection failed: {str(e)}") from e
    
    async def disconnect(self) -> None:
        """Close connections."""
        logger.info("Disconnected from semantic memory store")
    
    async def _ensure_core_entities(self) -> None:
        """Ensure SystemAgent core entity exists."""
        if not self.neo4j:
            return
        
        # Create SystemAgent node
        query = """
        MERGE (agent:SystemAgent {entity_id: 'system_agent'})
        SET agent.name = 'Governor System',
            agent.created_at = datetime(),
            agent.last_updated = datetime()
        """
        await self.neo4j.execute_write_query(query)
    
    def _generate_entity_id(self, user_id: str, name: str, entity_type: EntityType) -> str:
        """Generate consistent entity ID."""
        key = f"{user_id}:{entity_type.value}:{name.lower().strip()}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))
    
    async def upsert_entity(
        self,
        user_id: str,
        entity_type: EntityType,
        entity_name: str,
        properties: dict[str, Any] | None = None
    ) -> str:
        """Create or update an entity in the semantic graph.
        
        Args:
            user_id: User this entity belongs to
            entity_type: Type of the entity
            entity_name: Name of the entity
            properties: Additional properties
            
        Returns:
            Entity ID
            
        Raises:
            SemanticMemoryError: If upsert fails
        """
        if not self.neo4j:
            raise SemanticMemoryError("Store not connected - use async context manager")
        
        try:
            entity_id = self._generate_entity_id(user_id, entity_name, entity_type)
            
            # Use MERGE to create or update with compatible syntax
            query = f"""
            MERGE (e:{entity_type.value} {{entity_id: $entity_id, user_id: $user_id}})
            ON CREATE SET 
                e.name = $name,
                e.created_at = datetime(),
                e.last_updated = datetime(),
                e += $properties
            ON MATCH SET 
                e.name = $name,
                e.last_updated = datetime(),
                e += $properties
            RETURN e.entity_id as entity_id
            """
            
            result = await self.neo4j.execute_write_query(query, {
                "entity_id": entity_id,
                "user_id": user_id,
                "name": entity_name,
                "properties": properties or {}
            })
            
            if not result:
                raise SemanticMemoryError("Entity upsert returned no results")
            
            logger.debug(f"Upserted entity {entity_id} for user {user_id}")
            return entity_id
            
        except Exception as e:
            logger.error(f"Failed to upsert entity {entity_name}: {str(e)}")
            raise SemanticMemoryError(f"Entity upsert failed: {str(e)}") from e
    
    async def upsert_relationship(
        self,
        user_id: str,
        from_entity_id: str,
        to_entity_id: str,
        relationship_type: RelationshipType,
        weight: float = 0.5,
        decay_rate: float = 0.01,
        properties: dict[str, Any] | None = None
    ) -> None:
        """Create or update a weighted relationship between entities.
        
        Args:
            user_id: User this relationship belongs to
            from_entity_id: Source entity ID
            to_entity_id: Target entity ID
            relationship_type: Type of relationship
            weight: Confidence weight (0.0 to 1.0)
            decay_rate: How fast this relationship decays
            properties: Additional properties
        """
        if not self.neo4j:
            raise SemanticMemoryError("Store not connected")
        
        try:
            # MERGE relationship with weight updates
            query = f"""
            MATCH (from {{entity_id: $from_id, user_id: $user_id}})
            MATCH (to {{entity_id: $to_id, user_id: $user_id}})
            MERGE (from)-[r:{relationship_type.value}]->(to)
            ON CREATE SET 
                r.weight = $weight,
                r.decay_rate = $decay_rate,
                r.created_at = datetime(),
                r.last_verified = datetime(),
                r.user_id = $user_id,
                r += $properties
            ON MATCH SET
                r.weight = $weight,
                r.decay_rate = $decay_rate,
                r.last_verified = datetime(),
                r += $properties
            """
            
            await self.neo4j.execute_write_query(query, {
                "from_id": from_entity_id,
                "to_id": to_entity_id,
                "user_id": user_id,
                "weight": max(0.0, min(1.0, weight)),  # Clamp to valid range
                "decay_rate": max(0.0, min(1.0, decay_rate)),
                "properties": properties or {}
            })
            
            logger.debug(
                f"Upserted relationship {relationship_type.value} "
                f"from {from_entity_id} to {to_entity_id}"
            )
            
        except Exception as e:
            logger.error(f"Failed to upsert relationship: {str(e)}")
            raise SemanticMemoryError(f"Relationship upsert failed: {str(e)}") from e
    
    async def extract_and_store_entities(
        self,
        user_id: str,
        user_message: str,
        assistant_response: str
    ) -> EntityExtractionResult:
        """Extract entities from conversation and store in graph.
        
        Args:
            user_id: User identifier
            user_message: User's message
            assistant_response: Assistant's response
            
        Returns:
            Extraction results
        """
        try:
            # Extract entities using the entity extractor
            async with self.entity_extractor as extractor:
                extraction = await extractor.extract_from_conversation(
                    user_message, assistant_response, user_id
                )
            
            # Ensure user entity exists
            await self.upsert_entity(
                user_id=user_id,
                entity_type=EntityType.USER,
                entity_name="User",
                properties={"is_primary_user": True}
            )
            
            # Store extracted entities
            for entity in extraction.entities:
                await self.upsert_entity(
                    user_id=user_id,
                    entity_type=entity.entity_type,
                    entity_name=entity.name,
                    properties=entity.properties
                )
            
            # Store relationships
            for relationship in extraction.relationships:
                await self.upsert_relationship(
                    user_id=user_id,
                    from_entity_id=relationship.from_entity_id,
                    to_entity_id=relationship.to_entity_id,
                    relationship_type=relationship.relationship_type,
                    weight=relationship.weight,
                    decay_rate=relationship.decay_rate,
                    properties=relationship.properties
                )
            
            logger.debug(
                f"Stored {len(extraction.entities)} entities and "
                f"{len(extraction.relationships)} relationships for user {user_id}"
            )
            
            return extraction
            
        except Exception as e:
            logger.error(f"Failed to extract and store entities: {str(e)}")
            raise SemanticMemoryError(f"Entity extraction failed: {str(e)}") from e
    
    async def _vector_search_entities(
        self,
        query_embedding: list[float],
        user_id: str,
        limit: int = 5,
        similarity_threshold: float = 0.7
    ) -> list[dict[str, Any]]:
        """Search for similar entities using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            user_id: User identifier for tenant isolation
            limit: Number of similar entities to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar entities with similarity scores
        """
        if not self.neo4j:
            raise SemanticMemoryError("Store not connected")
        
        try:
            results = await self.neo4j.query_vector_index(
                query_embedding=query_embedding,
                k=limit,
                user_id=user_id,
                similarity_threshold=similarity_threshold
            )
            
            logger.debug(f"Found {len(results)} similar entities via vector search")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            raise SemanticMemoryError(f"Vector search failed: {str(e)}") from e
    
    async def recall_local(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        similarity_threshold: float = 0.7
    ) -> str:
        """Perform local recall using vector search + graph traversal.
        
        This method:
        1. Embeds the query
        2. Finds top similar entities (anchors) via vector search
        3. Performs graph traversal to get anchors + 1-hop neighbors
        4. Formats the subgraph as text context
        
        Args:
            user_id: User identifier
            query: Query text
            limit: Number of anchor entities to find
            similarity_threshold: Minimum similarity for anchor selection
            
        Returns:
            Formatted text context of the local subgraph
        """
        if not self.neo4j:
            raise SemanticMemoryError("Store not connected")
        
        try:
            # Step 1: Embed the query
            query_embedding = await self.embedding_provider.embed_text(query)
            
            # Step 2: Find similar anchor entities
            anchor_results = await self._vector_search_entities(
                query_embedding=query_embedding,
                user_id=user_id,
                limit=limit,
                similarity_threshold=similarity_threshold
            )
            
            if not anchor_results:
                return "No relevant entities found in local memory."
            
            # Extract anchor entity IDs
            anchor_ids = [result["entity"]["entity_id"] for result in anchor_results]
            
            # Step 3: Get anchors + 1-hop neighbors + relationships
            subgraph_query = """
            MATCH (anchor {user_id: $user_id})
            WHERE anchor.entity_id IN $anchor_ids
            
            // Get anchors and their 1-hop neighbors
            OPTIONAL MATCH (anchor)-[r]-(neighbor {user_id: $user_id})
            WHERE r.weight >= 0.3
            
            // Collect all entities and relationships
            WITH anchor, collect(DISTINCT neighbor) as neighbors, 
                 collect(DISTINCT {rel: r, neighbor: neighbor}) as relationships
            
            RETURN 
                anchor.entity_id as anchor_id,
                anchor.name as anchor_name,
                labels(anchor)[0] as anchor_type,
                anchor.properties as anchor_props,
                [
                    n IN neighbors WHERE n IS NOT NULL | {
                        id: n.entity_id,
                        name: n.name,
                        type: labels(n)[0],
                        properties: n.properties
                    }
                ] as neighbor_entities,
                [
                    rel_data IN relationships WHERE rel_data.neighbor IS NOT NULL | {
                        type: type(rel_data.rel),
                        weight: rel_data.rel.weight,
                        target_name: rel_data.neighbor.name
                    }
                ] as neighbor_relationships
            """
            
            subgraph_results = await self.neo4j.execute_query(subgraph_query, {
                "user_id": user_id,
                "anchor_ids": anchor_ids
            })
            
            # Step 4: Format as text context
            context_parts = ["## Local Memory Context\n"]
            
            for result in subgraph_results:
                anchor_name = result["anchor_name"]
                anchor_type = result["anchor_type"]
                
                context_parts.append(f"**{anchor_name}** ({anchor_type})")
                
                # Add relationships
                relationships = result["neighbor_relationships"]
                for rel in relationships:
                    rel_type = rel["type"].replace("_", " ").title()
                    target = rel["target_name"]
                    weight = rel["weight"]
                    context_parts.append(
                        f"  - {rel_type}: {target} (confidence: {weight:.2f})"
                    )
                
                context_parts.append("")  # Empty line between entities
            
            if len(context_parts) <= 2:  # Only header + empty parts
                return "No detailed relationships found in local memory."
            
            formatted_context = "\n".join(context_parts).strip()
            logger.debug(f"Generated local recall context for query: {query}")
            return formatted_context
            
        except Exception as e:
            logger.error(f"Local recall failed: {str(e)}")
            raise SemanticMemoryError(f"Local recall failed: {str(e)}") from e
    
    async def recall_global(
        self,
        user_id: str,
        query: str,
        limit: int = 3,
        similarity_threshold: float = 0.6
    ) -> str:
        """Perform global recall using community summaries.
        
        This method:
        1. Retrieves all CommunitySummary nodes for the user
        2. If embeddings exist, filters by vector similarity to query
        3. Returns the summary text
        
        Args:
            user_id: User identifier
            query: Query text
            limit: Maximum number of community summaries to return
            similarity_threshold: Minimum similarity for community selection
            
        Returns:
            Formatted community summary text
        """
        if not self.neo4j:
            raise SemanticMemoryError("Store not connected")
        
        try:
            # For now, retrieve all CommunitySummary nodes
            # TODO: Implement community detection and storage
            community_query = """
            MATCH (c:CommunitySummary {user_id: $user_id})
            RETURN c.community_id as id,
                   c.level as level,
                   c.summary as summary,
                   c.embedding as embedding
            ORDER BY c.level, c.community_id
            LIMIT $limit
            """
            
            community_results = await self.neo4j.execute_query(community_query, {
                "user_id": user_id,
                "limit": limit
            })
            
            if not community_results:
                return "No community summaries available in global memory."
            
            # If query provided and communities have embeddings, filter by similarity
            if query and any(result.get("embedding") for result in community_results):
                try:
                    query_embedding = await self.embedding_provider.embed_text(query)
                    
                    # Calculate similarity scores for communities with embeddings
                    scored_communities: list[tuple[float, dict[str, Any]]] = []
                    for result in community_results:
                        if result.get("embedding"):
                            # Calculate cosine similarity
                            community_embedding = result["embedding"]
                            similarity = self._cosine_similarity(query_embedding, community_embedding)
                            
                            if similarity >= similarity_threshold:
                                scored_communities.append((similarity, result))
                    
                    # Sort by similarity and take top results
                    scored_communities.sort(key=lambda x: x[0], reverse=True)
                    community_results = [result for _, result in scored_communities[:limit]]
                    
                except Exception as e:
                    logger.warning(f"Failed to filter communities by similarity: {e}")
                    # Continue with unfiltered results
            
            # Format community summaries
            context_parts = ["## Global Memory Context\n"]
            
            for i, result in enumerate(community_results, 1):
                summary = result["summary"]
                level = result.get("level", 0)
                
                context_parts.append(f"**Community {i}** (Level {level})")
                context_parts.append(f"{summary}")
                context_parts.append("")  # Empty line between communities
            
            if len(context_parts) <= 2:  # Only header
                return "No community summaries found in global memory."
            
            formatted_context = "\n".join(context_parts).strip()
            logger.debug(f"Generated global recall context for query: {query}")
            return formatted_context
            
        except Exception as e:
            logger.error(f"Global recall failed: {str(e)}")
            raise SemanticMemoryError(f"Global recall failed: {str(e)}") from e
    
    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if len(vec1) != len(vec2) or not vec1 or not vec2:
            return 0.0
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=True))
        
        # Calculate magnitudes
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0.0 or magnitude2 == 0.0:
            return 0.0
        
        # Cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)
        
        # Ensure result is in [0, 1] range (convert from [-1, 1])
        return max(0.0, similarity)
    
    
    async def get_user_graph_stats(self, user_id: str) -> GraphStats:
        """Get statistics about user's semantic graph.
        
        Args:
            user_id: User identifier
            
        Returns:
            Graph statistics
        """
        if not self.neo4j:
            raise SemanticMemoryError("Store not connected")
        
        try:
            # Count entities by type
            entity_query = """
            MATCH (n {user_id: $user_id})
            RETURN labels(n)[0] as entity_type, count(*) as count
            """
            
            # Count relationships by type
            relationship_query = """
            MATCH ({user_id: $user_id})-[r]-({user_id: $user_id})
            RETURN type(r) as relationship_type, 
                   count(*) as count,
                   avg(r.weight) as avg_weight
            """
            
            entity_results = await self.neo4j.execute_query(entity_query, {"user_id": user_id})
            rel_results = await self.neo4j.execute_query(relationship_query, {"user_id": user_id})
            
            # Build statistics
            entity_counts = {row["entity_type"]: row["count"] for row in entity_results}
            rel_counts = {row["relationship_type"]: row["count"] for row in rel_results}
            
            total_entities = sum(entity_counts.values())
            total_relationships = sum(rel_counts.values())
            avg_weight = (
                sum(row["avg_weight"] or 0 for row in rel_results) / len(rel_results)
                if rel_results else 0.0
            )
            
            return GraphStats(
                total_entities=total_entities,
                total_relationships=total_relationships,
                entity_types_count=entity_counts,
                relationship_types_count=rel_counts,
                average_weight=avg_weight
            )
            
        except Exception as e:
            logger.error(f"Failed to get graph stats: {str(e)}")
            return GraphStats()
    
    async def get_entities_for_user(self, user_id: str) -> list[dict]:
        """Get all entities for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of entity dictionaries
        """
        if not self.neo4j:
            raise SemanticMemoryError("Store not connected")
        
        try:
            query = """
            MATCH (n {user_id: $user_id})
            RETURN n.name as name, 
                   COALESCE(n.original_type, n.type, labels(n)[0]) as type, 
                   n.entity_id as entity_id,
                   n.properties as properties
            ORDER BY n.created_at DESC
            """
            
            results = await self.neo4j.execute_query(query, {"user_id": user_id})
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "name": result.get("name"),
                    "type": result.get("type"),
                    "entity_id": result.get("entity_id")
                })
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to get entities for user {user_id}: {str(e)}")
            return []
    
    async def get_relationships_for_user(self, user_id: str) -> list[dict]:
        """Get all relationships for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of relationship dictionaries
        """
        if not self.neo4j:
            raise SemanticMemoryError("Store not connected")
        
        try:
            query = """
            MATCH (a {user_id: $user_id})-[r]->(b {user_id: $user_id})
            RETURN type(r) as relationship_type, 
                   COALESCE(r.original_relationship, type(r)) as display_relationship,
                   a.name as from_entity, 
                   b.name as to_entity, 
                   r.weight as weight,
                   r.properties as properties
            ORDER BY r.weight DESC
            """
            
            results = await self.neo4j.execute_query(query, {"user_id": user_id})
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "relationship_type": result.get("display_relationship", result.get("relationship_type")),
                    "from_entity": result.get("from_entity"),
                    "to_entity": result.get("to_entity"),
                    "weight": result.get("weight", 0.5)
                })
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to get relationships for user {user_id}: {str(e)}")
            return []

    async def delete_user_graph(self, user_id: str) -> int:
        """Delete all graph data for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of deleted nodes and relationships
        """
        if not self.neo4j:
            raise SemanticMemoryError("Store not connected")
        
        try:
            # Delete all user data
            query = """
            MATCH (n {user_id: $user_id})
            OPTIONAL MATCH (n)-[r]-()
            DELETE n, r
            RETURN count(n) as deleted_count
            """
            
            results = await self.neo4j.execute_write_query(query, {"user_id": user_id})
            deleted_count = results[0]["deleted_count"] if results else 0
            
            logger.info(f"Deleted {deleted_count} graph elements for user {user_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete user graph: {str(e)}")
            raise SemanticMemoryError(f"Graph deletion failed: {str(e)}") from e