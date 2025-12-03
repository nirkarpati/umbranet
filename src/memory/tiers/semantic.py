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
    GraphEntity,
    GraphRelationship,
    GraphStats,
    RelationshipType,
    SemanticFact,
)
from ..database.neo4j_client import Neo4jConnection, get_neo4j_connection
from ..services.entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)


class SemanticMemoryError(Exception):
    """Exception raised for semantic memory operations."""
    pass


class SemanticMemoryStore:
    """Neo4j-based semantic memory store with weighted probabilistic relationships."""
    
    def __init__(self):
        """Initialize semantic memory store."""
        self.neo4j: Neo4jConnection | None = None
        self.entity_extractor = EntityExtractor()
    
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
            user_entity_id = await self.upsert_entity(
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
    
    async def get_related_facts(
        self,
        user_id: str,
        query_text: str,
        max_depth: int = 2,
        min_weight: float = 0.3,
        limit: int = 10
    ) -> list[SemanticFact]:
        """Get semantic facts related to query text via graph traversal.
        
        Args:
            user_id: User identifier
            query_text: Text to find related facts for
            max_depth: Maximum graph traversal depth
            min_weight: Minimum relationship weight
            limit: Maximum facts to return
            
        Returns:
            List of semantic facts
        """
        if not self.neo4j:
            raise SemanticMemoryError("Store not connected")
        
        try:
            # Extract key terms from query for entity matching
            query_terms = [
                term.strip().lower() 
                for term in query_text.split() 
                if len(term.strip()) > 2
            ]
            
            # Build fuzzy entity matching query
            entity_matches = []
            for term in query_terms[:5]:  # Limit to avoid overly complex queries
                entity_matches.append(f"toLower(e.name) CONTAINS '{term}'")
            
            if not entity_matches:
                return []
            
            # Query for related facts through graph traversal
            query = f"""
            MATCH (e {{user_id: $user_id}})
            WHERE {' OR '.join(entity_matches)}
            MATCH (e)-[r*1..{max_depth}]-(related {{user_id: $user_id}})
            WHERE ALL(rel in r WHERE rel.weight >= $min_weight)
            WITH e, related, r[-1] as last_rel
            WHERE e <> related
            RETURN DISTINCT
                e.entity_id as subject_id,
                e.name as subject_name,
                labels(e) as subject_labels,
                e as subject_props,
                type(last_rel) as relationship_type,
                last_rel.weight as relationship_weight,
                last_rel as relationship_props,
                related.entity_id as object_id,
                related.name as object_name,
                labels(related) as object_labels,
                related as object_props
            ORDER BY last_rel.weight DESC
            LIMIT $limit
            """
            
            results = await self.neo4j.execute_query(query, {
                "user_id": user_id,
                "min_weight": min_weight,
                "limit": limit
            })
            
            # Convert to SemanticFact objects
            facts = []
            for row in results:
                # Create subject entity
                subject = GraphEntity(
                    entity_id=row["subject_id"],
                    entity_type=EntityType(row["subject_labels"][0]),
                    name=row["subject_name"],
                    user_id=user_id,
                    properties=dict(row["subject_props"])
                )
                
                # Create object entity
                object_entity = GraphEntity(
                    entity_id=row["object_id"],
                    entity_type=EntityType(row["object_labels"][0]),
                    name=row["object_name"],
                    user_id=user_id,
                    properties=dict(row["object_props"])
                )
                
                # Create relationship
                relationship = GraphRelationship(
                    from_entity_id=row["subject_id"],
                    to_entity_id=row["object_id"],
                    relationship_type=RelationshipType(row["relationship_type"]),
                    weight=float(row["relationship_weight"]),
                    user_id=user_id,
                    properties=dict(row["relationship_props"])
                )
                
                facts.append(SemanticFact(
                    subject=subject,
                    relationship=relationship,
                    object=object_entity,
                    confidence=float(row["relationship_weight"]),
                    context={"query_terms": query_terms}
                ))
            
            logger.debug(f"Retrieved {len(facts)} semantic facts for query: {query_text}")
            return facts
            
        except Exception as e:
            logger.error(f"Failed to get related facts: {str(e)}")
            raise SemanticMemoryError(f"Fact retrieval failed: {str(e)}") from e
    
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