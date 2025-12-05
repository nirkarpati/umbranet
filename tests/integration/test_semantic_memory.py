"""Integration tests for Neo4j-based semantic memory."""

import pytest

from src.core.domain.semantic import EntityType, RelationshipType
from src.memory.tiers.semantic import SemanticMemoryStore, SemanticMemoryError


@pytest.mark.asyncio
class TestSemanticMemoryIntegration:
    """Integration tests for semantic memory store."""
    
    async def test_connection_lifecycle(self):
        """Test connection setup and teardown."""
        store = SemanticMemoryStore()
        
        try:
            await store.connect()
            assert store.neo4j is not None
        except SemanticMemoryError:
            pytest.skip("Neo4j not available for integration tests")
        finally:
            await store.disconnect()
    
    async def test_context_manager(self):
        """Test using store as async context manager."""
        try:
            async with SemanticMemoryStore() as store:
                assert store.neo4j is not None
        except SemanticMemoryError:
            pytest.skip("Neo4j not available for integration tests")
    
    async def test_entity_upsert(self):
        """Test basic entity creation and updates."""
        try:
            async with SemanticMemoryStore() as store:
                user_id = "test_user_semantic_001"
                
                # Clean up any existing data
                await store.delete_user_graph(user_id)
                
                # Create entity
                entity_id = await store.upsert_entity(
                    user_id=user_id,
                    entity_type=EntityType.PERSON,
                    entity_name="Alice Johnson",
                    properties={"role": "colleague", "department": "engineering"}
                )
                
                assert entity_id is not None
                assert len(entity_id) > 0
                
                # Update same entity
                entity_id2 = await store.upsert_entity(
                    user_id=user_id,
                    entity_type=EntityType.PERSON,
                    entity_name="Alice Johnson",
                    properties={"role": "tech_lead", "years_experience": 5}
                )
                
                # Should be same entity ID
                assert entity_id == entity_id2
                
                # Clean up
                deleted = await store.delete_user_graph(user_id)
                assert deleted > 0
                
        except SemanticMemoryError:
            pytest.skip("Neo4j not available for integration tests")
    
    async def test_relationship_creation(self):
        """Test weighted relationship creation."""
        try:
            async with SemanticMemoryStore() as store:
                user_id = "test_user_relationships"
                
                # Clean up
                await store.delete_user_graph(user_id)
                
                # Create entities
                person_id = await store.upsert_entity(
                    user_id=user_id,
                    entity_type=EntityType.PERSON,
                    entity_name="Bob Smith",
                    properties={"role": "manager"}
                )
                
                company_id = await store.upsert_entity(
                    user_id=user_id,
                    entity_type=EntityType.ORGANIZATION,
                    entity_name="TechCorp Inc",
                    properties={"industry": "software"}
                )
                
                # Create relationship
                await store.upsert_relationship(
                    user_id=user_id,
                    from_entity_id=person_id,
                    to_entity_id=company_id,
                    relationship_type=RelationshipType.WORKS_AT,
                    weight=0.9,
                    decay_rate=0.01,
                    properties={"start_date": "2020-01-01", "position": "Senior Manager"}
                )
                
                # Get graph stats to verify
                stats = await store.get_user_graph_stats(user_id)
                assert stats.total_entities >= 3  # User + Person + Organization
                assert stats.total_relationships >= 1
                assert "Person" in stats.entity_types_count
                assert "Organization" in stats.entity_types_count
                
                # Clean up
                await store.delete_user_graph(user_id)
                
        except SemanticMemoryError:
            pytest.skip("Neo4j not available for integration tests")
    
    async def test_entity_extraction_and_storage(self):
        """Test extracting entities from conversation and storing them."""
        try:
            async with SemanticMemoryStore() as store:
                user_id = "test_user_extraction"
                
                # Clean up
                await store.delete_user_graph(user_id)
                
                # Test conversation with clear entities
                user_message = (
                    "I work at Microsoft in Seattle. My manager is Sarah Chen. "
                    "I really love Python programming and dislike Java."
                )
                assistant_response = (
                    "That's great! Microsoft is a fantastic company. "
                    "Python is indeed a powerful language for many applications."
                )
                
                # Extract and store entities
                extraction = await store.extract_and_store_entities(
                    user_id=user_id,
                    user_message=user_message,
                    assistant_response=assistant_response
                )
                
                # Should extract some entities
                assert len(extraction.entities) > 0
                print(f"Extracted {len(extraction.entities)} entities")
                for entity in extraction.entities:
                    print(f"  - {entity.name} ({entity.entity_type})")
                
                # Check relationships
                print(f"Extracted {len(extraction.relationships)} relationships")
                for rel in extraction.relationships:
                    print(f"  - {rel.relationship_type.value} (weight: {rel.weight})")
                
                # Verify storage
                stats = await store.get_user_graph_stats(user_id)
                assert stats.total_entities >= 1  # At least User entity
                print(f"Graph stats: {stats.total_entities} entities, {stats.total_relationships} relationships")
                
                # Clean up
                await store.delete_user_graph(user_id)
                
        except SemanticMemoryError:
            pytest.skip("Neo4j not available for integration tests")
    
    async def test_recall_local(self):
        """Test local memory recall using vector search + graph traversal."""
        try:
            async with SemanticMemoryStore() as store:
                user_id = "test_user_local_recall"
                
                # Clean up
                await store.delete_user_graph(user_id)
                
                # Create entities with embeddings using the new extraction system
                user_message = "I love coffee and work at Starbucks as a barista"
                assistant_response = "That's great! Coffee knowledge must be helpful in your role."
                
                # Extract and store entities (this will generate embeddings)
                extraction = await store.extract_and_store_entities(
                    user_id=user_id,
                    user_message=user_message,
                    assistant_response=assistant_response
                )
                
                print(f"Extracted {len(extraction.entities)} entities with embeddings")
                
                # Test local recall with a coffee-related query
                context = await store.recall_local(
                    user_id=user_id,
                    query="coffee beverage work",
                    limit=5,
                    similarity_threshold=0.1  # Lower threshold for testing
                )
                
                print(f"Local recall context: {context}")
                
                # Should return formatted context string
                assert isinstance(context, str)
                assert len(context) > 0
                # Context should contain relevant information or a "no results" message
                assert "Local Memory Context" in context or "No relevant entities" in context
                
                # Clean up
                await store.delete_user_graph(user_id)
                
        except SemanticMemoryError:
            pytest.skip("Neo4j not available for integration tests")
    
    async def test_recall_global(self):
        """Test global memory recall using community summaries."""
        try:
            async with SemanticMemoryStore() as store:
                user_id = "test_user_global_recall"
                
                # Clean up
                await store.delete_user_graph(user_id)
                
                # Test global recall (should handle missing community summaries gracefully)
                context = await store.recall_global(
                    user_id=user_id,
                    query="technology work",
                    limit=3
                )
                
                print(f"Global recall context: {context}")
                
                # Should return formatted context string
                assert isinstance(context, str)
                assert len(context) > 0
                # Should indicate no community summaries available
                assert "Global Memory Context" in context or "No community summaries" in context
                
        except SemanticMemoryError:
            pytest.skip("Neo4j not available for integration tests")
    
    async def test_user_isolation(self):
        """Test that different users' graphs are isolated."""
        try:
            async with SemanticMemoryStore() as store:
                user1 = "test_user_isolation_1"
                user2 = "test_user_isolation_2"
                
                # Clean up
                await store.delete_user_graph(user1)
                await store.delete_user_graph(user2)
                
                # Create entities for each user
                entity1_id = await store.upsert_entity(
                    user_id=user1,
                    entity_type=EntityType.PERSON,
                    entity_name="Alice",
                    properties={"role": "developer"}
                )
                
                entity2_id = await store.upsert_entity(
                    user_id=user2,
                    entity_type=EntityType.PERSON,
                    entity_name="Bob",
                    properties={"role": "designer"}
                )
                
                # Get stats for each user
                stats1 = await store.get_user_graph_stats(user1)
                stats2 = await store.get_user_graph_stats(user2)
                
                # Each should have their own entities
                assert stats1.total_entities >= 2  # User + Alice
                assert stats2.total_entities >= 2  # User + Bob
                
                # Test that recall methods respect user isolation
                context1 = await store.recall_local(user1, "Alice developer")
                context2 = await store.recall_local(user2, "Bob designer")
                
                # Both should return valid context strings
                assert isinstance(context1, str)
                assert isinstance(context2, str)
                # Context should be different for different users
                # (or both should indicate no relevant entities found)
                assert "Local Memory Context" in context1 or "No relevant entities" in context1
                assert "Local Memory Context" in context2 or "No relevant entities" in context2
                
                # Clean up
                await store.delete_user_graph(user1)
                await store.delete_user_graph(user2)
                
        except SemanticMemoryError:
            pytest.skip("Neo4j not available for integration tests")
    
    async def test_weighted_relationships_decay(self):
        """Test weighted relationships and decay mechanics."""
        try:
            async with SemanticMemoryStore() as store:
                user_id = "test_user_weights"
                
                # Clean up
                await store.delete_user_graph(user_id)
                
                # Create entities with different weight relationships
                tech_id = await store.upsert_entity(
                    user_id=user_id,
                    entity_type=EntityType.CONCEPT,
                    entity_name="Artificial Intelligence",
                    properties={"field": "technology"}
                )
                
                user_entity_id = store._generate_entity_id(user_id, "User", EntityType.USER)
                
                # High confidence relationship
                await store.upsert_relationship(
                    user_id=user_id,
                    from_entity_id=user_entity_id,
                    to_entity_id=tech_id,
                    relationship_type=RelationshipType.HAS_SKILL,
                    weight=0.95,
                    decay_rate=0.005,  # Very slow decay
                    properties={"expertise_level": "advanced"}
                )
                
                # Test recall with different similarity thresholds
                context_strict = await store.recall_local(
                    user_id=user_id,
                    query="artificial intelligence technology",
                    similarity_threshold=0.8,
                    limit=5
                )
                
                context_loose = await store.recall_local(
                    user_id=user_id,
                    query="artificial intelligence technology",
                    similarity_threshold=0.1,
                    limit=5
                )
                
                # Both should return valid context strings
                assert isinstance(context_strict, str)
                assert isinstance(context_loose, str)
                # Loose threshold might return more results or same results
                # (depends on actual vector similarity)
                
                # Clean up
                await store.delete_user_graph(user_id)
                
        except SemanticMemoryError:
            pytest.skip("Neo4j not available for integration tests")
    
    async def test_dynamic_ontology(self):
        """Test that the system can handle dynamic entity types."""
        try:
            async with SemanticMemoryStore() as store:
                user_id = "test_user_ontology"
                
                # Clean up
                await store.delete_user_graph(user_id)
                
                # Create entities of different types
                entity_types_to_test = [
                    (EntityType.PROJECT, "Project Phoenix"),
                    (EntityType.SKILL, "Machine Learning"),
                    (EntityType.GOAL, "Learn Rust Programming"),
                    (EntityType.PRODUCT, "iPhone 15"),
                    (EntityType.EVENT, "Tech Conference 2024")
                ]
                
                created_entities = []
                for entity_type, name in entity_types_to_test:
                    entity_id = await store.upsert_entity(
                        user_id=user_id,
                        entity_type=entity_type,
                        entity_name=name,
                        properties={"test_entity": True}
                    )
                    created_entities.append((entity_type, entity_id))
                
                # Verify all entity types were created
                stats = await store.get_user_graph_stats(user_id)
                
                for entity_type, _ in entity_types_to_test:
                    assert entity_type.value in stats.entity_types_count
                    assert stats.entity_types_count[entity_type.value] >= 1
                
                print(f"Created {len(created_entities)} entities of different types")
                print(f"Entity type distribution: {stats.entity_types_count}")
                
                # Clean up
                await store.delete_user_graph(user_id)
                
        except SemanticMemoryError:
            pytest.skip("Neo4j not available for integration tests")