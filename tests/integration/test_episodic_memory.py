"""Integration tests for episodic memory with PGVector."""

import pytest
from datetime import datetime

from src.core.embeddings.local_provider import LocalEmbeddingProvider
from src.core.embeddings.provider_factory import ProviderType, get_embedding_provider
from src.memory.tiers.episodic import EpisodicMemoryStore, EpisodicMemoryError


@pytest.mark.asyncio
class TestEpisodicMemoryIntegration:
    """Integration tests for episodic memory store."""
    
    async def test_connection_lifecycle(self):
        """Test connection setup and teardown."""
        store = EpisodicMemoryStore()
        
        try:
            await store.connect()
            assert store.postgres is not None
        except EpisodicMemoryError:
            pytest.skip("PostgreSQL not available for integration tests")
        finally:
            await store.disconnect()
    
    async def test_context_manager(self):
        """Test using store as async context manager."""
        try:
            async with EpisodicMemoryStore() as store:
                assert store.postgres is not None
        except EpisodicMemoryError:
            pytest.skip("PostgreSQL not available for integration tests")
    
    async def test_log_and_recall_interaction(self):
        """Test basic interaction logging and recall."""
        try:
            # Use local embeddings to avoid API key requirements
            local_provider = LocalEmbeddingProvider()
            async with EpisodicMemoryStore(local_provider) as store:
                user_id = "test_user_episodic_001"
                
                # Clean up any existing data
                await store.delete_user_data(user_id)
                
                # Log an interaction
                episode_id = await store.log_interaction(
                    user_id=user_id,
                    user_message="What's the weather like today?",
                    assistant_response="I can help you check the weather. What's your location?",
                    metadata={"channel": "telegram", "test": True}
                )
                
                assert episode_id is not None
                assert len(episode_id) > 0
                
                # Check interaction count
                count = await store.get_interaction_count(user_id)
                assert count == 1
                
                # Recall similar interactions
                memories = await store.recall(
                    user_id=user_id,
                    query_text="weather information",
                    limit=5,
                    similarity_threshold=0.3  # Lower threshold for testing
                )
                
                assert len(memories) >= 1
                assert memories[0].user_id == user_id
                assert "weather" in memories[0].content.lower()
                assert "test" in memories[0].metadata
                assert memories[0].metadata["episode_id"] == episode_id
                
                # Clean up
                deleted = await store.delete_user_data(user_id)
                assert deleted == 1
                
        except (EpisodicMemoryError, ImportError):
            pytest.skip("PostgreSQL or sentence-transformers not available")
    
    async def test_multiple_interactions_and_similarity_search(self):
        """Test logging multiple interactions and similarity search."""
        try:
            local_provider = LocalEmbeddingProvider()
            async with EpisodicMemoryStore(local_provider) as store:
                user_id = "test_user_similarity"
                
                # Clean up
                await store.delete_user_data(user_id)
                
                # Log several different types of interactions
                interactions = [
                    ("What's the weather?", "I can check the weather for you."),
                    ("Book a flight to Paris", "I'll help you find flights to Paris."),
                    ("Is it raining today?", "Let me check the current weather conditions."),
                    ("Schedule a meeting", "I can help schedule your meeting."),
                    ("What's the forecast?", "I'll get the weather forecast for you.")
                ]
                
                episode_ids = []
                for i, (user_msg, assistant_msg) in enumerate(interactions):
                    episode_id = await store.log_interaction(
                        user_id=user_id,
                        user_message=user_msg,
                        assistant_response=assistant_msg,
                        metadata={"interaction_num": i}
                    )
                    episode_ids.append(episode_id)
                
                # Test similarity search for weather-related queries
                weather_memories = await store.recall(
                    user_id=user_id,
                    query_text="weather conditions today",
                    limit=10,
                    similarity_threshold=0.1
                )
                
                # Should find weather-related interactions
                assert len(weather_memories) >= 2
                weather_content = " ".join([m.content.lower() for m in weather_memories])
                assert "weather" in weather_content
                
                # Test recall for flight booking
                flight_memories = await store.recall(
                    user_id=user_id,
                    query_text="book flight travel",
                    limit=5,
                    similarity_threshold=0.1
                )
                
                # Should find travel-related content
                assert len(flight_memories) >= 1
                
                # Test recent interactions
                recent = await store.get_recent_interactions(user_id, limit=3)
                assert len(recent) == 3
                assert recent[0].timestamp >= recent[1].timestamp  # Ordered by timestamp
                
                # Verify total count
                total_count = await store.get_interaction_count(user_id)
                assert total_count == len(interactions)
                
                # Clean up
                await store.delete_user_data(user_id)
                
        except (EpisodicMemoryError, ImportError):
            pytest.skip("PostgreSQL or sentence-transformers not available")
    
    async def test_embedding_provider_selection(self):
        """Test different embedding providers."""
        try:
            # Test with local provider explicitly
            local_provider = LocalEmbeddingProvider()
            async with EpisodicMemoryStore(local_provider) as store:
                user_id = "test_provider_local"
                await store.delete_user_data(user_id)
                
                episode_id = await store.log_interaction(
                    user_id=user_id,
                    user_message="Test message",
                    assistant_response="Test response"
                )
                
                memories = await store.recall(
                    user_id=user_id,
                    query_text="test",
                    limit=1,
                    similarity_threshold=0.1
                )
                
                assert len(memories) == 1
                assert memories[0].metadata["episode_id"] == episode_id
                
                await store.delete_user_data(user_id)
                
        except (EpisodicMemoryError, ImportError):
            pytest.skip("PostgreSQL or sentence-transformers not available")
    
    async def test_metadata_handling(self):
        """Test metadata storage and retrieval."""
        try:
            local_provider = LocalEmbeddingProvider()
            async with EpisodicMemoryStore(local_provider) as store:
                user_id = "test_metadata"
                await store.delete_user_data(user_id)
                
                complex_metadata = {
                    "channel": "telegram",
                    "message_id": 12345,
                    "location": {"lat": 47.6062, "lng": -122.3321},
                    "tool_used": "weather_api",
                    "confidence": 0.95,
                    "tags": ["weather", "seattle"]
                }
                
                episode_id = await store.log_interaction(
                    user_id=user_id,
                    user_message="Weather in Seattle?",
                    assistant_response="The weather in Seattle is sunny.",
                    metadata=complex_metadata
                )
                
                memories = await store.recall(
                    user_id=user_id,
                    query_text="seattle weather",
                    limit=1,
                    similarity_threshold=0.1
                )
                
                assert len(memories) == 1
                memory = memories[0]
                
                # Check that all metadata is preserved
                assert memory.metadata["channel"] == "telegram"
                assert memory.metadata["message_id"] == 12345
                assert memory.metadata["location"]["lat"] == 47.6062
                assert memory.metadata["tool_used"] == "weather_api"
                assert memory.metadata["confidence"] == 0.95
                assert "weather" in memory.metadata["tags"]
                assert "seattle" in memory.metadata["tags"]
                
                await store.delete_user_data(user_id)
                
        except (EpisodicMemoryError, ImportError):
            pytest.skip("PostgreSQL or sentence-transformers not available")
    
    async def test_user_isolation(self):
        """Test that different users' data is properly isolated."""
        try:
            local_provider = LocalEmbeddingProvider()
            async with EpisodicMemoryStore(local_provider) as store:
                user1 = "test_user_1"
                user2 = "test_user_2"
                
                # Clean up
                await store.delete_user_data(user1)
                await store.delete_user_data(user2)
                
                # Log interactions for both users
                await store.log_interaction(
                    user_id=user1,
                    user_message="User 1 message",
                    assistant_response="Response for user 1"
                )
                
                await store.log_interaction(
                    user_id=user2,
                    user_message="User 2 message", 
                    assistant_response="Response for user 2"
                )
                
                # Each user should only see their own data
                user1_memories = await store.recall(
                    user_id=user1,
                    query_text="user message",
                    limit=10,
                    similarity_threshold=0.1
                )
                
                user2_memories = await store.recall(
                    user_id=user2,
                    query_text="user message",
                    limit=10,
                    similarity_threshold=0.1
                )
                
                assert len(user1_memories) == 1
                assert len(user2_memories) == 1
                assert "user 1" in user1_memories[0].content.lower()
                assert "user 2" in user2_memories[0].content.lower()
                
                # Verify counts
                assert await store.get_interaction_count(user1) == 1
                assert await store.get_interaction_count(user2) == 1
                
                # Clean up
                await store.delete_user_data(user1)
                await store.delete_user_data(user2)
                
        except (EpisodicMemoryError, ImportError):
            pytest.skip("PostgreSQL or sentence-transformers not available")