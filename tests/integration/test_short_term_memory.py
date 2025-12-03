"""Integration tests for Redis-based short-term memory."""

import asyncio
import pytest
from datetime import datetime

from src.memory.tiers.short_term import ShortTermMemoryClient, RedisConnectionError


@pytest.mark.asyncio
class TestShortTermMemoryIntegration:
    """Integration tests for short-term memory Redis client."""
    
    async def test_connection_lifecycle(self):
        """Test Redis connection setup and teardown."""
        client = ShortTermMemoryClient()
        
        # Test connection
        try:
            await client.connect()
            assert client.redis_client is not None
        except RedisConnectionError:
            pytest.skip("Redis not available for integration tests")
        finally:
            await client.disconnect()
    
    async def test_context_manager(self):
        """Test using client as async context manager."""
        try:
            async with ShortTermMemoryClient() as client:
                assert client.redis_client is not None
        except RedisConnectionError:
            pytest.skip("Redis not available for integration tests")
    
    async def test_add_and_retrieve_context(self):
        """Test basic add turn and context retrieval."""
        try:
            async with ShortTermMemoryClient() as client:
                user_id = "test_user_001"
                
                # Clear any existing session
                await client.clear_session(user_id)
                
                # Add first turn
                await client.add_turn(
                    user_id=user_id,
                    user_message="Hello, what's the weather?",
                    assistant_response="I can help you check the weather. What's your location?"
                )
                
                # Get context
                context = await client.get_context(user_id)
                
                assert len(context.recent_messages) == 1
                assert context.recent_messages[0].user_message == "Hello, what's the weather?"
                assert context.recent_messages[0].assistant_response == "I can help you check the weather. What's your location?"
                assert context.summary is None  # No summary yet
                assert context.token_count > 0
                
                # Add second turn
                await client.add_turn(
                    user_id=user_id,
                    user_message="I'm in Seattle",
                    assistant_response="The weather in Seattle is currently sunny, 72°F."
                )
                
                # Get updated context
                context = await client.get_context(user_id)
                
                assert len(context.recent_messages) == 2
                assert context.recent_messages[1].user_message == "I'm in Seattle"
                
                # Clean up
                await client.clear_session(user_id)
                
        except RedisConnectionError:
            pytest.skip("Redis not available for integration tests")
    
    async def test_session_stats(self):
        """Test session statistics functionality."""
        try:
            async with ShortTermMemoryClient() as client:
                user_id = "test_user_stats"
                
                # Clear session
                await client.clear_session(user_id)
                
                # Add some turns
                await client.add_turn(
                    user_id=user_id,
                    user_message="Test message 1",
                    assistant_response="Test response 1"
                )
                
                await client.add_turn(
                    user_id=user_id,
                    user_message="Test message 2", 
                    assistant_response="Test response 2"
                )
                
                # Get stats
                stats = await client.get_session_stats(user_id)
                
                assert stats["buffer_size"] == 2
                assert stats["has_summary"] is False
                assert stats["total_tokens"] > 0
                assert "last_updated" in stats
                
                # Clean up
                await client.clear_session(user_id)
                
        except RedisConnectionError:
            pytest.skip("Redis not available for integration tests")
    
    async def test_token_budget_management(self):
        """Test token budget and summarization trigger."""
        try:
            # Use very small token budget to trigger summarization quickly
            async with ShortTermMemoryClient(max_token_budget=100) as client:
                user_id = "test_user_budget"
                
                # Clear session
                await client.clear_session(user_id)
                
                # Add many turns to exceed budget
                for i in range(5):
                    await client.add_turn(
                        user_id=user_id,
                        user_message=f"This is a longer test message number {i} to help exceed the token budget quickly",
                        assistant_response=f"This is a longer test response number {i} to help exceed the token budget and trigger summarization"
                    )
                
                # Get context - should have summary now
                context = await client.get_context(user_id)
                
                # Should have fewer messages in buffer due to summarization
                # and possibly a summary (if OpenAI API key is configured)
                assert len(context.recent_messages) < 5
                
                # Clean up
                await client.clear_session(user_id)
                
        except RedisConnectionError:
            pytest.skip("Redis not available for integration tests")
    
    async def test_concurrent_access(self):
        """Test concurrent access to the same user session."""
        try:
            async with ShortTermMemoryClient() as client:
                user_id = "test_user_concurrent"
                
                # Clear session
                await client.clear_session(user_id)
                
                # Simulate concurrent adds
                async def add_turn(turn_id):
                    await client.add_turn(
                        user_id=user_id,
                        user_message=f"Message {turn_id}",
                        assistant_response=f"Response {turn_id}"
                    )
                
                # Run concurrent operations
                await asyncio.gather(*[
                    add_turn(i) for i in range(3)
                ])
                
                # Verify all turns were added
                context = await client.get_context(user_id)
                assert len(context.recent_messages) == 3
                
                # Clean up
                await client.clear_session(user_id)
                
        except RedisConnectionError:
            pytest.skip("Redis not available for integration tests")