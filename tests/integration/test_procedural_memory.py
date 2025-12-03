"""Integration tests for procedural memory (profiles and instructions)."""

import pytest

from src.core.domain.procedural import (
    BehavioralInstruction,
    InstructionCategory,
    ProfileCategory,
    ProfileEntry,
    UserProfile,
)
from src.memory.tiers.procedural import ProceduralMemoryStore, ProceduralMemoryError


@pytest.mark.asyncio
class TestProceduralMemoryIntegration:
    """Integration tests for procedural memory store."""
    
    async def test_connection_lifecycle(self):
        """Test connection setup and teardown."""
        store = ProceduralMemoryStore()
        
        try:
            await store.connect()
            assert store.postgres is not None
        except ProceduralMemoryError:
            pytest.skip("PostgreSQL not available for integration tests")
        finally:
            await store.disconnect()
    
    async def test_context_manager(self):
        """Test using store as async context manager."""
        try:
            async with ProceduralMemoryStore() as store:
                assert store.postgres is not None
        except ProceduralMemoryError:
            pytest.skip("PostgreSQL not available for integration tests")
    
    async def test_profile_operations(self):
        """Test basic profile CRUD operations."""
        try:
            async with ProceduralMemoryStore() as store:
                user_id = "test_user_profile_001"
                
                # Clean up any existing data
                await store.delete_all_user_data(user_id)
                
                # Set profile values
                await store.set_profile_value(
                    user_id=user_id,
                    category=ProfileCategory.PERSONAL,
                    key="name",
                    value="Alice Johnson"
                )
                
                await store.set_profile_value(
                    user_id=user_id,
                    category=ProfileCategory.TIMEZONE,
                    key="primary",
                    value="America/Los_Angeles"
                )
                
                await store.set_profile_value(
                    user_id=user_id,
                    category=ProfileCategory.PREFERENCES,
                    key="response_format",
                    value="brief"
                )
                
                # Test single value retrieval
                name = await store.get_profile_value(user_id, ProfileCategory.PERSONAL, "name")
                assert name == "Alice Johnson"
                
                timezone = await store.get_profile_value(user_id, ProfileCategory.TIMEZONE, "primary")
                assert timezone == "America/Los_Angeles"
                
                # Test complete profile retrieval
                profile = await store.get_user_profile(user_id)
                assert isinstance(profile, UserProfile)
                assert profile.user_id == user_id
                assert len(profile.entries) == 3
                
                # Test profile methods
                assert profile.get_value(ProfileCategory.PERSONAL, "name") == "Alice Johnson"
                assert profile.get_value(ProfileCategory.PREFERENCES, "response_format") == "brief"
                assert profile.get_value(ProfileCategory.PERSONAL, "nonexistent") is None
                
                # Test profile update (upsert)
                await store.set_profile_value(
                    user_id=user_id,
                    category=ProfileCategory.PERSONAL,
                    key="name",
                    value="Alice Smith"  # Updated name
                )
                
                updated_name = await store.get_profile_value(user_id, ProfileCategory.PERSONAL, "name")
                assert updated_name == "Alice Smith"
                
                # Test profile deletion
                deleted = await store.delete_profile_value(user_id, ProfileCategory.PREFERENCES, "response_format")
                assert deleted is True
                
                deleted_value = await store.get_profile_value(user_id, ProfileCategory.PREFERENCES, "response_format")
                assert deleted_value is None
                
                # Clean up
                await store.delete_all_user_data(user_id)
                
        except ProceduralMemoryError:
            pytest.skip("PostgreSQL not available for integration tests")
    
    async def test_instruction_operations(self):
        """Test behavioral instruction CRUD operations."""
        try:
            async with ProceduralMemoryStore() as store:
                user_id = "test_user_instructions"
                
                # Clean up
                await store.delete_all_user_data(user_id)
                
                # Add instruction
                instruction_id = await store.add_instruction(
                    user_id=user_id,
                    category=InstructionCategory.COMMUNICATION_STYLE,
                    title="Keep responses brief",
                    instruction="Always provide concise, to-the-point responses. Avoid verbose explanations unless specifically requested.",
                    confidence=0.9,
                    priority=8,
                    examples=["User asks for weather", "User requests simple facts"],
                    exceptions=["User asks for detailed explanation", "Complex technical topics"],
                    metadata={"source": "user_preference", "created_by": "onboarding"}
                )
                
                assert instruction_id is not None
                assert len(instruction_id) > 0
                
                # Add another instruction
                instruction_id2 = await store.add_instruction(
                    user_id=user_id,
                    category=InstructionCategory.TRAVEL_BOOKING,
                    title="Prefer direct flights",
                    instruction="When booking flights, prioritize direct flights over connecting flights, even if slightly more expensive.",
                    confidence=0.8,
                    priority=6
                )
                
                # Test instruction retrieval
                all_instructions = await store.get_all_instructions(user_id)
                assert len(all_instructions) == 2
                
                # Verify instruction details
                brief_instruction = next(
                    (instr for instr in all_instructions if "brief" in instr.title.lower()), 
                    None
                )
                assert brief_instruction is not None
                assert brief_instruction.confidence == 0.9
                assert brief_instruction.priority == 8
                assert len(brief_instruction.examples) == 2
                assert len(brief_instruction.exceptions) == 2
                
                # Test relevant instruction search
                relevant_comm = await store.get_relevant_instructions(
                    user_id=user_id,
                    query_text="how should I respond to the user",
                    categories=[InstructionCategory.COMMUNICATION_STYLE],
                    min_confidence=0.5,
                    limit=5
                )
                
                assert len(relevant_comm) >= 1
                assert any("brief" in instr.instruction.title.lower() for instr in relevant_comm)
                
                # Test travel-related search
                relevant_travel = await store.get_relevant_instructions(
                    user_id=user_id,
                    query_text="book flight to Paris",
                    categories=[InstructionCategory.TRAVEL_BOOKING],
                    min_confidence=0.5,
                    limit=5
                )
                
                assert len(relevant_travel) >= 1
                assert any("flight" in instr.instruction.instruction.lower() for instr in relevant_travel)
                
                # Test instruction update
                updated = await store.update_instruction(
                    user_id=user_id,
                    instruction_id=instruction_id,
                    confidence=0.95,
                    priority=9
                )
                assert updated is True
                
                # Verify update
                updated_instructions = await store.get_all_instructions(user_id)
                updated_brief = next(
                    (instr for instr in updated_instructions if instr.instruction_id == instruction_id),
                    None
                )
                assert updated_brief.confidence == 0.95
                assert updated_brief.priority == 9
                
                # Test instruction deletion
                deleted = await store.delete_instruction(user_id, instruction_id2)
                assert deleted is True
                
                remaining_instructions = await store.get_all_instructions(user_id)
                assert len(remaining_instructions) == 1
                
                # Clean up
                await store.delete_all_user_data(user_id)
                
        except ProceduralMemoryError:
            pytest.skip("PostgreSQL not available for integration tests")
    
    async def test_vectorized_instruction_search(self):
        """Test vector similarity search for instructions."""
        try:
            async with ProceduralMemoryStore() as store:
                user_id = "test_user_vector_search"
                
                # Clean up
                await store.delete_all_user_data(user_id)
                
                # Add diverse instructions
                instructions = [
                    {
                        "category": InstructionCategory.COMMUNICATION_STYLE,
                        "title": "Use professional tone",
                        "instruction": "Maintain a professional, business-appropriate tone in all communications."
                    },
                    {
                        "category": InstructionCategory.TRAVEL_BOOKING,
                        "title": "Avoid red-eye flights",
                        "instruction": "When booking flights, avoid overnight flights (red-eye flights) that depart between 9 PM and 6 AM."
                    },
                    {
                        "category": InstructionCategory.SHOPPING,
                        "title": "Compare prices",
                        "instruction": "Always compare prices from at least 3 different vendors before making purchase recommendations."
                    },
                    {
                        "category": InstructionCategory.SCHEDULING,
                        "title": "No morning meetings",
                        "instruction": "Avoid scheduling meetings before 10 AM as user is not a morning person."
                    },
                    {
                        "category": InstructionCategory.GENERAL,
                        "title": "Ask for clarification",
                        "instruction": "When user requests are ambiguous, always ask clarifying questions before proceeding."
                    }
                ]
                
                instruction_ids = []
                for instr in instructions:
                    inst_id = await store.add_instruction(
                        user_id=user_id,
                        category=instr["category"],
                        title=instr["title"],
                        instruction=instr["instruction"],
                        confidence=0.8,
                        priority=5
                    )
                    instruction_ids.append(inst_id)
                
                # Test similarity searches
                test_queries = [
                    ("how to talk to the user", ["professional tone"]),
                    ("book a flight for tonight", ["red-eye flights"]),
                    ("find the best price for a laptop", ["compare prices"]),
                    ("schedule a meeting for 9 AM", ["morning meetings"]),
                    ("user request is unclear", ["clarification"])
                ]
                
                for query, expected_keywords in test_queries:
                    relevant = await store.get_relevant_instructions(
                        user_id=user_id,
                        query_text=query,
                        min_confidence=0.1,  # Low threshold for testing
                        limit=10
                    )
                    
                    # Should find at least one relevant instruction
                    assert len(relevant) > 0, f"No instructions found for query: {query}"
                    
                    # Check if expected keywords appear in results
                    found_relevant = False
                    for instr in relevant:
                        instruction_text = (instr.instruction.title + " " + instr.instruction.instruction).lower()
                        if any(keyword.lower() in instruction_text for keyword in expected_keywords):
                            found_relevant = True
                            break
                    
                    assert found_relevant, f"Expected keywords {expected_keywords} not found for query: {query}"
                    
                    # Verify relevance scores are reasonable
                    for instr in relevant:
                        assert 0.0 <= instr.relevance_score <= 1.0
                
                # Clean up
                await store.delete_all_user_data(user_id)
                
        except ProceduralMemoryError:
            pytest.skip("PostgreSQL not available for integration tests")
    
    async def test_procedural_memory_stats(self):
        """Test procedural memory statistics functionality."""
        try:
            async with ProceduralMemoryStore() as store:
                user_id = "test_user_stats"
                
                # Clean up
                await store.delete_all_user_data(user_id)
                
                # Add profile entries
                profile_entries = [
                    (ProfileCategory.PERSONAL, "name", "Test User"),
                    (ProfileCategory.PERSONAL, "email", "test@example.com"),
                    (ProfileCategory.TIMEZONE, "primary", "UTC"),
                    (ProfileCategory.PREFERENCES, "style", "casual"),
                    (ProfileCategory.PREFERENCES, "format", "brief"),
                ]
                
                for category, key, value in profile_entries:
                    await store.set_profile_value(user_id, category, key, value)
                
                # Add instructions
                instruction_categories = [
                    InstructionCategory.COMMUNICATION_STYLE,
                    InstructionCategory.TRAVEL_BOOKING,
                    InstructionCategory.SHOPPING,
                    InstructionCategory.SCHEDULING,
                ]
                
                for i, category in enumerate(instruction_categories):
                    await store.add_instruction(
                        user_id=user_id,
                        category=category,
                        title=f"Test instruction {i+1}",
                        instruction=f"This is test instruction {i+1} for category {category.value}",
                        confidence=0.7 + (i * 0.05),  # Varying confidence
                        priority=5 + i
                    )
                
                # Add one inactive instruction
                await store.add_instruction(
                    user_id=user_id,
                    category=InstructionCategory.GENERAL,
                    title="Inactive instruction",
                    instruction="This instruction is inactive",
                    confidence=0.6,
                    priority=3
                )
                
                # Deactivate the last instruction
                all_instructions = await store.get_all_instructions(user_id, include_inactive=True)
                inactive_instruction = next(
                    instr for instr in all_instructions if "inactive" in instr.title.lower()
                )
                await store.update_instruction(
                    user_id=user_id,
                    instruction_id=inactive_instruction.instruction_id,
                    is_active=False
                )
                
                # Get statistics
                stats = await store.get_procedural_memory_stats(user_id)
                
                # Verify profile stats
                assert stats.total_profile_entries == 5
                assert stats.profile_categories["personal"] == 2
                assert stats.profile_categories["timezone"] == 1
                assert stats.profile_categories["preferences"] == 2
                
                # Verify instruction stats
                assert stats.total_instructions == 5
                assert stats.active_instructions == 4  # One is inactive
                assert 0.6 <= stats.average_instruction_confidence <= 1.0
                
                # Verify instruction categories
                assert stats.instruction_categories["communication_style"] == 1
                assert stats.instruction_categories["travel_booking"] == 1
                assert stats.instruction_categories["shopping"] == 1
                assert stats.instruction_categories["scheduling"] == 1
                assert stats.instruction_categories["general"] == 1
                
                # Verify timestamps
                assert stats.last_profile_update is not None
                assert stats.last_instruction_update is not None
                
                # Clean up
                deleted_profile, deleted_instructions = await store.delete_all_user_data(user_id)
                assert deleted_profile == 5
                assert deleted_instructions == 5
                
        except ProceduralMemoryError:
            pytest.skip("PostgreSQL not available for integration tests")
    
    async def test_user_data_isolation(self):
        """Test that different users' data is properly isolated."""
        try:
            async with ProceduralMemoryStore() as store:
                user1 = "test_user_isolation_1"
                user2 = "test_user_isolation_2"
                
                # Clean up
                await store.delete_all_user_data(user1)
                await store.delete_all_user_data(user2)
                
                # Add data for user 1
                await store.set_profile_value(user1, ProfileCategory.PERSONAL, "name", "User One")
                await store.add_instruction(
                    user_id=user1,
                    category=InstructionCategory.COMMUNICATION_STYLE,
                    title="User 1 style",
                    instruction="User 1 communication style instruction"
                )
                
                # Add data for user 2
                await store.set_profile_value(user2, ProfileCategory.PERSONAL, "name", "User Two")
                await store.add_instruction(
                    user_id=user2,
                    category=InstructionCategory.COMMUNICATION_STYLE,
                    title="User 2 style",
                    instruction="User 2 communication style instruction"
                )
                
                # Verify user 1 can only see their data
                profile1 = await store.get_user_profile(user1)
                assert profile1.get_value(ProfileCategory.PERSONAL, "name") == "User One"
                
                instructions1 = await store.get_all_instructions(user1)
                assert len(instructions1) == 1
                assert "User 1" in instructions1[0].instruction
                
                # Verify user 2 can only see their data
                profile2 = await store.get_user_profile(user2)
                assert profile2.get_value(ProfileCategory.PERSONAL, "name") == "User Two"
                
                instructions2 = await store.get_all_instructions(user2)
                assert len(instructions2) == 1
                assert "User 2" in instructions2[0].instruction
                
                # Verify cross-user searches don't return other users' data
                relevant1 = await store.get_relevant_instructions(user1, "communication style")
                relevant2 = await store.get_relevant_instructions(user2, "communication style")
                
                assert all("User 1" in instr.instruction.instruction for instr in relevant1)
                assert all("User 2" in instr.instruction.instruction for instr in relevant2)
                
                # Clean up
                await store.delete_all_user_data(user1)
                await store.delete_all_user_data(user2)
                
        except ProceduralMemoryError:
            pytest.skip("PostgreSQL not available for integration tests")
    
    async def test_instruction_confidence_and_priority_filtering(self):
        """Test filtering instructions by confidence and priority."""
        try:
            async with ProceduralMemoryStore() as store:
                user_id = "test_user_filtering"
                
                # Clean up
                await store.delete_all_user_data(user_id)
                
                # Add instructions with different confidence and priority levels
                test_instructions = [
                    {"title": "High confidence, high priority", "confidence": 0.9, "priority": 8},
                    {"title": "High confidence, low priority", "confidence": 0.9, "priority": 2},
                    {"title": "Low confidence, high priority", "confidence": 0.3, "priority": 8},
                    {"title": "Low confidence, low priority", "confidence": 0.2, "priority": 2},
                ]
                
                for instr in test_instructions:
                    await store.add_instruction(
                        user_id=user_id,
                        category=InstructionCategory.GENERAL,
                        title=instr["title"],
                        instruction=f"Instruction: {instr['title']}",
                        confidence=instr["confidence"],
                        priority=instr["priority"]
                    )
                
                # Test high confidence filter
                high_confidence = await store.get_relevant_instructions(
                    user_id=user_id,
                    query_text="instruction",
                    min_confidence=0.8,
                    min_priority=1,
                    limit=10
                )
                
                assert len(high_confidence) == 2  # Only high confidence instructions
                for instr in high_confidence:
                    assert instr.instruction.confidence >= 0.8
                
                # Test high priority filter
                high_priority = await store.get_relevant_instructions(
                    user_id=user_id,
                    query_text="instruction",
                    min_confidence=0.1,
                    min_priority=7,
                    limit=10
                )
                
                assert len(high_priority) == 2  # Only high priority instructions
                for instr in high_priority:
                    assert instr.instruction.priority >= 7
                
                # Test combined filters
                high_conf_high_prio = await store.get_relevant_instructions(
                    user_id=user_id,
                    query_text="instruction",
                    min_confidence=0.8,
                    min_priority=7,
                    limit=10
                )
                
                assert len(high_conf_high_prio) == 1  # Only one meets both criteria
                assert high_conf_high_prio[0].instruction.confidence >= 0.8
                assert high_conf_high_prio[0].instruction.priority >= 7
                
                # Clean up
                await store.delete_all_user_data(user_id)
                
        except ProceduralMemoryError:
            pytest.skip("PostgreSQL not available for integration tests")