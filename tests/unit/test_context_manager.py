"""Unit tests for the Context Manager system."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.governor.context import ContextAssembler, ContextData
from src.core.domain.state import GovernorState, StateNode
from src.core.domain.tools import ToolCall, RiskLevel


class TestContextData:
    """Test cases for ContextData structure."""
    
    def test_context_data_creation(self):
        """Test ContextData structure creation."""
        context_data = ContextData(
            persona="Test persona",
            environment={"time": "now"},
            memory={"recent": []},
            tasks=[{"type": "test"}],
            metadata={"user": "test"}
        )
        
        assert context_data.persona == "Test persona"
        assert context_data.environment["time"] == "now"
        assert context_data.memory["recent"] == []
        assert len(context_data.tasks) == 1
        assert context_data.metadata["user"] == "test"


class TestContextAssembler:
    """Test cases for the ContextAssembler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.assembler = ContextAssembler()
        self.test_state = GovernorState(
            user_id="user_123",
            session_id="session_456",
            current_node=StateNode.ANALYZE
        )
        self.test_state.total_turns = 5
        self.test_state.total_tools_executed = 2
        self.test_state.error_count = 0
        self.test_state.awaiting_confirmation = False
        self.test_state.pending_tools = []
        self.test_state.conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What's the weather?"},
        ]
        self.test_state.created_at = datetime.utcnow() - timedelta(minutes=30)
    
    def test_assembler_initialization(self):
        """Test ContextAssembler initialization."""
        assembler = ContextAssembler()
        assert isinstance(assembler.persona_cache, dict)
        assert isinstance(assembler.environment_cache, dict)
        assert len(assembler.persona_cache) == 0
        assert len(assembler.environment_cache) == 0
    
    def test_assemble_context_basic(self):
        """Test basic context assembly."""
        result = self.assembler.assemble_context(
            user_id="user_123",
            current_input="Test message",
            state=self.test_state,
            available_tools=["get_weather", "send_email"]
        )
        
        assert isinstance(result, str)
        assert "Headless Governor" in result
        assert "CURRENT ENVIRONMENT:" in result
        assert "Test message" in result
        assert "get_weather" in result
        assert "send_email" in result
        assert "analyze" in result.lower()
    
    def test_gather_context_data(self):
        """Test context data gathering."""
        context_data = self.assembler._gather_context_data(
            user_id="user_123",
            current_input="Test input",
            state=self.test_state
        )
        
        assert isinstance(context_data, ContextData)
        assert "Headless Governor" in context_data.persona
        assert "current_time" in context_data.environment
        assert "recent_conversation" in context_data.memory
        assert isinstance(context_data.tasks, list)
        assert context_data.metadata["user_id"] == "user_123"
        assert context_data.metadata["session_id"] == "session_456"
        assert context_data.metadata["current_node"] == "analyze"
    
    def test_get_persona_default(self):
        """Test default persona retrieval."""
        persona = self.assembler._get_persona("user_123")
        
        assert isinstance(persona, str)
        assert "Headless Governor" in persona
        assert "CORE PRINCIPLES:" in persona
        assert "INTERACTION STYLE:" in persona
        assert "security" in persona.lower()
        assert "autonomous" in persona.lower()
    
    @patch('src.governor.context.assembler.datetime')
    def test_get_environment_context(self, mock_datetime):
        """Test environment context retrieval."""
        # Mock datetime to ensure predictable results
        mock_now = datetime(2024, 11, 27, 14, 30, 0)  # Wednesday afternoon
        mock_datetime.utcnow.return_value = mock_now
        
        env_context = self.assembler._get_environment_context("user_123")
        
        assert isinstance(env_context, dict)
        assert "current_time" in env_context
        assert "timestamp" in env_context
        assert "day_of_week" in env_context
        assert "time_period" in env_context
        assert env_context["day_of_week"] == "Wednesday"
        assert env_context["time_period"] == "afternoon"
        assert env_context["system_load"] == "normal"
    
    def test_get_environment_context_caching(self):
        """Test environment context caching."""
        # First call
        env1 = self.assembler._get_environment_context("user_123")
        
        # Second call should use cache
        env2 = self.assembler._get_environment_context("user_123")
        
        assert env1 == env2
        assert len(self.assembler.environment_cache) > 0
    
    def test_get_memory_context(self):
        """Test memory context retrieval."""
        memory_context = self.assembler._get_memory_context(
            user_id="user_123",
            current_input="Test input",
            state=self.test_state
        )
        
        assert isinstance(memory_context, dict)
        assert "recent_conversation" in memory_context
        assert "conversation_length" in memory_context
        assert "session_age_minutes" in memory_context
        assert "user_preferences" in memory_context
        assert "conversation_topics" in memory_context
        assert "session_summary" in memory_context
        
        assert memory_context["conversation_length"] == 3
        assert memory_context["session_age_minutes"] >= 29  # About 30 minutes
        assert len(memory_context["recent_conversation"]) == 3
    
    def test_get_active_tasks_no_pending(self):
        """Test active tasks with no pending tools."""
        tasks = self.assembler._get_active_tasks("user_123", self.test_state)
        
        assert isinstance(tasks, list)
        assert len(tasks) == 0
    
    def test_get_active_tasks_with_pending(self):
        """Test active tasks with pending tools."""
        # Add pending tool
        tool_call = ToolCall(
            tool_name="get_weather",
            arguments={"location": "Seattle"},
            risk_level=RiskLevel.SAFE,
            execution_id="exec_123",
            user_id="user_123",
            session_id="session_456"
        )
        self.test_state.pending_tools = [tool_call]
        self.test_state.awaiting_confirmation = True
        
        tasks = self.assembler._get_active_tasks("user_123", self.test_state)
        
        assert len(tasks) == 1
        assert tasks[0]["type"] == "pending_tool_execution"
        assert tasks[0]["tool_name"] == "get_weather"
        assert tasks[0]["status"] == "awaiting_confirmation"
        assert tasks[0]["risk_level"] == "safe"
        assert tasks[0]["execution_id"] == "exec_123"
    
    def test_build_prompt_complete(self):
        """Test complete prompt building."""
        context_data = ContextData(
            persona="Test Persona",
            environment={
                "current_time": "2024-11-27 14:30:00 UTC",
                "day_of_week": "Wednesday",
                "time_period": "afternoon",
                "system_load": "normal"
            },
            memory={
                "recent_conversation": [{"role": "user", "content": "Hello"}],
                "conversation_length": 1,
                "session_age_minutes": 30.0,
                "conversation_topics": ["general"]
            },
            tasks=[{
                "type": "pending_tool_execution",
                "tool_name": "get_weather",
                "status": "pending"
            }],
            metadata={"user_id": "user_123"}
        )
        
        prompt = self.assembler._build_prompt(
            context_data=context_data,
            current_input="Test input",
            state=self.test_state,
            available_tools=["get_weather", "calculate"]
        )
        
        assert "Test Persona" in prompt
        assert "CURRENT ENVIRONMENT:" in prompt
        assert "2024-11-27 14:30:00 UTC" in prompt
        assert "Wednesday" in prompt
        assert "afternoon" in prompt
        assert "RECENT CONVERSATION:" in prompt
        assert "30" in prompt  # session age
        assert "general" in prompt  # topics
        assert "ACTIVE TASKS:" in prompt
        assert "get_weather" in prompt
        assert "AVAILABLE TOOLS:" in prompt
        assert "calculate" in prompt
        assert "CURRENT SITUATION:" in prompt
        assert "Test input" in prompt
        assert "analyze" in prompt
    
    def test_get_time_period(self):
        """Test time period classification."""
        assert self.assembler._get_time_period(6) == "morning"
        assert self.assembler._get_time_period(9) == "morning"
        assert self.assembler._get_time_period(12) == "afternoon"
        assert self.assembler._get_time_period(15) == "afternoon"
        assert self.assembler._get_time_period(18) == "evening"
        assert self.assembler._get_time_period(20) == "evening"
        assert self.assembler._get_time_period(22) == "night"
        assert self.assembler._get_time_period(2) == "night"
    
    def test_calculate_session_age(self):
        """Test session age calculation."""
        # State with created_at 30 minutes ago
        age = self.assembler._calculate_session_age(self.test_state)
        assert 29 <= age <= 31  # About 30 minutes, allowing for test execution time
        
        # State without created_at
        state_no_time = GovernorState(user_id="test", session_id="test")
        age = self.assembler._calculate_session_age(state_no_time)
        assert age < 1.0  # Should be very close to 0
    
    def test_get_user_preferences(self):
        """Test user preferences retrieval."""
        preferences = self.assembler._get_user_preferences("user_123")
        
        assert isinstance(preferences, dict)
        assert "communication_style" in preferences
        assert "confirmation_level" in preferences
        assert "timezone" in preferences
        assert "preferred_tools" in preferences
        assert preferences["communication_style"] == "professional"
        assert preferences["confirmation_level"] == "standard"
        assert isinstance(preferences["preferred_tools"], list)
    
    def test_extract_topics(self):
        """Test topic extraction from messages."""
        messages = [
            {"content": "What's the weather like today?"},
            {"content": "Can you send an email to John?"},
            {"content": "Help me calculate 2+2"},
            {"content": "Search for Python tutorials"},
            {"content": "Read my file please"}
        ]
        
        topics = self.assembler._extract_topics(messages)
        
        assert isinstance(topics, list)
        expected_topics = {"weather", "email", "calculations", "search", "files"}
        assert set(topics) == expected_topics
    
    def test_extract_topics_general(self):
        """Test topic extraction with no specific topics."""
        messages = [
            {"content": "Hello there"},
            {"content": "How are you doing?"},
        ]
        
        topics = self.assembler._extract_topics(messages)
        assert topics == ["general"]
    
    def test_generate_session_summary(self):
        """Test session summary generation."""
        # New session
        new_state = GovernorState(user_id="test", session_id="test")
        summary = self.assembler._generate_session_summary(new_state)
        assert "New session started" in summary
        
        # Active conversation
        active_state = GovernorState(user_id="test", session_id="test")
        active_state.total_turns = 5
        active_state.total_tools_executed = 0
        active_state.error_count = 0
        summary = self.assembler._generate_session_summary(active_state)
        assert "Active conversation" in summary
        assert "5 exchanges" in summary
        
        # Productive session
        productive_state = GovernorState(user_id="test", session_id="test")
        productive_state.total_turns = 10
        productive_state.total_tools_executed = 3
        productive_state.error_count = 1
        summary = self.assembler._generate_session_summary(productive_state)
        assert "Productive session" in summary
        assert "3 tools executed" in summary
        assert "1 errors" in summary
    
    def test_clear_cache(self):
        """Test cache clearing functionality."""
        # Populate caches
        self.assembler.persona_cache["test"] = "data"
        self.assembler.environment_cache["test"] = {"data": "value"}
        
        assert len(self.assembler.persona_cache) == 1
        assert len(self.assembler.environment_cache) == 1
        
        # Clear caches
        self.assembler.clear_cache()
        
        assert len(self.assembler.persona_cache) == 0
        assert len(self.assembler.environment_cache) == 0
    
    def test_assemble_context_with_empty_state(self):
        """Test context assembly with minimal state."""
        minimal_state = GovernorState(
            user_id="user_456", 
            session_id="session_789",
            current_node=StateNode.IDLE
        )
        
        result = self.assembler.assemble_context(
            user_id="user_456",
            current_input="Hello",
            state=minimal_state
        )
        
        assert isinstance(result, str)
        assert "Hello" in result
        assert "idle" in result.lower()
        assert "user_456" not in result  # User ID shouldn't be in the prompt content
        assert "session_789" in result  # Session ID should be in current situation
    
    def test_assemble_context_with_tools_and_confirmation(self):
        """Test context assembly with pending confirmation."""
        # Set up state with pending confirmation
        tool_call = ToolCall(
            tool_name="send_email",
            arguments={"to": "test@example.com"},
            risk_level=RiskLevel.DANGEROUS,
            execution_id="exec_456",
            user_id="user_123",
            session_id="session_456"
        )
        
        self.test_state.pending_tools = [tool_call]
        self.test_state.awaiting_confirmation = True
        self.test_state.current_node = StateNode.AWAIT_CONFIRMATION
        
        result = self.assembler.assemble_context(
            user_id="user_123",
            current_input="Yes, please send it",
            state=self.test_state,
            available_tools=["send_email", "get_weather"]
        )
        
        assert "ACTIVE TASKS:" in result
        assert "send_email" in result
        assert "awaiting_confirmation" in result
        assert "dangerous" in result
        assert "Awaiting Confirmation: Yes" in result
        assert "await_confirmation" in result.lower()


class TestContextAssemblerIntegration:
    """Integration tests for Context Manager."""
    
    def test_full_context_assembly_flow(self):
        """Test complete context assembly flow with realistic data."""
        assembler = ContextAssembler()
        
        # Create realistic state
        state = GovernorState(
            user_id="alice_123",
            session_id="sess_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
            current_node=StateNode.TOOL_DECISION
        )
        state.total_turns = 8
        state.total_tools_executed = 3
        state.error_count = 0
        state.awaiting_confirmation = False
        state.pending_tools = []
        state.conversation_history = [
            {"role": "user", "content": "Good morning! What's the weather like?"},
            {"role": "assistant", "content": "I'll check the weather for you."},
            {"role": "user", "content": "Also, can you send an email to my team about the meeting?"},
            {"role": "assistant", "content": "I can help with that. What should the email say?"},
            {"role": "user", "content": "Tell them the meeting is moved to 3 PM tomorrow"},
        ]
        state.created_at = datetime.utcnow() - timedelta(minutes=45)
        
        # Assemble context
        result = assembler.assemble_context(
            user_id="alice_123",
            current_input="Yes, please send that email now",
            state=state,
            available_tools=["get_weather", "send_email", "calculate", "search_web"]
        )
        
        # Verify comprehensive prompt
        assert len(result) > 500  # Should be a substantial prompt
        
        # Check all major sections are present
        assert "Headless Governor" in result
        assert "CURRENT ENVIRONMENT:" in result
        assert "RECENT CONVERSATION:" in result
        assert "AVAILABLE TOOLS:" in result
        assert "CURRENT SITUATION:" in result
        
        # Check specific content
        assert any(period in result.lower() for period in ["morning", "afternoon", "evening", "night"])
        assert "45" in result or "Session length:" in result
        assert "weather" in result  # from topics
        assert "email" in result  # from topics and tools
        assert "send_email" in result
        assert "get_weather" in result
        assert "tool_decision" in result
        assert "Yes, please send that email now" in result
    
    def test_context_consistency_across_calls(self):
        """Test that context assembly is consistent across multiple calls."""
        assembler = ContextAssembler()
        
        state = GovernorState(
            user_id="bob_456",
            session_id="consistent_test",
            current_node=StateNode.ANALYZE
        )
        state.created_at = datetime.utcnow() - timedelta(minutes=15)
        
        # Make multiple calls with same input
        result1 = assembler.assemble_context("bob_456", "Test consistency", state)
        result2 = assembler.assemble_context("bob_456", "Test consistency", state)
        
        # Results should be very similar (timestamps might differ slightly)
        # Compare major sections
        assert "Headless Governor" in result1 and "Headless Governor" in result2
        assert "Test consistency" in result1 and "Test consistency" in result2
        assert "analyze" in result1.lower() and "analyze" in result2.lower()
        
        # Environment context should be cached and identical
        lines1 = result1.split('\n')
        lines2 = result2.split('\n')
        env_lines1 = [l for l in lines1 if 'Time:' in l or 'Day:' in l]
        env_lines2 = [l for l in lines2 if 'Time:' in l or 'Day:' in l]
        assert env_lines1 == env_lines2  # Should be cached and identical