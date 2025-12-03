"""Unit tests for domain state models."""

from datetime import datetime
from unittest.mock import patch

import pytest

from src.core.domain.state import (
    ActiveTask,
    ConversationTurn,
    GovernorState,
    StateNode,
)
from src.core.domain.tools import RiskLevel, ToolCall, ToolStatus


class TestStateNode:
    """Test cases for StateNode enum."""

    def test_all_state_nodes(self) -> None:
        """Test that all expected state nodes exist."""
        expected_states = {
            "idle", "analyze", "tool_decision", 
            "policy_check", "execute", "await_confirmation", "respond"
        }
        
        actual_states = {node.value for node in StateNode}
        assert actual_states == expected_states


class TestConversationTurn:
    """Test cases for ConversationTurn model."""

    def test_valid_conversation_turn(self) -> None:
        """Test creating a valid ConversationTurn."""
        turn = ConversationTurn(
            turn_id="turn_123",
            user_input="What's the weather?",
            assistant_response="I'll check the weather for you.",
            tools_used=["get_weather"]
        )
        
        assert turn.turn_id == "turn_123"
        assert turn.user_input == "What's the weather?"
        assert turn.assistant_response == "I'll check the weather for you."
        assert turn.tools_used == ["get_weather"]
        assert isinstance(turn.timestamp, datetime)


class TestActiveTask:
    """Test cases for ActiveTask model."""

    def test_valid_active_task(self) -> None:
        """Test creating a valid ActiveTask."""
        scheduled_time = datetime.utcnow()
        
        task = ActiveTask(
            task_id="task_123",
            task_type="reminder",
            description="Meeting in 30 minutes",
            scheduled_for=scheduled_time
        )
        
        assert task.task_id == "task_123"
        assert task.task_type == "reminder"
        assert task.description == "Meeting in 30 minutes"
        assert task.status == "active"  # Default
        assert task.scheduled_for == scheduled_time
        assert isinstance(task.created_at, datetime)


class TestGovernorState:
    """Test cases for GovernorState model."""

    def test_valid_governor_state(self) -> None:
        """Test creating a valid GovernorState."""
        state = GovernorState(
            user_id="user_123",
            session_id="session_456"
        )
        
        assert state.user_id == "user_123"
        assert state.session_id == "session_456"
        assert state.current_node == StateNode.IDLE  # Default
        assert state.previous_node is None
        assert state.context == {}
        assert state.conversation_history == []
        assert state.pending_tools == []
        assert state.executing_tool is None
        assert state.awaiting_confirmation is False
        assert state.confirmation_context is None
        assert state.active_tasks == []
        assert state.error_count == 0
        assert state.total_turns == 0
        assert state.total_tools_executed == 0

    def test_add_conversation_turn(self) -> None:
        """Test adding a conversation turn."""
        state = GovernorState(
            user_id="user_123",
            session_id="session_456"
        )
        
        state.add_conversation_turn(
            user_input="Hello",
            assistant_response="Hi there!",
            tools_used=["greeting_tool"]
        )
        
        assert len(state.conversation_history) == 1
        assert state.total_turns == 1
        assert state.last_user_input == "Hello"
        assert state.last_assistant_response == "Hi there!"
        
        turn = state.conversation_history[0]
        assert turn.user_input == "Hello"
        assert turn.assistant_response == "Hi there!"
        assert turn.tools_used == ["greeting_tool"]

    def test_state_transition(self) -> None:
        """Test state transitions."""
        state = GovernorState(
            user_id="user_123",
            session_id="session_456"
        )
        
        assert state.current_node == StateNode.IDLE
        assert state.previous_node is None
        
        state.transition_to(StateNode.ANALYZE)
        
        assert state.current_node == StateNode.ANALYZE
        assert state.previous_node == StateNode.IDLE

    def test_tool_call_management(self) -> None:
        """Test tool call management methods."""
        state = GovernorState(
            user_id="user_123",
            session_id="session_456"
        )
        
        tool_call = ToolCall(
            tool_name="test_tool",
            arguments={"param": "value"},
            risk_level=RiskLevel.SAFE,
            execution_id="exec_123",
            user_id="user_123",
            session_id="session_456"
        )
        
        # Add tool call
        state.add_tool_call(tool_call)
        assert len(state.pending_tools) == 1
        assert state.executing_tool is None
        
        # Start execution
        state.start_tool_execution(tool_call)
        assert len(state.pending_tools) == 0
        assert state.executing_tool == tool_call
        
        # Complete execution
        state.complete_tool_execution(success=True)
        assert state.executing_tool is None
        assert state.total_tools_executed == 1

    def test_confirmation_flow(self) -> None:
        """Test confirmation request and resolution."""
        state = GovernorState(
            user_id="user_123",
            session_id="session_456"
        )
        
        context = {"tool": "send_email", "recipient": "test@example.com"}
        
        # Request confirmation
        state.request_confirmation(context)
        assert state.awaiting_confirmation is True
        assert state.confirmation_context == context
        
        # Resolve confirmation
        state.resolve_confirmation(approved=True)
        assert state.awaiting_confirmation is False
        assert state.confirmation_context is None

    def test_active_task_management(self) -> None:
        """Test active task management."""
        state = GovernorState(
            user_id="user_123",
            session_id="session_456"
        )
        
        task = ActiveTask(
            task_id="task_123",
            task_type="reminder",
            description="Test task"
        )
        
        # Add task
        state.add_active_task(task)
        assert len(state.active_tasks) == 1
        
        # Remove task
        removed = state.remove_active_task("task_123")
        assert removed is True
        assert len(state.active_tasks) == 0
        
        # Try to remove non-existent task
        removed = state.remove_active_task("nonexistent")
        assert removed is False

    def test_error_recording(self) -> None:
        """Test error recording."""
        state = GovernorState(
            user_id="user_123",
            session_id="session_456"
        )
        
        assert state.error_count == 0
        assert state.last_error is None
        
        state.record_error("Test error message")
        
        assert state.error_count == 1
        assert state.last_error == "Test error message"

    def test_response_time_tracking(self) -> None:
        """Test response time tracking with exponential moving average."""
        state = GovernorState(
            user_id="user_123",
            session_id="session_456"
        )
        
        assert state.average_response_time_ms is None
        
        # First measurement
        state.update_response_time(100.0)
        assert state.average_response_time_ms == 100.0
        
        # Second measurement (should use exponential moving average)
        state.update_response_time(200.0)
        expected = 0.1 * 200.0 + 0.9 * 100.0  # alpha=0.1
        assert state.average_response_time_ms == expected

    def test_conversation_history_limit(self) -> None:
        """Test that conversation history is limited to reasonable size."""
        state = GovernorState(
            user_id="user_123",
            session_id="session_456"
        )
        
        # Add many turns (more than the 50 limit)
        for i in range(60):
            state.add_conversation_turn(
                user_input=f"Message {i}",
                assistant_response=f"Response {i}"
            )
        
        # Should only keep the last 50
        assert len(state.conversation_history) == 50
        assert state.conversation_history[0].user_input == "Message 10"  # First kept
        assert state.conversation_history[-1].user_input == "Message 59"  # Last added