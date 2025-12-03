"""Unit tests for the Governor workflow."""

import pytest

from src.core.domain.state import GovernorState, StateNode
from src.core.domain.tools import RiskLevel, ToolCall
from src.core.workflow.governor_workflow import GovernorWorkflow


class TestGovernorWorkflow:
    """Test cases for GovernorWorkflow."""
    
    def test_governor_workflow_initialization(self):
        """Test Governor workflow initialization."""
        workflow = GovernorWorkflow()
        
        # Test that all nodes are initialized
        assert workflow.idle_node is not None
        assert workflow.analyze_node is not None
        assert workflow.tool_decision_node is not None
        assert workflow.policy_check_node is not None
        assert workflow.execute_node is not None
        assert workflow.await_confirmation_node is not None
        assert workflow.respond_node is not None
        
        # Test that all conditionals are initialized
        assert workflow.idle_conditional is not None
        assert workflow.analyze_conditional is not None
        assert workflow.tool_decision_conditional is not None
        assert workflow.policy_check_conditional is not None
        assert workflow.execute_conditional is not None
        assert workflow.await_confirmation_conditional is not None
        assert workflow.respond_conditional is not None
    
    def test_governor_workflow_build_graph(self):
        """Test Governor workflow graph building."""
        workflow = GovernorWorkflow()
        
        graph = workflow.build_graph()
        
        assert graph is not None
        # Verify all nodes are added to the graph
        # Note: This is testing the graph structure, implementation details may vary
    
    def test_governor_workflow_entry_and_finish_points(self):
        """Test workflow entry and finish points."""
        workflow = GovernorWorkflow()
        
        assert workflow.get_entry_point() == StateNode.IDLE.value
        assert workflow.get_finish_point() == StateNode.IDLE.value
    
    def test_governor_workflow_compile(self):
        """Test Governor workflow compilation."""
        workflow = GovernorWorkflow()
        
        compiled_graph = workflow.compile()
        
        assert compiled_graph is not None
        assert workflow._compiled_graph is not None
    
    @pytest.mark.asyncio
    async def test_process_user_input_new_session(self):
        """Test processing user input for a new session."""
        workflow = GovernorWorkflow()
        
        result_state = await workflow.process_user_input(
            user_input="Hello, can you help me?",
            user_id="user_123",
            session_id="session_456"
        )
        
        assert result_state is not None
        assert result_state.user_id == "user_123"
        assert result_state.session_id == "session_456"
        assert result_state.last_user_input == "Hello, can you help me?"
        assert result_state.last_assistant_response is not None
        assert result_state.total_turns >= 1
        assert len(result_state.conversation_history) >= 1
    
    @pytest.mark.asyncio
    async def test_process_user_input_existing_session(self):
        """Test processing user input for an existing session."""
        workflow = GovernorWorkflow()
        
        # Create existing state
        existing_state = GovernorState(
            user_id="user_123",
            session_id="session_456",
            total_turns=2
        )
        existing_state.add_conversation_turn(
            user_input="Previous message",
            assistant_response="Previous response"
        )
        
        result_state = await workflow.process_user_input(
            user_input="Follow up message",
            user_id="user_123", 
            session_id="session_456",
            existing_state=existing_state
        )
        
        assert result_state.user_id == "user_123"
        assert result_state.session_id == "session_456"
        assert result_state.last_user_input == "Follow up message"
        assert result_state.total_turns >= 2
        assert len(result_state.conversation_history) >= 2
    
    @pytest.mark.asyncio
    async def test_handle_confirmation_response(self):
        """Test handling confirmation responses."""
        workflow = GovernorWorkflow()
        
        # Create state with pending confirmation
        state = GovernorState(
            user_id="user_123",
            session_id="session_456",
            awaiting_confirmation=True
        )
        state.request_confirmation({
            "confirmation_type": "test",
            "message": "Test confirmation"
        })
        
        result_state = await workflow.handle_confirmation_response("yes", state)
        
        assert result_state is not None
        assert result_state.last_user_input == "yes"
    
    def test_get_workflow_status(self):
        """Test getting workflow status."""
        workflow = GovernorWorkflow()
        
        state = GovernorState(
            user_id="user_123",
            session_id="session_456"
        )
        state.transition_to(StateNode.ANALYZE)
        
        # Add a pending tool
        tool_call = ToolCall(
            tool_name="test_tool",
            arguments={"param": "value"},
            risk_level=RiskLevel.SAFE,
            execution_id="exec_123",
            user_id="user_123",
            session_id="session_456"
        )
        state.add_tool_call(tool_call)
        
        status = workflow.get_workflow_status(state)
        
        assert status["current_node"] == StateNode.ANALYZE.value
        assert status["previous_node"] == StateNode.IDLE.value
        assert status["awaiting_confirmation"] == False
        assert status["pending_tools_count"] == 1
        assert status["executing_tool"] is None
        assert status["total_turns"] == 0
        assert status["error_count"] == 0
        assert "session_age_hours" in status
        assert "average_response_time_ms" in status
    
    def test_get_conversation_summary_empty(self):
        """Test getting conversation summary for empty session."""
        workflow = GovernorWorkflow()
        
        state = GovernorState(
            user_id="user_123",
            session_id="session_456"
        )
        
        summary = workflow.get_conversation_summary(state)
        
        assert summary["total_turns"] == 0
        assert summary["recent_turns"] == []
        assert summary["total_tools_executed"] == 0
        assert summary["success_rate"] == 1.0
        assert "session_started" in summary
        assert "last_activity" in summary
    
    def test_get_conversation_summary_with_history(self):
        """Test getting conversation summary with conversation history."""
        workflow = GovernorWorkflow()
        
        state = GovernorState(
            user_id="user_123",
            session_id="session_456"
        )
        
        # Add conversation turns
        for i in range(3):
            state.add_conversation_turn(
                user_input=f"User message {i+1}",
                assistant_response=f"Assistant response {i+1}",
                tools_used=[f"tool_{i+1}"] if i % 2 == 0 else []
            )
        
        # Simulate some tool executions
        state.total_tools_executed = 2
        
        summary = workflow.get_conversation_summary(state)
        
        assert summary["total_turns"] == 3
        assert len(summary["recent_turns"]) == 3
        assert summary["total_tools_executed"] == 2
        assert summary["success_rate"] == 1.0  # No errors
        
        # Check recent turns structure
        recent_turn = summary["recent_turns"][0]
        assert "user_input" in recent_turn
        assert "assistant_response" in recent_turn
        assert "tools_used" in recent_turn
        assert "timestamp" in recent_turn
    
    def test_reset_session(self):
        """Test resetting a session."""
        workflow = GovernorWorkflow()
        
        # Create state with history and data
        state = GovernorState(
            user_id="user_123",
            session_id="session_456"
        )
        state.add_conversation_turn("Test", "Response")
        state.transition_to(StateNode.ANALYZE)
        state.record_error("Test error")
        
        # Reset the session
        reset_state = workflow.reset_session(state)
        
        # Verify reset state
        assert reset_state.user_id == "user_123"
        assert reset_state.session_id == "session_456"
        assert reset_state.current_node == StateNode.IDLE
        assert reset_state.total_turns == 0
        assert reset_state.error_count == 0
        assert len(reset_state.conversation_history) == 0
        assert reset_state.last_user_input is None
        assert reset_state.last_assistant_response is None
    
    def test_extract_tools_used_empty(self):
        """Test extracting tools used when no tools were executed."""
        workflow = GovernorWorkflow()
        
        state = GovernorState(
            user_id="user_123",
            session_id="session_456"
        )
        
        tools_used = workflow._extract_tools_used(state)
        
        assert tools_used == []
    
    def test_extract_tools_used_with_results(self):
        """Test extracting tools used from execution results."""
        workflow = GovernorWorkflow()
        
        state = GovernorState(
            user_id="user_123",
            session_id="session_456"
        )
        
        # Add execution results to context
        state.context["execution_results"] = {
            "results": [
                {
                    "tool_name": "weather_tool",
                    "status": "success",
                    "result": {"temp": "70F"}
                },
                {
                    "tool_name": "email_tool",
                    "status": "failed",
                    "error": "Connection error"
                },
                {
                    "tool_name": "calendar_tool", 
                    "status": "success",
                    "result": {"event_created": True}
                }
            ]
        }
        
        tools_used = workflow._extract_tools_used(state)
        
        # Should only include successful tools
        assert tools_used == ["weather_tool", "calendar_tool"]