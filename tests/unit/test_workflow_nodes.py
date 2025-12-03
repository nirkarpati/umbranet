"""Unit tests for workflow node implementations."""

import pytest
from unittest.mock import patch

from src.core.domain.state import GovernorState, StateNode
from src.core.domain.tools import RiskLevel, ToolCall
from src.core.workflow.nodes import (
    IdleNode, IdleConditional,
    AnalyzeNode, AnalyzeConditional,
    ToolDecisionNode, ToolDecisionConditional,
    PolicyCheckNode, PolicyCheckConditional,
    ExecuteNode, ExecuteConditional,
    AwaitConfirmationNode, AwaitConfirmationConditional,
    RespondNode, RespondConditional,
)


class TestIdleNode:
    """Test cases for IdleNode."""
    
    def test_idle_node_name(self):
        """Test idle node name property."""
        node = IdleNode()
        assert node.name == StateNode.IDLE.value
    
    @pytest.mark.asyncio
    async def test_idle_node_no_input(self):
        """Test idle node with no user input."""
        node = IdleNode()
        state = GovernorState(user_id="user_123", session_id="session_456")
        
        result = await node(state)
        
        assert result.current_node == StateNode.IDLE  # Should stay idle
        assert result.last_user_input is None
    
    @pytest.mark.asyncio
    async def test_idle_node_with_input(self):
        """Test idle node with user input."""
        node = IdleNode()
        state = GovernorState(
            user_id="user_123", 
            session_id="session_456",
            last_user_input="Hello, how are you?"
        )
        
        result = await node(state)
        
        assert result.current_node == StateNode.ANALYZE
        assert result.previous_node == StateNode.IDLE
        assert "idle_analysis" in result.context
        assert "processing_started_at" in result.context


class TestIdleConditional:
    """Test cases for IdleConditional."""
    
    def test_idle_conditional_name(self):
        """Test idle conditional name."""
        conditional = IdleConditional()
        assert conditional.name == "idle_routing"
    
    def test_idle_conditional_no_input(self):
        """Test idle conditional routing with no input."""
        conditional = IdleConditional()
        state = GovernorState(user_id="user_123", session_id="session_456")
        
        next_node = conditional(state)
        
        assert next_node == StateNode.IDLE.value
    
    def test_idle_conditional_with_input(self):
        """Test idle conditional routing with input."""
        conditional = IdleConditional()
        state = GovernorState(
            user_id="user_123",
            session_id="session_456", 
            last_user_input="Hello"
        )
        
        next_node = conditional(state)
        
        assert next_node == StateNode.ANALYZE.value
    
    def test_idle_conditional_awaiting_confirmation(self):
        """Test idle conditional with pending confirmation."""
        conditional = IdleConditional()
        state = GovernorState(
            user_id="user_123",
            session_id="session_456",
            last_user_input="Hello",
            awaiting_confirmation=True
        )
        
        next_node = conditional(state)
        
        assert next_node == StateNode.AWAIT_CONFIRMATION.value


class TestAnalyzeNode:
    """Test cases for AnalyzeNode."""
    
    def test_analyze_node_name(self):
        """Test analyze node name property."""
        node = AnalyzeNode()
        assert node.name == StateNode.ANALYZE.value
    
    @pytest.mark.asyncio
    async def test_analyze_node_no_input(self):
        """Test analyze node with no user input."""
        node = AnalyzeNode()
        state = GovernorState(user_id="user_123", session_id="session_456")
        
        result = await node(state)
        
        assert result.current_node == StateNode.RESPOND
        assert result.error_count == 1
        assert "No user input to analyze" in result.last_error
    
    @pytest.mark.asyncio
    async def test_analyze_node_simple_question(self):
        """Test analyze node with a simple question."""
        node = AnalyzeNode()
        state = GovernorState(
            user_id="user_123",
            session_id="session_456",
            last_user_input="What time is it?"
        )
        
        result = await node(state)
        
        assert result.current_node == StateNode.RESPOND
        assert "analysis" in result.context
        
        analysis = result.context["analysis"]
        assert analysis["is_question"] is True
        assert analysis["intent"] == "simple_question"
        assert analysis["needs_tools"] is False
    
    @pytest.mark.asyncio
    async def test_analyze_node_tool_request(self):
        """Test analyze node with a request that needs tools."""
        node = AnalyzeNode()
        state = GovernorState(
            user_id="user_123",
            session_id="session_456",
            last_user_input="What's the weather like in Seattle?"
        )
        
        result = await node(state)
        
        assert result.current_node == StateNode.TOOL_DECISION
        assert "analysis" in result.context
        
        analysis = result.context["analysis"]
        assert analysis["needs_tools"] is True
        assert "weather" in analysis["suggested_tools"]


class TestAnalyzeConditional:
    """Test cases for AnalyzeConditional."""
    
    def test_analyze_conditional_needs_tools(self):
        """Test analyze conditional when tools are needed."""
        conditional = AnalyzeConditional()
        state = GovernorState(user_id="user_123", session_id="session_456")
        state.context = {
            "analysis": {
                "needs_tools": True,
                "suggested_tools": ["weather"]
            }
        }
        
        next_node = conditional(state)
        
        assert next_node == StateNode.TOOL_DECISION.value
    
    def test_analyze_conditional_no_tools(self):
        """Test analyze conditional when no tools are needed."""
        conditional = AnalyzeConditional()
        state = GovernorState(user_id="user_123", session_id="session_456")
        state.context = {
            "analysis": {
                "needs_tools": False,
                "intent": "simple_question"
            }
        }
        
        next_node = conditional(state)
        
        assert next_node == StateNode.RESPOND.value


class TestToolDecisionNode:
    """Test cases for ToolDecisionNode."""
    
    def test_tool_decision_node_name(self):
        """Test tool decision node name property."""
        node = ToolDecisionNode()
        assert node.name == StateNode.TOOL_DECISION.value
    
    @pytest.mark.asyncio
    async def test_tool_decision_node_no_analysis(self):
        """Test tool decision node with no analysis."""
        node = ToolDecisionNode()
        state = GovernorState(user_id="user_123", session_id="session_456")
        
        result = await node(state)
        
        assert result.current_node == StateNode.RESPOND
        assert result.error_count == 1
    
    @pytest.mark.asyncio
    async def test_tool_decision_node_with_weather_request(self):
        """Test tool decision node with weather tool suggestion."""
        node = ToolDecisionNode()
        state = GovernorState(
            user_id="user_123",
            session_id="session_456",
            last_user_input="What's the weather in Seattle?"
        )
        state.context = {
            "analysis": {
                "suggested_tools": ["weather"],
                "entities": {},
                "intent": "information_request"
            }
        }
        
        result = await node(state)
        
        assert result.current_node == StateNode.EXECUTE  # Weather is safe
        assert len(result.pending_tools) == 1
        assert result.pending_tools[0].tool_name == "weather"
        assert result.pending_tools[0].risk_level == RiskLevel.SAFE
        assert "tool_decisions" in result.context


class TestPolicyCheckNode:
    """Test cases for PolicyCheckNode."""
    
    def test_policy_check_node_name(self):
        """Test policy check node name property."""
        node = PolicyCheckNode()
        assert node.name == StateNode.POLICY_CHECK.value
    
    @pytest.mark.asyncio
    async def test_policy_check_node_safe_tool(self):
        """Test policy check with safe tool."""
        node = PolicyCheckNode()
        state = GovernorState(user_id="user_123", session_id="session_456")
        
        # Add a safe tool call
        tool_call = ToolCall(
            tool_name="weather",
            arguments={"location": "Seattle"},
            risk_level=RiskLevel.SAFE,
            execution_id="exec_123",
            user_id="user_123",
            session_id="session_456"
        )
        state.add_tool_call(tool_call)
        
        result = await node(state)
        
        assert result.current_node == StateNode.EXECUTE
        assert "policy_evaluation" in result.context
        
        policy_eval = result.context["policy_evaluation"]
        assert policy_eval["approved_count"] == 1
        assert policy_eval["confirmation_required_count"] == 0
    
    @pytest.mark.asyncio
    async def test_policy_check_node_dangerous_tool(self):
        """Test policy check with dangerous tool."""
        node = PolicyCheckNode()
        state = GovernorState(user_id="user_123", session_id="session_456")
        
        # Add a dangerous tool call
        tool_call = ToolCall(
            tool_name="email",
            arguments={"recipient": "test@example.com", "message": "Test"},
            risk_level=RiskLevel.DANGEROUS,
            execution_id="exec_123",
            user_id="user_123",
            session_id="session_456"
        )
        state.add_tool_call(tool_call)
        
        result = await node(state)
        
        assert result.current_node == StateNode.AWAIT_CONFIRMATION
        assert result.awaiting_confirmation is True
        assert "policy_evaluation" in result.context
        
        policy_eval = result.context["policy_evaluation"]
        assert policy_eval["confirmation_required_count"] == 1


class TestExecuteNode:
    """Test cases for ExecuteNode."""
    
    def test_execute_node_name(self):
        """Test execute node name property."""
        node = ExecuteNode()
        assert node.name == StateNode.EXECUTE.value
    
    @pytest.mark.asyncio
    async def test_execute_node_no_tools(self):
        """Test execute node with no pending tools."""
        node = ExecuteNode()
        state = GovernorState(user_id="user_123", session_id="session_456")
        
        result = await node(state)
        
        assert result.current_node == StateNode.RESPOND
        assert result.error_count == 1
    
    @pytest.mark.asyncio
    async def test_execute_node_weather_tool(self):
        """Test execute node with weather tool."""
        node = ExecuteNode()
        state = GovernorState(user_id="user_123", session_id="session_456")
        
        # Add weather tool
        tool_call = ToolCall(
            tool_name="weather",
            arguments={"location": "Seattle"},
            risk_level=RiskLevel.SAFE,
            execution_id="exec_123",
            user_id="user_123",
            session_id="session_456"
        )
        state.add_tool_call(tool_call)
        
        result = await node(state)
        
        assert result.current_node == StateNode.RESPOND
        assert len(result.pending_tools) == 0  # Tool was executed and removed
        assert "execution_results" in result.context
        
        exec_results = result.context["execution_results"]
        assert exec_results["total_executed"] == 1
        assert exec_results["successful_count"] == 1


class TestRespondNode:
    """Test cases for RespondNode."""
    
    def test_respond_node_name(self):
        """Test respond node name property."""
        node = RespondNode()
        assert node.name == StateNode.RESPOND.value
    
    @pytest.mark.asyncio
    async def test_respond_node_simple_response(self):
        """Test respond node generating simple response."""
        node = RespondNode()
        state = GovernorState(
            user_id="user_123",
            session_id="session_456",
            last_user_input="Hello"
        )
        state.context = {
            "analysis": {
                "intent": "conversation",
                "sentiment": "positive"
            }
        }
        
        result = await node(state)
        
        assert result.current_node == StateNode.IDLE
        assert result.last_assistant_response is not None
        assert "generated_response" in result.context
    
    @pytest.mark.asyncio
    async def test_respond_node_with_execution_results(self):
        """Test respond node with tool execution results."""
        node = RespondNode()
        state = GovernorState(
            user_id="user_123",
            session_id="session_456",
            last_user_input="What's the weather?"
        )
        state.context = {
            "execution_results": {
                "total_executed": 1,
                "successful_count": 1,
                "failed_count": 0,
                "results": [
                    {
                        "tool_name": "weather",
                        "status": "success", 
                        "result": {
                            "location": "Seattle",
                            "temperature": "72°F",
                            "condition": "Sunny"
                        }
                    }
                ]
            }
        }
        
        result = await node(state)
        
        assert result.current_node == StateNode.IDLE
        assert result.last_assistant_response is not None
        assert "Weather in Seattle" in result.last_assistant_response
    
    @pytest.mark.asyncio
    async def test_respond_node_error_response(self):
        """Test respond node with error."""
        node = RespondNode()
        state = GovernorState(
            user_id="user_123",
            session_id="session_456",
            last_user_input="Test"
        )
        state.record_error("Test error message")
        
        result = await node(state)
        
        assert result.current_node == StateNode.IDLE
        assert result.last_assistant_response is not None
        assert "error" in result.last_assistant_response.lower()


class TestAwaitConfirmationNode:
    """Test cases for AwaitConfirmationNode."""
    
    def test_await_confirmation_node_name(self):
        """Test await confirmation node name property."""
        node = AwaitConfirmationNode()
        assert node.name == StateNode.AWAIT_CONFIRMATION.value
    
    @pytest.mark.asyncio
    async def test_await_confirmation_node_not_awaiting(self):
        """Test await confirmation when not actually awaiting."""
        node = AwaitConfirmationNode()
        state = GovernorState(user_id="user_123", session_id="session_456")
        
        result = await node(state)
        
        assert result.current_node == StateNode.RESPOND
        assert result.awaiting_confirmation is False
    
    @pytest.mark.asyncio
    async def test_await_confirmation_node_approval(self):
        """Test await confirmation with user approval."""
        node = AwaitConfirmationNode()
        state = GovernorState(
            user_id="user_123",
            session_id="session_456",
            awaiting_confirmation=True,
            last_user_input="yes"
        )
        state.request_confirmation({
            "confirmation_type": "tool_execution",
            "tools_awaiting_confirmation": [
                {"execution_id": "exec_123", "tool_name": "email"}
            ]
        })
        
        result = await node(state)
        
        assert result.current_node == StateNode.EXECUTE
        assert result.awaiting_confirmation is False
        assert "confirmation_log" in result.context
    
    @pytest.mark.asyncio
    async def test_await_confirmation_node_denial(self):
        """Test await confirmation with user denial."""
        node = AwaitConfirmationNode()
        state = GovernorState(
            user_id="user_123",
            session_id="session_456",
            awaiting_confirmation=True,
            last_user_input="no"
        )
        state.request_confirmation({
            "confirmation_type": "tool_execution",
            "tools_awaiting_confirmation": [
                {"execution_id": "exec_123", "tool_name": "email"}
            ]
        })
        
        result = await node(state)
        
        assert result.current_node == StateNode.RESPOND
        assert result.awaiting_confirmation is False
        assert result.context["response_reason"] == "user_denied_confirmation"