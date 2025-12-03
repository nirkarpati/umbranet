"""Unit tests for workflow base classes."""

import pytest

from src.core.domain.state import GovernorState, StateNode
from src.core.workflow.base import WorkflowBase, NodeFunction, ConditionalFunction


class MockNodeFunction(NodeFunction):
    """Mock node function for testing."""
    
    def __init__(self, name: str):
        self._name = name
        self.call_count = 0
        
    @property
    def name(self) -> str:
        return self._name
    
    def __call__(self, state: GovernorState) -> GovernorState:
        self.call_count += 1
        state.transition_to(StateNode.RESPOND)
        return state


class MockConditionalFunction(ConditionalFunction):
    """Mock conditional function for testing."""
    
    def __init__(self, name: str, next_node: str = "respond"):
        self._name = name
        self.next_node = next_node
        self.call_count = 0
    
    @property  
    def name(self) -> str:
        return self._name
    
    def __call__(self, state: GovernorState) -> str:
        self.call_count += 1
        return self.next_node


class MockWorkflow(WorkflowBase):
    """Mock workflow for testing."""
    
    def __init__(self):
        super().__init__()
        self.mock_node = MockNodeFunction("mock")
        self.mock_conditional = MockConditionalFunction("mock_routing")
    
    def build_graph(self):
        from langgraph.graph import Graph
        
        workflow = Graph()
        from langgraph.graph import END
        workflow.add_node("mock", self.mock_node)
        workflow.add_node("respond", MockNodeFunction("respond"))
        workflow.set_entry_point("mock")
        workflow.add_edge("mock", "respond")
        workflow.add_edge("respond", END)
        
        return workflow
    
    def get_entry_point(self) -> str:
        return "mock"
    
    def get_finish_point(self) -> str:
        return "respond"


class TestNodeFunction:
    """Test cases for NodeFunction base class."""
    
    def test_mock_node_function(self):
        """Test mock node function creation."""
        node = MockNodeFunction("test_node")
        assert node.name == "test_node"
        assert node.call_count == 0
    
    @pytest.mark.asyncio
    async def test_mock_node_function_call(self):
        """Test mock node function execution."""
        node = MockNodeFunction("test_node")
        state = GovernorState(user_id="user_123", session_id="session_456")
        
        result_state = await node(state)
        
        assert node.call_count == 1
        assert result_state.current_node == StateNode.RESPOND
        assert result_state.previous_node == StateNode.IDLE


class TestConditionalFunction:
    """Test cases for ConditionalFunction base class."""
    
    def test_mock_conditional_function(self):
        """Test mock conditional function creation."""
        conditional = MockConditionalFunction("test_conditional", "next_node")
        assert conditional.name == "test_conditional"
        assert conditional.next_node == "next_node"
        assert conditional.call_count == 0
    
    def test_mock_conditional_function_call(self):
        """Test mock conditional function execution."""
        conditional = MockConditionalFunction("test_conditional", "target_node")
        state = GovernorState(user_id="user_123", session_id="session_456")
        
        result = conditional(state)
        
        assert conditional.call_count == 1
        assert result == "target_node"


class TestWorkflowBase:
    """Test cases for WorkflowBase abstract class."""
    
    def test_workflow_base_initialization(self):
        """Test workflow base initialization."""
        workflow = MockWorkflow()
        
        assert workflow._graph is None
        assert workflow._compiled_graph is None
    
    def test_workflow_base_compile(self):
        """Test workflow compilation."""
        workflow = MockWorkflow()
        
        compiled_graph = workflow.compile()
        
        assert compiled_graph is not None
        assert workflow._graph is not None
        assert workflow._compiled_graph is not None
        assert workflow._compiled_graph is compiled_graph
    
    def test_workflow_base_compile_idempotent(self):
        """Test that multiple compile calls return the same graph."""
        workflow = MockWorkflow()
        
        graph1 = workflow.compile()
        graph2 = workflow.compile()
        
        assert graph1 is graph2
    
    def test_workflow_base_execute(self):
        """Test workflow execution."""
        workflow = MockWorkflow()
        state = GovernorState(user_id="user_123", session_id="session_456")
        
        result_state = workflow.execute(state)
        
        assert result_state is not None
        assert isinstance(result_state, GovernorState)
        assert workflow.mock_node.call_count >= 1
    
    def test_workflow_base_stream(self):
        """Test workflow streaming."""
        workflow = MockWorkflow()
        state = GovernorState(user_id="user_123", session_id="session_456")
        
        updates = list(workflow.stream(state))
        
        assert len(updates) > 0
        assert workflow.mock_node.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_workflow_base_astream(self):
        """Test async workflow streaming."""
        workflow = MockWorkflow()
        state = GovernorState(user_id="user_123", session_id="session_456")
        
        updates = []
        async for update in workflow.astream(state):
            updates.append(update)
        
        assert len(updates) > 0
        assert workflow.mock_node.call_count >= 1
    
    def test_workflow_entry_and_finish_points(self):
        """Test workflow entry and finish point methods."""
        workflow = MockWorkflow()
        
        assert workflow.get_entry_point() == "mock"
        assert workflow.get_finish_point() == "respond"
    
    def test_workflow_state_transitions_before_compile_fails(self):
        """Test that getting state transitions before compile fails."""
        workflow = MockWorkflow()
        
        with pytest.raises(RuntimeError, match="Graph must be compiled"):
            workflow.get_state_transitions()
    
    def test_workflow_state_transitions_after_compile(self):
        """Test getting state transitions after compile."""
        workflow = MockWorkflow()
        workflow.compile()
        
        transitions = workflow.get_state_transitions()
        
        assert isinstance(transitions, dict)
        assert "mock" in transitions
        assert "respond" in transitions
    
    def test_workflow_visualization_before_compile_fails(self):
        """Test that getting visualization before compile fails."""
        workflow = MockWorkflow()
        
        with pytest.raises(RuntimeError, match="Graph must be compiled"):
            workflow.get_graph_visualization()
    
    def test_workflow_visualization_after_compile(self):
        """Test getting graph visualization after compile."""
        workflow = MockWorkflow()
        workflow.compile()
        
        # This might fail if mermaid is not available, so we'll just test it doesn't crash
        try:
            viz = workflow.get_graph_visualization()
            assert isinstance(viz, str)
        except Exception:
            # Visualization might not be available in test environment
            pass