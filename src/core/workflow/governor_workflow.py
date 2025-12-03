"""Governor workflow implementation using LangGraph.

This module implements the complete 7-state Governor workflow that handles
conversation flow from user input through analysis, tool execution, and response.
"""

from typing import Any

from langgraph.graph import Graph, END

from ..domain.state import GovernorState, StateNode
from .base import WorkflowBase
from .nodes import (
    AnalyzeNode, AnalyzeConditional,
    AwaitConfirmationNode, AwaitConfirmationConditional,
    ExecuteNode, ExecuteConditional,
    IdleNode, IdleConditional,
    PolicyCheckNode, PolicyCheckConditional,
    RespondNode, RespondConditional,
    ToolDecisionNode, ToolDecisionConditional,
)


class GovernorWorkflow(WorkflowBase):
    """Complete Governor workflow implementation.
    
    This workflow implements the 7-state conversation flow:
    1. Idle - Wait for user input and initial processing
    2. Analyze - Understand user intent and requirements  
    3. Tool Decision - Select appropriate tools to execute
    4. Policy Check - Evaluate security policies and permissions
    5. Execute - Run approved tools with error handling
    6. Await Confirmation - Handle user confirmation requests
    7. Respond - Generate and deliver responses to users
    """
    
    def __init__(self) -> None:
        """Initialize the Governor workflow."""
        super().__init__()
        
        # Initialize node instances
        self.idle_node = IdleNode()
        self.analyze_node = AnalyzeNode()
        self.tool_decision_node = ToolDecisionNode()
        self.policy_check_node = PolicyCheckNode()
        self.execute_node = ExecuteNode()
        self.await_confirmation_node = AwaitConfirmationNode()
        self.respond_node = RespondNode()
        
        # Initialize conditional functions
        self.idle_conditional = IdleConditional()
        self.analyze_conditional = AnalyzeConditional()
        self.tool_decision_conditional = ToolDecisionConditional()
        self.policy_check_conditional = PolicyCheckConditional()
        self.execute_conditional = ExecuteConditional()
        self.await_confirmation_conditional = AwaitConfirmationConditional()
        self.respond_conditional = RespondConditional()
    
    def build_graph(self) -> Graph:
        """Build the complete Governor workflow graph.
        
        Returns:
            Configured StateGraph with all nodes and edges
        """
        # Create the state graph  
        workflow = Graph()
        
        # Add all workflow nodes
        workflow.add_node(StateNode.IDLE.value, self.idle_node)
        workflow.add_node(StateNode.ANALYZE.value, self.analyze_node)
        workflow.add_node(StateNode.TOOL_DECISION.value, self.tool_decision_node)
        workflow.add_node(StateNode.POLICY_CHECK.value, self.policy_check_node)
        workflow.add_node(StateNode.EXECUTE.value, self.execute_node)
        workflow.add_node(StateNode.AWAIT_CONFIRMATION.value, self.await_confirmation_node)
        workflow.add_node(StateNode.RESPOND.value, self.respond_node)
        
        # Set the entry point
        workflow.set_entry_point(StateNode.IDLE.value)
        
        # Add conditional edges for dynamic routing
        workflow.add_conditional_edges(
            StateNode.IDLE.value,
            self.idle_conditional,
            {
                StateNode.IDLE.value: StateNode.IDLE.value,
                StateNode.ANALYZE.value: StateNode.ANALYZE.value,
                StateNode.AWAIT_CONFIRMATION.value: StateNode.AWAIT_CONFIRMATION.value,
                StateNode.RESPOND.value: StateNode.RESPOND.value,
            }
        )
        
        workflow.add_conditional_edges(
            StateNode.ANALYZE.value,
            self.analyze_conditional,
            {
                StateNode.TOOL_DECISION.value: StateNode.TOOL_DECISION.value,
                StateNode.RESPOND.value: StateNode.RESPOND.value,
            }
        )
        
        workflow.add_conditional_edges(
            StateNode.TOOL_DECISION.value,
            self.tool_decision_conditional,
            {
                StateNode.POLICY_CHECK.value: StateNode.POLICY_CHECK.value,
                StateNode.EXECUTE.value: StateNode.EXECUTE.value,
                StateNode.RESPOND.value: StateNode.RESPOND.value,
            }
        )
        
        workflow.add_conditional_edges(
            StateNode.POLICY_CHECK.value,
            self.policy_check_conditional,
            {
                StateNode.EXECUTE.value: StateNode.EXECUTE.value,
                StateNode.AWAIT_CONFIRMATION.value: StateNode.AWAIT_CONFIRMATION.value,
                StateNode.RESPOND.value: StateNode.RESPOND.value,
            }
        )
        
        workflow.add_conditional_edges(
            StateNode.EXECUTE.value,
            self.execute_conditional,
            {
                StateNode.AWAIT_CONFIRMATION.value: StateNode.AWAIT_CONFIRMATION.value,
                StateNode.RESPOND.value: StateNode.RESPOND.value,
            }
        )
        
        workflow.add_conditional_edges(
            StateNode.AWAIT_CONFIRMATION.value,
            self.await_confirmation_conditional,
            {
                StateNode.ANALYZE.value: StateNode.ANALYZE.value,
                StateNode.EXECUTE.value: StateNode.EXECUTE.value,
                StateNode.RESPOND.value: StateNode.RESPOND.value,
                StateNode.AWAIT_CONFIRMATION.value: StateNode.AWAIT_CONFIRMATION.value,
            }
        )
        
        workflow.add_conditional_edges(
            StateNode.RESPOND.value,
            self.respond_conditional,
            {
                StateNode.IDLE.value: StateNode.IDLE.value,
                StateNode.AWAIT_CONFIRMATION.value: StateNode.AWAIT_CONFIRMATION.value,
                END: END,
            }
        )
        
        return workflow
    
    def get_entry_point(self) -> str:
        """Get the workflow entry point.
        
        Returns:
            Name of the entry node
        """
        return StateNode.IDLE.value
    
    def get_finish_point(self) -> str:
        """Get the workflow finish point.
        
        Returns:
            Name of the finish node
        """
        return StateNode.IDLE.value  # Governor workflow loops back to idle
    
    async def process_user_input(
        self, 
        user_input: str, 
        user_id: str, 
        session_id: str,
        existing_state: GovernorState | None = None
    ) -> GovernorState:
        """Process user input through the complete workflow.
        
        Args:
            user_input: User's message or command
            user_id: Unique user identifier
            session_id: Session identifier
            existing_state: Existing state to continue from
            
        Returns:
            Final state after processing
        """
        # Create or update state
        if existing_state:
            state = existing_state
            state.last_user_input = user_input
        else:
            state = GovernorState(
                user_id=user_id,
                session_id=session_id,
                last_user_input=user_input
            )
        
        # Execute the workflow
        final_state = await self.aexecute(state)
        
        # Add conversation turn
        if final_state.last_assistant_response:
            final_state.add_conversation_turn(
                user_input=user_input,
                assistant_response=final_state.last_assistant_response,
                tools_used=self._extract_tools_used(final_state)
            )
        
        return final_state
    
    def _extract_tools_used(self, state: GovernorState) -> list[str]:
        """Extract list of tools that were successfully executed.
        
        Args:
            state: Final workflow state
            
        Returns:
            List of tool names that were executed
        """
        execution_results = state.context.get("execution_results", {})
        results = execution_results.get("results", [])
        
        tools_used = []
        for result in results:
            if result.get("status") == "success":
                tools_used.append(result.get("tool_name", "unknown"))
        
        return tools_used
    
    async def handle_confirmation_response(
        self,
        response: str,
        state: GovernorState
    ) -> GovernorState:
        """Handle user response to confirmation request.
        
        Args:
            response: User's confirmation response
            state: Current workflow state
            
        Returns:
            Updated state after processing confirmation
        """
        # Update state with confirmation response
        state.last_user_input = response
        
        # Continue workflow from current state
        return await self.aexecute(state)
    
    def get_workflow_status(self, state: GovernorState) -> dict[str, Any]:
        """Get current workflow status and metrics.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dictionary containing workflow status information
        """
        return {
            "current_node": state.current_node.value,
            "previous_node": state.previous_node.value if state.previous_node else None,
            "awaiting_confirmation": state.awaiting_confirmation,
            "pending_tools_count": len(state.pending_tools),
            "executing_tool": state.executing_tool.tool_name if state.executing_tool else None,
            "active_tasks_count": len(state.active_tasks),
            "total_turns": state.total_turns,
            "total_tools_executed": state.total_tools_executed,
            "error_count": state.error_count,
            "session_age_hours": (state.updated_at - state.created_at).total_seconds() / 3600,
            "average_response_time_ms": state.average_response_time_ms
        }
    
    def get_conversation_summary(self, state: GovernorState) -> dict[str, Any]:
        """Get a summary of the conversation history.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dictionary containing conversation summary
        """
        recent_turns = state.conversation_history[-5:] if state.conversation_history else []
        
        return {
            "total_turns": state.total_turns,
            "recent_turns": [
                {
                    "user_input": turn.user_input,
                    "assistant_response": turn.assistant_response,
                    "tools_used": turn.tools_used,
                    "timestamp": turn.timestamp.isoformat()
                }
                for turn in recent_turns
            ],
            "session_started": state.created_at.isoformat(),
            "last_activity": state.updated_at.isoformat(),
            "total_tools_executed": state.total_tools_executed,
            "success_rate": (
                1.0 - (state.error_count / max(state.total_turns, 1))
                if state.total_turns > 0 else 1.0
            )
        }
    
    def reset_session(self, state: GovernorState) -> GovernorState:
        """Reset the session to a clean state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Reset state ready for new conversation
        """
        # Preserve user and session IDs but reset everything else
        return GovernorState(
            user_id=state.user_id,
            session_id=state.session_id,
            current_node=StateNode.IDLE
        )