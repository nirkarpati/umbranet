"""State models for the Governor system.

These models define the state management structures that control
the conversation flow through the LangGraph state machine.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, validator

from .tools import ToolCall


class StateNode(str, Enum):
    """The 7 states of the Governor state machine.
    
    These represent the deterministic flow that every conversation
    follows through the LangGraph workflow.
    """
    
    IDLE = "idle"
    ANALYZE = "analyze"
    TOOL_DECISION = "tool_decision"
    POLICY_CHECK = "policy_check"
    EXECUTE = "execute"
    AWAIT_CONFIRMATION = "await_confirmation"
    RESPOND = "respond"


class ConversationTurn(BaseModel):
    """A single turn in the conversation history."""
    
    turn_id: str = Field(..., description="Unique identifier for this turn")
    user_input: str = Field(..., description="What the user said")
    assistant_response: str = Field(..., description="What the assistant responded")
    tools_used: list[str] = Field(default_factory=list, description="Tools executed this turn")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActiveTask(BaseModel):
    """Represents an active background task or process."""
    
    task_id: str = Field(..., description="Unique identifier for this task")
    task_type: str = Field(..., description="Type of task (reminder, automation, etc)")
    description: str = Field(..., description="Human-readable task description")
    status: str = Field(default="active", description="Task status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    scheduled_for: datetime | None = Field(None, description="When task should execute")
    metadata: dict[str, Any] = Field(default_factory=dict)


class GovernorState(BaseModel):
    """Current state of a conversation flow through the Governor system.
    
    This represents the complete context and state for a user's session
    as it moves through the LangGraph state machine.
    """
    
    # Core identifiers
    user_id: str = Field(
        ...,
        description="Unique identifier for the user"
    )
    
    session_id: str = Field(
        ...,
        description="Unique identifier for this conversation session"
    )
    
    # Current state information
    current_node: StateNode = Field(
        default=StateNode.IDLE,
        description="Current position in the state machine"
    )
    
    previous_node: StateNode | None = Field(
        None,
        description="Previous state for debugging/recovery"
    )
    
    # Context and conversation data
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Dynamic context data for prompt assembly"
    )
    
    conversation_history: list[ConversationTurn] = Field(
        default_factory=list,
        description="Recent conversation turns for context"
    )
    
    # Tool execution state
    pending_tools: list[ToolCall] = Field(
        default_factory=list,
        description="Tools queued for execution"
    )
    
    executing_tool: ToolCall | None = Field(
        None,
        description="Currently executing tool"
    )
    
    # Confirmation flow state
    awaiting_confirmation: bool = Field(
        default=False,
        description="Whether system is waiting for user confirmation"
    )
    
    confirmation_context: dict[str, Any] | None = Field(
        None,
        description="Context for pending confirmation"
    )
    
    # Active processes and tasks
    active_tasks: list[ActiveTask] = Field(
        default_factory=list,
        description="Background tasks and processes"
    )
    
    # Timing and metadata
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this state was first created"
    )
    
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this state was last updated"
    )
    
    last_user_input: str | None = Field(
        None,
        description="The most recent user message"
    )
    
    last_assistant_response: str | None = Field(
        None,
        description="The most recent assistant response"
    )
    
    # Error handling and recovery
    error_count: int = Field(
        default=0,
        description="Number of errors in this session"
    )
    
    last_error: str | None = Field(
        None,
        description="Most recent error message"
    )
    
    # Performance and monitoring
    total_turns: int = Field(
        default=0,
        description="Total number of conversation turns"
    )
    
    total_tools_executed: int = Field(
        default=0,
        description="Total number of tools executed"
    )
    
    average_response_time_ms: float | None = Field(
        None,
        description="Average response time for this session"
    )
    
    @validator('conversation_history')
    def validate_history_length(cls, v: list[ConversationTurn]) -> list[ConversationTurn]:
        """Limit conversation history to reasonable size."""
        max_history = 50  # Keep last 50 turns
        if len(v) > max_history:
            return v[-max_history:]
        return v
    
    @validator('current_node')
    def validate_state_transition(cls, v: StateNode, values: dict[str, Any]) -> StateNode:
        """Basic validation of state transitions."""
        # Add more sophisticated validation logic here as needed
        return v
    
    def add_conversation_turn(
        self, 
        user_input: str, 
        assistant_response: str,
        tools_used: list[str] | None = None
    ) -> None:
        """Add a new conversation turn to the history."""
        turn = ConversationTurn(
            turn_id=f"{self.session_id}_turn_{self.total_turns + 1}",
            user_input=user_input,
            assistant_response=assistant_response,
            tools_used=tools_used or []
        )
        
        self.conversation_history.append(turn)
        
        # Limit conversation history to 50 turns
        max_history = 50
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]
        
        self.total_turns += 1
        self.last_user_input = user_input
        self.last_assistant_response = assistant_response
        self.updated_at = datetime.utcnow()
    
    def transition_to(self, new_state: StateNode) -> None:
        """Safely transition to a new state."""
        self.previous_node = self.current_node
        self.current_node = new_state
        self.updated_at = datetime.utcnow()
    
    def add_tool_call(self, tool_call: ToolCall) -> None:
        """Add a tool call to the pending queue."""
        self.pending_tools.append(tool_call)
        self.updated_at = datetime.utcnow()
    
    def start_tool_execution(self, tool_call: ToolCall) -> None:
        """Mark a tool as currently executing."""
        self.executing_tool = tool_call
        if tool_call in self.pending_tools:
            self.pending_tools.remove(tool_call)
        self.updated_at = datetime.utcnow()
    
    def complete_tool_execution(self, success: bool = True) -> None:
        """Mark the current tool execution as complete."""
        if self.executing_tool:
            if success:
                self.total_tools_executed += 1
            self.executing_tool = None
        self.updated_at = datetime.utcnow()
    
    def request_confirmation(self, context: dict[str, Any]) -> None:
        """Enter confirmation waiting state."""
        self.awaiting_confirmation = True
        self.confirmation_context = context
        self.updated_at = datetime.utcnow()
    
    def resolve_confirmation(self, approved: bool) -> None:
        """Resolve pending confirmation."""
        self.awaiting_confirmation = False
        self.confirmation_context = None
        self.updated_at = datetime.utcnow()
    
    def add_active_task(self, task: ActiveTask) -> None:
        """Add a new active task."""
        self.active_tasks.append(task)
        self.updated_at = datetime.utcnow()
    
    def remove_active_task(self, task_id: str) -> bool:
        """Remove an active task by ID."""
        original_length = len(self.active_tasks)
        self.active_tasks = [t for t in self.active_tasks if t.task_id != task_id]
        if len(self.active_tasks) < original_length:
            self.updated_at = datetime.utcnow()
            return True
        return False
    
    def record_error(self, error_message: str) -> None:
        """Record an error in this session."""
        self.error_count += 1
        self.last_error = error_message
        self.updated_at = datetime.utcnow()
    
    def update_response_time(self, response_time_ms: float) -> None:
        """Update the average response time calculation."""
        if self.average_response_time_ms is None:
            self.average_response_time_ms = response_time_ms
        else:
            # Exponential moving average with alpha = 0.1
            alpha = 0.1
            self.average_response_time_ms = (
                alpha * response_time_ms + 
                (1 - alpha) * self.average_response_time_ms
            )
        self.updated_at = datetime.utcnow()
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "session_id": "telegram_user_123_20241127",
                "current_node": "analyze",
                "context": {
                    "user_timezone": "America/Los_Angeles",
                    "preferred_language": "en"
                },
                "conversation_history": [
                    {
                        "turn_id": "telegram_user_123_20241127_turn_1",
                        "user_input": "What's the weather?",
                        "assistant_response": "I'll check the weather for you.",
                        "tools_used": ["get_weather"],
                        "timestamp": "2024-11-27T18:30:00Z"
                    }
                ],
                "total_turns": 1,
                "total_tools_executed": 1,
                "average_response_time_ms": 485.2
            }
        }