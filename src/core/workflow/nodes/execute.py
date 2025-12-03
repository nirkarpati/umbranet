"""Execute node implementation.

The execute node handles the actual execution of approved tools,
managing timeouts, retries, and error handling.
"""

import asyncio
from datetime import datetime
from typing import Any

from ...domain.state import GovernorState, StateNode
from ...domain.tools import ToolCall, ToolStatus
from ..base import NodeFunction


class ExecuteNode(NodeFunction):
    """Node that executes approved tool calls.
    
    This node handles the actual execution of tools, including
    timeout management, retry logic, and error handling.
    """
    
    @property
    def name(self) -> str:
        """Get the node name."""
        return StateNode.EXECUTE.value
    
    async def __call__(self, state: GovernorState) -> GovernorState:
        """Execute approved tool calls.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with execution results
        """
        if not state.pending_tools:
            state.record_error("No pending tools to execute")
            state.transition_to(StateNode.RESPOND)
            return state
        
        execution_results = []
        successful_executions = 0
        failed_executions = 0
        
        # Execute tools sequentially for now
        # In production, could execute safe tools in parallel
        for tool_call in state.pending_tools[:]:  # Create copy to modify during iteration
            try:
                # Only execute tools that don't require confirmation
                # or have been confirmed
                if not tool_call.requires_confirmation or self._is_tool_confirmed(tool_call, state):
                    result = await self._execute_tool_call(tool_call, state)
                    execution_results.append(result)
                    
                    if result["status"] == "success":
                        successful_executions += 1
                        state.complete_tool_execution(success=True)
                    else:
                        failed_executions += 1
                        state.complete_tool_execution(success=False)
                    
                    # Remove from pending after execution
                    state.pending_tools.remove(tool_call)
                else:
                    # Tool still requires confirmation, skip for now
                    continue
                    
            except Exception as e:
                # Handle unexpected execution errors
                error_msg = f"Unexpected error executing {tool_call.tool_name}: {str(e)}"
                state.record_error(error_msg)
                
                tool_call.fail_execution(error_msg)
                failed_executions += 1
                
                execution_results.append({
                    "tool_name": tool_call.tool_name,
                    "execution_id": tool_call.execution_id,
                    "status": "error",
                    "error": error_msg
                })
                
                if tool_call in state.pending_tools:
                    state.pending_tools.remove(tool_call)
        
        # Store execution results
        state.context.update({
            "execution_results": {
                "total_executed": len(execution_results),
                "successful_count": successful_executions,
                "failed_count": failed_executions,
                "results": execution_results,
                "execution_completed_at": datetime.utcnow().isoformat()
            }
        })
        
        # Determine next state
        if state.pending_tools:
            # Still have tools pending confirmation
            state.transition_to(StateNode.AWAIT_CONFIRMATION)
        else:
            # All tools processed, move to response
            state.transition_to(StateNode.RESPOND)
        
        return state
    
    async def _execute_tool_call(self, tool_call: ToolCall, state: GovernorState) -> dict[str, Any]:
        """Execute a single tool call with timeout and error handling.
        
        Args:
            tool_call: Tool call to execute
            state: Current workflow state
            
        Returns:
            Execution result dictionary
        """
        state.start_tool_execution(tool_call)
        tool_call.start_execution()
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._run_tool_function(tool_call),
                timeout=tool_call.timeout_seconds
            )
            
            tool_call.complete_execution(result)
            
            return {
                "tool_name": tool_call.tool_name,
                "execution_id": tool_call.execution_id,
                "status": "success",
                "result": result,
                "execution_time_ms": tool_call.execution_time_ms
            }
            
        except asyncio.TimeoutError:
            error_msg = f"Tool execution timed out after {tool_call.timeout_seconds} seconds"
            tool_call.fail_execution(error_msg)
            tool_call.status = ToolStatus.TIMEOUT
            
            return {
                "tool_name": tool_call.tool_name,
                "execution_id": tool_call.execution_id,
                "status": "timeout",
                "error": error_msg,
                "timeout_seconds": tool_call.timeout_seconds
            }
            
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            tool_call.fail_execution(error_msg)
            
            # Check if retry is possible
            if tool_call.increment_retry():
                # Schedule for retry
                return await self._retry_tool_execution(tool_call, state)
            
            return {
                "tool_name": tool_call.tool_name,
                "execution_id": tool_call.execution_id,
                "status": "failed",
                "error": error_msg,
                "retry_count": tool_call.retry_count
            }
    
    async def _run_tool_function(self, tool_call: ToolCall) -> dict[str, Any]:
        """Run the actual tool function.
        
        Args:
            tool_call: Tool call to execute
            
        Returns:
            Tool execution result
            
        Note:
            This is a simulation of tool execution. In production,
            this would interface with actual tool implementations.
        """
        # Simulate tool execution based on tool type
        tool_name = tool_call.tool_name
        arguments = tool_call.arguments
        
        # Simulate execution delay
        await asyncio.sleep(0.1)  # Small delay to simulate real work
        
        if tool_name == "weather":
            return await self._execute_weather_tool(arguments)
        elif tool_name == "email":
            return await self._execute_email_tool(arguments)
        elif tool_name == "calendar":
            return await self._execute_calendar_tool(arguments)
        elif tool_name == "search":
            return await self._execute_search_tool(arguments)
        elif tool_name == "file_manager":
            return await self._execute_file_manager_tool(arguments)
        elif tool_name == "calculator":
            return await self._execute_calculator_tool(arguments)
        elif tool_name == "context_analyzer":
            return await self._execute_context_analyzer_tool(arguments)
        elif tool_name == "memory_search":
            return await self._execute_memory_search_tool(arguments)
        else:
            return await self._execute_generic_tool(tool_name, arguments)
    
    async def _execute_weather_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Simulate weather tool execution."""
        location = arguments.get("location", "unknown location")
        
        # Simulate API call delay
        await asyncio.sleep(0.5)
        
        return {
            "location": location,
            "temperature": "72°F",
            "condition": "Partly cloudy",
            "humidity": "65%",
            "wind": "5 mph NW",
            "forecast": "Clear skies expected through the afternoon"
        }
    
    async def _execute_email_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Simulate email tool execution."""
        recipients = arguments.get("recipients", [])
        message = arguments.get("message", "")
        
        # Simulate sending delay
        await asyncio.sleep(1.0)
        
        return {
            "status": "sent",
            "recipients": recipients,
            "message_id": f"msg_{hash(message + str(datetime.utcnow())) % 10000}",
            "sent_at": datetime.utcnow().isoformat(),
            "delivery_status": "delivered"
        }
    
    async def _execute_calendar_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Simulate calendar tool execution."""
        description = arguments.get("description", "")
        date = arguments.get("date", "")
        time = arguments.get("time", "")
        
        # Simulate calendar API delay
        await asyncio.sleep(0.3)
        
        return {
            "event_id": f"evt_{hash(description + date + time) % 10000}",
            "title": description,
            "date": date,
            "time": time,
            "status": "created",
            "calendar": "primary"
        }
    
    async def _execute_search_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Simulate search tool execution."""
        query = arguments.get("query", "")
        
        # Simulate search delay
        await asyncio.sleep(0.8)
        
        return {
            "query": query,
            "results_count": 3,
            "results": [
                {
                    "title": f"Result 1 for '{query}'",
                    "url": "https://example.com/result1",
                    "snippet": "This is a relevant search result..."
                },
                {
                    "title": f"Result 2 for '{query}'", 
                    "url": "https://example.com/result2",
                    "snippet": "Another relevant search result..."
                },
                {
                    "title": f"Result 3 for '{query}'",
                    "url": "https://example.com/result3", 
                    "snippet": "A third relevant search result..."
                }
            ]
        }
    
    async def _execute_file_manager_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Simulate file manager tool execution."""
        operation = arguments.get("operation", "")
        
        # Simulate file operation delay
        await asyncio.sleep(0.4)
        
        return {
            "operation": operation,
            "status": "completed",
            "files_affected": 1,
            "path": "/home/user/documents/",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _execute_calculator_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Simulate calculator tool execution."""
        expression = arguments.get("expression", "")
        
        # Simple math evaluation simulation
        try:
            # Very basic math parsing for demo
            if "+" in expression:
                parts = expression.split("+")
                if len(parts) == 2:
                    result = float(parts[0].strip()) + float(parts[1].strip())
                else:
                    result = "Error: Complex expression"
            elif "multiply" in expression or "*" in expression:
                result = "42"  # Placeholder
            else:
                result = "42"  # Default placeholder result
        except:
            result = "Error: Invalid expression"
        
        return {
            "expression": expression,
            "result": result,
            "calculation_type": "basic_math"
        }
    
    async def _execute_context_analyzer_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Simulate context analyzer tool execution."""
        text = arguments.get("text", "")
        
        await asyncio.sleep(0.3)
        
        return {
            "text_length": len(text),
            "complexity": "medium",
            "topics": ["general", "request"],
            "sentiment": "neutral",
            "key_phrases": ["help", "need", "please"]
        }
    
    async def _execute_memory_search_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Simulate memory search tool execution."""
        query = arguments.get("query", "")
        
        await asyncio.sleep(0.6)
        
        return {
            "query": query,
            "matches_found": 2,
            "matches": [
                {
                    "relevance": 0.8,
                    "content": "Previous conversation snippet...",
                    "timestamp": "2024-11-26T15:30:00Z"
                },
                {
                    "relevance": 0.6,
                    "content": "Earlier discussion about...",
                    "timestamp": "2024-11-25T10:15:00Z"
                }
            ]
        }
    
    async def _execute_generic_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Simulate generic tool execution."""
        await asyncio.sleep(0.2)
        
        return {
            "tool_name": tool_name,
            "arguments": arguments,
            "status": "executed",
            "result": f"Generic result for {tool_name}",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _retry_tool_execution(self, tool_call: ToolCall, state: GovernorState) -> dict[str, Any]:
        """Attempt to retry a failed tool execution.
        
        Args:
            tool_call: Tool call to retry
            state: Current workflow state
            
        Returns:
            Retry execution result
        """
        # Add exponential backoff delay
        delay = min(2 ** (tool_call.retry_count - 1), 10)  # Max 10 second delay
        await asyncio.sleep(delay)
        
        # Reset status for retry
        tool_call.status = ToolStatus.PENDING
        
        # Attempt execution again
        return await self._execute_tool_call(tool_call, state)
    
    def _is_tool_confirmed(self, tool_call: ToolCall, state: GovernorState) -> bool:
        """Check if a tool has been confirmed for execution.
        
        Args:
            tool_call: Tool call to check
            state: Current workflow state
            
        Returns:
            True if tool is confirmed for execution
        """
        # Check if we're not awaiting confirmation
        if not state.awaiting_confirmation:
            return True
        
        # Check confirmation context
        confirmation_context = state.confirmation_context or {}
        confirmed_tools = confirmation_context.get("confirmed_tools", [])
        
        return tool_call.execution_id in confirmed_tools


class ExecuteConditional:
    """Conditional routing from execute state."""
    
    @property
    def name(self) -> str:
        """Get the conditional name."""
        return "execute_routing"
    
    def __call__(self, state: GovernorState) -> str:
        """Determine next node from execute state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node name
        """
        # If there are still pending tools requiring confirmation
        if state.pending_tools and any(tool.requires_confirmation for tool in state.pending_tools):
            return StateNode.AWAIT_CONFIRMATION.value
        
        # If all tools are processed, go to respond
        return StateNode.RESPOND.value