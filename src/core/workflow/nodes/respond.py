"""Respond node implementation.

The respond node generates appropriate responses to users based on
the current state, execution results, and conversation context.
"""

from datetime import datetime
from typing import Any

from ...domain.events import GovernorResponse, ResponseType
from ...domain.state import GovernorState, StateNode
from ..base import NodeFunction


class RespondNode(NodeFunction):
    """Node that generates responses to users.
    
    This node creates appropriate responses based on the workflow state,
    tool execution results, errors, and confirmation requests.
    """
    
    @property
    def name(self) -> str:
        """Get the node name."""
        return StateNode.RESPOND.value
    
    async def __call__(self, state: GovernorState) -> GovernorState:
        """Generate appropriate response to the user.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with response ready for delivery
        """
        try:
            # Generate response based on current state and context
            response = await self._generate_response(state)
            
            # Store response in state context
            state.context.update({
                "generated_response": {
                    "content": response.content,
                    "response_type": response.response_type.value,
                    "urgency_level": response.urgency_level,
                    "requires_confirmation": response.requires_confirmation,
                    "generated_at": datetime.utcnow().isoformat()
                }
            })
            
            # Update state with response
            state.last_assistant_response = response.content
            
            # Calculate response time
            if state.context.get("processing_started_at"):
                start_time = datetime.fromisoformat(state.context["processing_started_at"])
                response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                state.update_response_time(response_time)
            
            # Transition back to idle for next interaction
            state.transition_to(StateNode.IDLE)
            
        except Exception as e:
            # Handle response generation errors
            error_msg = f"Error generating response: {str(e)}"
            state.record_error(error_msg)
            
            # Generate fallback error response
            state.last_assistant_response = "I apologize, but I encountered an error while processing your request. Please try again."
            state.transition_to(StateNode.IDLE)
        
        return state
    
    async def _generate_response(self, state: GovernorState) -> GovernorResponse:
        """Generate appropriate response based on state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Generated response object
        """
        # Determine response type based on context
        response_reason = state.context.get("response_reason", "")
        
        if state.awaiting_confirmation:
            return self._generate_confirmation_request(state)
        elif response_reason == "confirmation_timeout":
            return self._generate_timeout_response(state)
        elif response_reason == "user_denied_confirmation":
            return self._generate_denial_response(state)
        elif state.context.get("needs_clarification"):
            return self._generate_clarification_request(state)
        elif state.error_count > 0 and state.last_error:
            return self._generate_error_response(state)
        elif state.context.get("execution_results"):
            return self._generate_execution_response(state)
        elif state.context.get("analysis"):
            return self._generate_analysis_response(state)
        else:
            return self._generate_default_response(state)
    
    def _generate_confirmation_request(self, state: GovernorState) -> GovernorResponse:
        """Generate confirmation request response.
        
        Args:
            state: Current workflow state
            
        Returns:
            Confirmation request response
        """
        confirmation_context = state.confirmation_context or {}
        confirmation_type = confirmation_context.get("confirmation_type", "action")
        
        if confirmation_type == "tool_execution":
            tools = confirmation_context.get("tools_awaiting_confirmation", [])
            
            if len(tools) == 1:
                tool = tools[0]
                content = (
                    f"I need your confirmation to execute the '{tool['tool_name']}' tool. "
                    f"This is a {tool['risk_level']} operation. "
                    f"Please respond with 'yes' to proceed or 'no' to cancel."
                )
            else:
                tool_names = [tool["tool_name"] for tool in tools]
                content = (
                    f"I need your confirmation to execute {len(tools)} tools: "
                    f"{', '.join(tool_names)}. "
                    f"These operations have elevated risk levels. "
                    f"Please respond with 'yes' to proceed or 'no' to cancel."
                )
        else:
            content = (
                "I need your confirmation to proceed with this action. "
                "Please respond with 'yes' to continue or 'no' to cancel."
            )
        
        return GovernorResponse(
            user_id=state.user_id,
            session_id=state.session_id,
            content=content,
            response_type=ResponseType.CONFIRMATION_REQUEST,
            channel=self._get_response_channel(state),
            requires_confirmation=False,  # This IS the confirmation request
            urgency_level="high",
            metadata={
                "confirmation_type": confirmation_type,
                "timeout_minutes": confirmation_context.get("timeout_minutes", 10)
            }
        )
    
    def _generate_timeout_response(self, state: GovernorState) -> GovernorResponse:
        """Generate timeout response.
        
        Args:
            state: Current workflow state
            
        Returns:
            Timeout response
        """
        content = (
            "I didn't receive a response within the timeout period, "
            "so I've cancelled the pending operation. "
            "Please let me know if you'd like to try again."
        )
        
        return GovernorResponse(
            user_id=state.user_id,
            session_id=state.session_id,
            content=content,
            response_type=ResponseType.TEXT,
            channel=self._get_response_channel(state),
            urgency_level="normal"
        )
    
    def _generate_denial_response(self, state: GovernorState) -> GovernorResponse:
        """Generate denial response.
        
        Args:
            state: Current workflow state
            
        Returns:
            Denial acknowledgment response
        """
        content = (
            "I understand you've decided not to proceed with that operation. "
            "The request has been cancelled. Is there anything else I can help you with?"
        )
        
        return GovernorResponse(
            user_id=state.user_id,
            session_id=state.session_id,
            content=content,
            response_type=ResponseType.TEXT,
            channel=self._get_response_channel(state)
        )
    
    def _generate_clarification_request(self, state: GovernorState) -> GovernorResponse:
        """Generate clarification request.
        
        Args:
            state: Current workflow state
            
        Returns:
            Clarification request response
        """
        attempt = state.context.get("clarification_attempt", 1)
        
        if attempt == 1:
            content = (
                "I'm not sure if you want me to proceed or not. "
                "Please respond with 'yes' to continue, 'no' to cancel, "
                "or let me know if you'd like to modify the request."
            )
        elif attempt == 2:
            content = (
                "I still need a clear response. "
                "Please type 'yes', 'no', or 'cancel' to let me know what you'd like to do."
            )
        else:
            content = (
                "I'm having trouble understanding your response. "
                "I'll cancel this operation now. Please start over if you need assistance."
            )
        
        return GovernorResponse(
            user_id=state.user_id,
            session_id=state.session_id,
            content=content,
            response_type=ResponseType.TEXT,
            channel=self._get_response_channel(state),
            urgency_level="high"
        )
    
    def _generate_error_response(self, state: GovernorState) -> GovernorResponse:
        """Generate error response.
        
        Args:
            state: Current workflow state
            
        Returns:
            Error response
        """
        error_message = state.last_error or "An unknown error occurred"
        
        # Provide user-friendly error messages
        if "timeout" in error_message.lower():
            content = (
                "I encountered a timeout while processing your request. "
                "This might be due to a slow connection or a busy service. "
                "Please try again in a moment."
            )
        elif "permission" in error_message.lower():
            content = (
                "I don't have the necessary permissions to complete that action. "
                "Please check your account settings or contact an administrator."
            )
        elif "validation" in error_message.lower():
            content = (
                "There was an issue with the information provided. "
                "Please check your request and try again with valid data."
            )
        else:
            content = (
                "I encountered an error while processing your request. "
                "I've logged the issue and will try to do better next time. "
                "Please try rephrasing your request or contact support if the problem persists."
            )
        
        return GovernorResponse(
            user_id=state.user_id,
            session_id=state.session_id,
            content=content,
            response_type=ResponseType.ERROR,
            channel=self._get_response_channel(state),
            urgency_level="normal",
            metadata={
                "error_count": state.error_count,
                "error_details": error_message
            }
        )
    
    def _generate_execution_response(self, state: GovernorState) -> GovernorResponse:
        """Generate response based on tool execution results.
        
        Args:
            state: Current workflow state
            
        Returns:
            Execution results response
        """
        execution_results = state.context.get("execution_results", {})
        successful_count = execution_results.get("successful_count", 0)
        failed_count = execution_results.get("failed_count", 0)
        results = execution_results.get("results", [])
        
        if failed_count > 0 and successful_count == 0:
            # All executions failed
            content = (
                "I wasn't able to complete your request due to execution failures. "
                "Please try again or rephrase your request."
            )
            response_type = ResponseType.ERROR
        elif failed_count > 0:
            # Mixed results
            content = (
                f"I completed {successful_count} operation(s) successfully, "
                f"but {failed_count} operation(s) failed. "
                "Here are the results:\n\n"
            )
            content += self._format_execution_results(results)
            response_type = ResponseType.TEXT
        else:
            # All successful
            content = "I've completed your request successfully. "
            
            if len(results) == 1:
                # Single tool execution
                result = results[0]
                content += self._format_single_tool_result(result)
            else:
                # Multiple tool executions
                content += f"I executed {len(results)} operations:\n\n"
                content += self._format_execution_results(results)
            
            response_type = ResponseType.TEXT
        
        return GovernorResponse(
            user_id=state.user_id,
            session_id=state.session_id,
            content=content,
            response_type=response_type,
            channel=self._get_response_channel(state),
            metadata={
                "execution_summary": {
                    "total": len(results),
                    "successful": successful_count,
                    "failed": failed_count
                }
            }
        )
    
    def _generate_analysis_response(self, state: GovernorState) -> GovernorResponse:
        """Generate response based on analysis without tool execution.
        
        Args:
            state: Current workflow state
            
        Returns:
            Analysis-based response
        """
        analysis = state.context.get("analysis", {})
        intent = analysis.get("intent", "conversation")
        user_input = state.last_user_input or ""
        
        # Generate contextual response based on intent
        if intent == "simple_question":
            content = self._generate_simple_answer(user_input, analysis)
        elif intent == "conversation":
            content = self._generate_conversational_response(user_input, analysis)
        else:
            content = (
                "I understand you'd like me to help with that. "
                "However, I don't have the tools available to complete that specific request right now. "
                "Is there another way I can assist you?"
            )
        
        return GovernorResponse(
            user_id=state.user_id,
            session_id=state.session_id,
            content=content,
            response_type=ResponseType.TEXT,
            channel=self._get_response_channel(state)
        )
    
    def _generate_default_response(self, state: GovernorState) -> GovernorResponse:
        """Generate default response when no specific context is available.
        
        Args:
            state: Current workflow state
            
        Returns:
            Default response
        """
        user_input = state.last_user_input
        
        if not user_input:
            content = "Hello! How can I help you today?"
        else:
            content = (
                "I received your message, but I'm not sure how to help with that specific request. "
                "Could you please provide more details or rephrase your question?"
            )
        
        return GovernorResponse(
            user_id=state.user_id,
            session_id=state.session_id,
            content=content,
            response_type=ResponseType.TEXT,
            channel=self._get_response_channel(state)
        )
    
    def _format_execution_results(self, results: list[dict[str, Any]]) -> str:
        """Format execution results for user display.
        
        Args:
            results: List of execution result dictionaries
            
        Returns:
            Formatted results string
        """
        formatted = []
        
        for i, result in enumerate(results, 1):
            tool_name = result.get("tool_name", "unknown")
            status = result.get("status", "unknown")
            
            if status == "success":
                formatted.append(f"{i}. ✅ {tool_name}: Completed successfully")
                
                # Add tool-specific result details
                tool_result = result.get("result", {})
                details = self._extract_result_details(tool_name, tool_result)
                if details:
                    formatted.append(f"   {details}")
                    
            elif status == "failed":
                error = result.get("error", "Unknown error")
                formatted.append(f"{i}. ❌ {tool_name}: Failed - {error}")
            elif status == "timeout":
                timeout = result.get("timeout_seconds", "unknown")
                formatted.append(f"{i}. ⏱️ {tool_name}: Timed out after {timeout}s")
            else:
                formatted.append(f"{i}. ❓ {tool_name}: {status}")
        
        return "\n".join(formatted)
    
    def _format_single_tool_result(self, result: dict[str, Any]) -> str:
        """Format a single tool result for user display.
        
        Args:
            result: Tool execution result dictionary
            
        Returns:
            Formatted result string
        """
        tool_name = result.get("tool_name", "unknown")
        tool_result = result.get("result", {})
        
        return self._extract_result_details(tool_name, tool_result) or "Operation completed."
    
    def _extract_result_details(self, tool_name: str, tool_result: dict[str, Any]) -> str:
        """Extract meaningful details from tool results.
        
        Args:
            tool_name: Name of the executed tool
            tool_result: Tool execution result data
            
        Returns:
            User-friendly result details
        """
        if tool_name == "weather":
            location = tool_result.get("location", "Unknown location")
            temp = tool_result.get("temperature", "Unknown")
            condition = tool_result.get("condition", "Unknown")
            return f"Weather in {location}: {temp}, {condition}"
        
        elif tool_name == "email":
            recipients = tool_result.get("recipients", [])
            if recipients:
                return f"Email sent successfully to {', '.join(recipients)}"
            return "Email sent successfully"
        
        elif tool_name == "calendar":
            title = tool_result.get("title", "Event")
            date = tool_result.get("date", "")
            time = tool_result.get("time", "")
            datetime_str = f" on {date}" if date else ""
            datetime_str += f" at {time}" if time else ""
            return f"Calendar event '{title}' created{datetime_str}"
        
        elif tool_name == "search":
            results_count = tool_result.get("results_count", 0)
            return f"Found {results_count} search results"
        
        elif tool_name == "calculator":
            expression = tool_result.get("expression", "")
            calc_result = tool_result.get("result", "")
            return f"{expression} = {calc_result}"
        
        return ""
    
    def _generate_simple_answer(self, user_input: str, analysis: dict[str, Any]) -> str:
        """Generate a simple answer for basic questions.
        
        Args:
            user_input: User's question
            analysis: Analysis results
            
        Returns:
            Simple answer response
        """
        # This would use LLM capabilities in production
        # For now, provide a helpful response
        
        input_lower = user_input.lower()
        
        if "how are you" in input_lower:
            return "I'm functioning well and ready to help! How can I assist you today?"
        elif "what time" in input_lower:
            return f"The current time is {datetime.utcnow().strftime('%H:%M UTC')}."
        elif "what date" in input_lower or "what day" in input_lower:
            return f"Today is {datetime.utcnow().strftime('%A, %B %d, %Y')}."
        elif "help" in input_lower:
            return (
                "I'm your personal AI assistant! I can help with weather, email, "
                "calendar events, web searches, calculations, and more. "
                "Just tell me what you need!"
            )
        else:
            return (
                "That's an interesting question! I'd need additional tools or information "
                "to provide a comprehensive answer. Is there something specific I can help you with?"
            )
    
    def _generate_conversational_response(self, user_input: str, analysis: dict[str, Any]) -> str:
        """Generate a conversational response.
        
        Args:
            user_input: User's message
            analysis: Analysis results
            
        Returns:
            Conversational response
        """
        sentiment = analysis.get("sentiment", "neutral")
        
        if sentiment == "positive":
            return (
                "Thank you for the positive message! I'm glad to be of help. "
                "Is there anything specific you'd like me to assist you with today?"
            )
        elif sentiment == "negative":
            return (
                "I understand you might be experiencing some frustration. "
                "I'm here to help make things better. What can I do for you?"
            )
        else:
            return (
                "I appreciate you reaching out! "
                "How can I assist you today? I can help with various tasks like "
                "checking weather, managing calendar events, sending emails, and more."
            )
    
    def _get_response_channel(self, state: GovernorState) -> str:
        """Determine the appropriate response channel.
        
        Args:
            state: Current workflow state
            
        Returns:
            Channel identifier for response
        """
        # In production, this would be determined from the original request
        # For now, default to a generic channel
        return "api"


class RespondConditional:
    """Conditional routing from respond state."""
    
    @property
    def name(self) -> str:
        """Get the conditional name."""
        return "respond_routing"
    
    def __call__(self, state: GovernorState) -> str:
        """Determine next node from respond state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node name
        """
        from langgraph.graph import END
        
        # After responding, always return to idle state
        # unless we're still awaiting confirmation
        if state.awaiting_confirmation:
            return StateNode.AWAIT_CONFIRMATION.value
        
        # For now, end the workflow after responding
        # In a production system, this would return to idle for continuous operation
        return END