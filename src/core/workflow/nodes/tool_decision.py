"""Tool decision node implementation.

The tool decision node determines which tools to execute based on
analysis results and creates ToolCall objects for execution.
"""

import uuid
from typing import Any

from ...domain.state import GovernorState, StateNode
from ...domain.tools import RiskLevel, ToolCall, ToolStatus
from ..base import NodeFunction


class ToolDecisionNode(NodeFunction):
    """Node that decides which tools to execute.
    
    Based on the analysis results, this node selects appropriate tools,
    configures their parameters, and prepares them for execution.
    """
    
    @property
    def name(self) -> str:
        """Get the node name."""
        return StateNode.TOOL_DECISION.value
    
    async def __call__(self, state: GovernorState) -> GovernorState:
        """Make tool execution decisions.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with tool decisions
        """
        analysis = state.context.get("analysis", {})
        if not analysis:
            state.record_error("No analysis results available for tool decision")
            state.transition_to(StateNode.RESPOND)
            return state
        
        # Get suggested tools from analysis
        suggested_tools = analysis.get("suggested_tools", [])
        entities = analysis.get("entities", {})
        intent = analysis.get("intent", "")
        
        # Make tool selection decisions
        selected_tools = await self._select_tools(
            suggested_tools, 
            entities, 
            intent, 
            state.last_user_input or "",
            state
        )
        
        # Create ToolCall objects for selected tools
        tool_calls = []
        for tool_info in selected_tools:
            tool_call = self._create_tool_call(tool_info, state)
            tool_calls.append(tool_call)
            state.add_tool_call(tool_call)
        
        # Store tool decision results
        state.context.update({
            "tool_decisions": {
                "selected_tools": [tool["name"] for tool in selected_tools],
                "tool_count": len(selected_tools),
                "decision_reasoning": self._generate_decision_reasoning(selected_tools),
                "requires_policy_check": any(
                    tool["risk_level"] != RiskLevel.SAFE for tool in selected_tools
                )
            },
            "tool_decision_completed_at": state.updated_at.isoformat()
        })
        
        # Determine next state
        if selected_tools:
            # Check if any tools require policy evaluation
            needs_policy_check = any(
                tool["risk_level"] in [RiskLevel.SENSITIVE, RiskLevel.DANGEROUS] 
                for tool in selected_tools
            )
            
            if needs_policy_check:
                state.transition_to(StateNode.POLICY_CHECK)
            else:
                state.transition_to(StateNode.EXECUTE)
        else:
            # No tools selected, go to respond
            state.transition_to(StateNode.RESPOND)
        
        return state
    
    async def _select_tools(
        self, 
        suggested_tools: list[str],
        entities: dict[str, list[str]],
        intent: str,
        user_input: str,
        state: GovernorState
    ) -> list[dict[str, Any]]:
        """Select tools to execute based on analysis.
        
        Args:
            suggested_tools: Tools suggested by analysis
            entities: Extracted entities from user input
            intent: Classified user intent
            user_input: Original user input
            state: Current workflow state
            
        Returns:
            List of selected tool configurations
        """
        selected = []
        
        # Tool selection logic based on intent and entities
        for tool_name in suggested_tools:
            tool_config = await self._configure_tool(
                tool_name, entities, intent, user_input, state
            )
            if tool_config:
                selected.append(tool_config)
        
        # Add additional tools based on specific patterns
        additional_tools = self._identify_additional_tools(user_input, entities, intent)
        selected.extend(additional_tools)
        
        # Remove duplicates and validate
        seen_tools = set()
        validated_tools = []
        
        for tool in selected:
            tool_name = tool["name"]
            if tool_name not in seen_tools and self._validate_tool_config(tool):
                seen_tools.add(tool_name)
                validated_tools.append(tool)
        
        return validated_tools
    
    async def _configure_tool(
        self,
        tool_name: str,
        entities: dict[str, list[str]],
        intent: str,
        user_input: str,
        state: GovernorState
    ) -> dict[str, Any] | None:
        """Configure a specific tool for execution.
        
        Args:
            tool_name: Name of the tool to configure
            entities: Extracted entities
            intent: User intent
            user_input: Original user input
            state: Current workflow state
            
        Returns:
            Tool configuration or None if not applicable
        """
        # Tool-specific configuration logic
        if tool_name == "weather":
            return self._configure_weather_tool(entities, user_input)
        elif tool_name == "email":
            return self._configure_email_tool(entities, user_input)
        elif tool_name == "calendar":
            return self._configure_calendar_tool(entities, user_input)
        elif tool_name == "search":
            return self._configure_search_tool(entities, user_input)
        elif tool_name == "file_manager":
            return self._configure_file_manager_tool(entities, user_input)
        elif tool_name == "calculator":
            return self._configure_calculator_tool(entities, user_input)
        else:
            # Generic tool configuration
            return {
                "name": tool_name,
                "arguments": {"query": user_input},
                "risk_level": RiskLevel.SAFE,
                "timeout_seconds": 30.0
            }
    
    def _configure_weather_tool(self, entities: dict[str, list[str]], user_input: str) -> dict[str, Any]:
        """Configure weather tool."""
        # Extract location from input or use default
        location = "current location"  # Default
        
        # Look for location patterns in user input
        import re
        location_patterns = [
            r'\bin\s+([A-Za-z\s,]+?)(?:\s|$)',
            r'\bfor\s+([A-Za-z\s,]+?)(?:\s|$)',
            r'\bat\s+([A-Za-z\s,]+?)(?:\s|$)'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                break
        
        return {
            "name": "weather",
            "arguments": {"location": location},
            "risk_level": RiskLevel.SAFE,
            "timeout_seconds": 10.0
        }
    
    def _configure_email_tool(self, entities: dict[str, list[str]], user_input: str) -> dict[str, Any]:
        """Configure email tool."""
        emails = entities.get("emails", [])
        
        arguments = {"message": user_input}
        if emails:
            arguments["recipients"] = emails
        
        return {
            "name": "email",
            "arguments": arguments,
            "risk_level": RiskLevel.DANGEROUS,  # Email sending is high risk
            "timeout_seconds": 30.0
        }
    
    def _configure_calendar_tool(self, entities: dict[str, list[str]], user_input: str) -> dict[str, Any]:
        """Configure calendar tool."""
        dates = entities.get("dates", [])
        times = entities.get("times", [])
        
        arguments = {"description": user_input}
        if dates:
            arguments["date"] = dates[0]
        if times:
            arguments["time"] = times[0]
        
        return {
            "name": "calendar",
            "arguments": arguments,
            "risk_level": RiskLevel.SENSITIVE,  # Calendar access is sensitive
            "timeout_seconds": 15.0
        }
    
    def _configure_search_tool(self, entities: dict[str, list[str]], user_input: str) -> dict[str, Any]:
        """Configure search tool."""
        return {
            "name": "search",
            "arguments": {"query": user_input},
            "risk_level": RiskLevel.SAFE,
            "timeout_seconds": 20.0
        }
    
    def _configure_file_manager_tool(self, entities: dict[str, list[str]], user_input: str) -> dict[str, Any]:
        """Configure file manager tool."""
        return {
            "name": "file_manager",
            "arguments": {"operation": user_input},
            "risk_level": RiskLevel.SENSITIVE,  # File operations are sensitive
            "timeout_seconds": 30.0
        }
    
    def _configure_calculator_tool(self, entities: dict[str, list[str]], user_input: str) -> dict[str, Any]:
        """Configure calculator tool."""
        return {
            "name": "calculator",
            "arguments": {"expression": user_input},
            "risk_level": RiskLevel.SAFE,
            "timeout_seconds": 5.0
        }
    
    def _identify_additional_tools(
        self, 
        user_input: str, 
        entities: dict[str, list[str]], 
        intent: str
    ) -> list[dict[str, Any]]:
        """Identify additional tools that might be needed."""
        additional = []
        
        # Context awareness tool for complex requests
        if intent in ["action_request", "information_request"] and len(user_input) > 100:
            additional.append({
                "name": "context_analyzer",
                "arguments": {"text": user_input},
                "risk_level": RiskLevel.SAFE,
                "timeout_seconds": 10.0
            })
        
        # Memory tool for follow-up questions
        if intent == "conversation" and any(
            word in user_input.lower() 
            for word in ["remember", "recall", "earlier", "before", "previous"]
        ):
            additional.append({
                "name": "memory_search",
                "arguments": {"query": user_input},
                "risk_level": RiskLevel.SAFE,
                "timeout_seconds": 15.0
            })
        
        return additional
    
    def _validate_tool_config(self, tool_config: dict[str, Any]) -> bool:
        """Validate that a tool configuration is complete and valid."""
        required_fields = ["name", "arguments", "risk_level", "timeout_seconds"]
        
        for field in required_fields:
            if field not in tool_config:
                return False
        
        # Validate risk level
        if not isinstance(tool_config["risk_level"], RiskLevel):
            return False
        
        # Validate timeout
        timeout = tool_config["timeout_seconds"]
        if not isinstance(timeout, (int, float)) or timeout <= 0 or timeout > 300:
            return False
        
        return True
    
    def _create_tool_call(self, tool_config: dict[str, Any], state: GovernorState) -> ToolCall:
        """Create a ToolCall object from tool configuration."""
        return ToolCall(
            tool_name=tool_config["name"],
            arguments=tool_config["arguments"],
            risk_level=tool_config["risk_level"],
            execution_id=f"exec_{uuid.uuid4().hex[:12]}",
            user_id=state.user_id,
            session_id=state.session_id,
            timeout_seconds=tool_config["timeout_seconds"],
            status=ToolStatus.PENDING
        )
    
    def _generate_decision_reasoning(self, selected_tools: list[dict[str, Any]]) -> str:
        """Generate reasoning for tool selection decisions."""
        if not selected_tools:
            return "No tools required for this request"
        
        tool_names = [tool["name"] for tool in selected_tools]
        risk_levels = [tool["risk_level"] for tool in selected_tools]
        
        reasoning = f"Selected {len(selected_tools)} tool(s): {', '.join(tool_names)}. "
        
        safe_count = sum(1 for risk in risk_levels if risk == RiskLevel.SAFE)
        sensitive_count = sum(1 for risk in risk_levels if risk == RiskLevel.SENSITIVE)
        dangerous_count = sum(1 for risk in risk_levels if risk == RiskLevel.DANGEROUS)
        
        if dangerous_count > 0:
            reasoning += f"{dangerous_count} high-risk tool(s) require user confirmation. "
        if sensitive_count > 0:
            reasoning += f"{sensitive_count} sensitive tool(s) require policy check. "
        if safe_count > 0:
            reasoning += f"{safe_count} safe tool(s) can execute immediately."
        
        return reasoning


class ToolDecisionConditional:
    """Conditional routing from tool decision state."""
    
    @property
    def name(self) -> str:
        """Get the conditional name."""
        return "tool_decision_routing"
    
    def __call__(self, state: GovernorState) -> str:
        """Determine next node from tool decision state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node name
        """
        tool_decisions = state.context.get("tool_decisions", {})
        
        # If there was an error in tool decision, go to respond
        if state.error_count > 0:
            return StateNode.RESPOND.value
        
        # If no tools were selected, go to respond
        if tool_decisions.get("tool_count", 0) == 0:
            return StateNode.RESPOND.value
        
        # If tools require policy check, go to policy check
        if tool_decisions.get("requires_policy_check", False):
            return StateNode.POLICY_CHECK.value
        
        # If tools are all safe, go directly to execution
        return StateNode.EXECUTE.value