"""Context Assembler - Dynamic prompt assembly system.

This module provides the core context assembly functionality for the Governor system,
dynamically constructing prompts from persona, environment, memory, and task data.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ...core.domain.memory import ConversationTurn
from ...core.domain.state import GovernorState
from ...memory.tiers.procedural import ProceduralMemoryStore
from ...memory.tiers.short_term import ShortTermMemoryClient


@dataclass
class ContextData:
    """Structured context data for prompt assembly."""

    persona: str
    environment: dict[str, Any]
    memory: dict[str, Any]
    tasks: list[dict[str, Any]]
    metadata: dict[str, Any]


class ContextAssembler:
    """Dynamic system prompt construction for the Governor.

    Assembles contextual prompts using the formula:
    Prompt = Persona + Environment + Memory + Tasks
    """

    def __init__(self) -> None:
        """Initialize the context assembler."""
        self.persona_cache: dict[str, str] = {}
        self.environment_cache: dict[str, dict[str, Any]] = {}
        self.procedural_memory = ProceduralMemoryStore()

    async def assemble_context(
        self,
        user_id: str,
        current_input: str,
        state: GovernorState,
        available_tools: list[str] | None = None,
    ) -> str:
        """Assemble complete system prompt from multiple context sources.

        Args:
            user_id: Unique identifier for the user
            current_input: Current user message/input
            state: Current Governor state
            available_tools: List of available tool names

        Returns:
            Assembled system prompt string
        """
        context_data = await self._gather_context_data(user_id, current_input, state)

        return self._build_prompt(
            context_data=context_data,
            current_input=current_input,
            state=state,
            available_tools=available_tools or [],
        )

    async def _gather_context_data(
        self, user_id: str, current_input: str, state: GovernorState
    ) -> ContextData:
        """Gather all context data from various providers.

        Args:
            user_id: User identifier
            current_input: Current user input
            state: Current state

        Returns:
            Structured context data
        """
        # Gather context data in parallel for performance
        try:
            async with self.procedural_memory as proc_store:
                # Parallel retrieval of context components including Redis
                persona_task = asyncio.create_task(
                    self._get_persona(user_id, proc_store)
                )
                environment_task = asyncio.create_task(
                    self._get_environment_context(user_id, proc_store)
                )
                # Fetch short-term memory from Redis directly
                memory_task = asyncio.create_task(
                    self._get_memory_context(user_id)
                )
                tasks_task = asyncio.create_task(self._get_active_tasks(user_id, state))
                instructions_task = asyncio.create_task(
                    self._get_relevant_instructions(user_id, current_input, proc_store)
                )

                # Wait for all context components
                (
                    persona,
                    environment,
                    memory,
                    tasks,
                    instructions,
                ) = await asyncio.gather(
                    persona_task,
                    environment_task,
                    memory_task,
                    tasks_task,
                    instructions_task,
                    return_exceptions=True,
                )

                # Handle any exceptions gracefully
                if isinstance(persona, Exception):
                    persona = self._get_default_persona()
                if isinstance(environment, Exception):
                    environment = self._get_default_environment()
                if isinstance(memory, Exception):
                    # Return empty memory structure with error flag
                    memory = {
                        "recent_conversation": [],
                        "conversation_length": 0,
                        "session_age_minutes": 0.0,
                        "conversation_topics": ["general"],
                        "session_summary": "Memory unavailable",
                        "error": f"Memory fetch failed: {str(memory)}",
                    }
                if isinstance(tasks, Exception):
                    tasks = []
                if isinstance(instructions, Exception):
                    instructions = []

                # Add instructions to memory context
                if instructions and isinstance(memory, dict):
                    memory["relevant_instructions"] = [
                        {
                            "title": instr.instruction.title,
                            "instruction": instr.instruction.instruction,
                            "relevance": instr.relevance_score,
                            "category": instr.instruction.category.value,
                        }
                        for instr in instructions[:5]  # Top 5 most relevant
                    ]

                return ContextData(
                    persona=persona,
                    environment=environment,
                    memory=memory,
                    tasks=tasks,
                    metadata={
                        "user_id": user_id,
                        "session_id": state.session_id,
                        "current_node": state.current_node.value,
                        "timestamp": datetime.utcnow().isoformat(),
                        "input_length": len(current_input),
                        "conversation_turns": state.total_turns,
                        "tools_executed": state.total_tools_executed,
                        "instructions_retrieved": len(instructions),
                    },
                )
        except Exception as e:
            # Fallback to minimal context if everything fails
            try:
                memory = await self._get_memory_context(user_id)
            except Exception:
                memory = {
                    "recent_conversation": [],
                    "conversation_length": 0,
                    "session_age_minutes": 0.0,
                    "conversation_topics": ["general"],
                    "session_summary": "Memory unavailable",
                    "error": "Memory fetch failed",
                }
            
            try:
                tasks = await self._get_active_tasks(user_id, state)
            except Exception:
                tasks = []

            return ContextData(
                persona=self._get_default_persona(),
                environment=self._get_default_environment(),
                memory=memory,
                tasks=tasks,
                metadata={
                    "user_id": user_id,
                    "session_id": state.session_id,
                    "current_node": state.current_node.value,
                    "timestamp": datetime.utcnow().isoformat(),
                    "input_length": len(current_input),
                    "conversation_turns": state.total_turns,
                    "tools_executed": state.total_tools_executed,
                    "procedural_memory_error": str(e),
                },
            )

    async def _get_memory_context(self, user_id: str) -> dict[str, Any]:
        """Fetch short-term memory context directly from Redis.

        Args:
            user_id: User identifier

        Returns:
            Short-term memory context dictionary
        """
        async with ShortTermMemoryClient() as stm:
            context = await stm.get_context(user_id)
            recent_messages = context.recent_messages
            
            return self._format_short_term_memory(recent_messages)

    def _format_short_term_memory(
        self, recent_messages: list[ConversationTurn]
    ) -> dict[str, Any]:
        """Format conversation turns into dictionary structure expected by _build_prompt.

        Args:
            recent_messages: List of ConversationTurn objects from Redis

        Returns:
            Formatted short-term memory context dictionary
        """
        # Convert ConversationTurn objects to the expected format
        formatted_messages = []
        for turn in recent_messages:
            formatted_messages.extend([
                {
                    "role": "user",
                    "content": turn.user_message,
                    "timestamp": turn.timestamp.isoformat(),
                    "metadata": turn.metadata
                },
                {
                    "role": "assistant", 
                    "content": turn.assistant_response,
                    "timestamp": turn.timestamp.isoformat(),
                    "metadata": turn.metadata
                }
            ])

        # Calculate session age from last message if available
        session_age_minutes = 0.0
        if recent_messages:
            last_turn = recent_messages[-1]
            time_diff = datetime.utcnow() - last_turn.timestamp
            session_age_minutes = time_diff.total_seconds() / 60

        return {
            "recent_conversation": formatted_messages,
            "conversation_length": len(recent_messages) * 2,  # user + assistant
            "session_age_minutes": session_age_minutes,
            "conversation_topics": self._extract_topics(recent_messages),
            "session_summary": f"Active session with {len(recent_messages)} turns",
        }

    async def _get_persona(
        self, user_id: str, proc_store: ProceduralMemoryStore
    ) -> str:
        """Get user-specific persona configuration from profile data.

        Args:
            user_id: User identifier
            proc_store: Procedural memory store

        Returns:
            Personalized persona prompt string
        """
        try:
            # Get user profile
            profile = await proc_store.get_user_profile(user_id)

            # Build personalized persona
            base_persona = self._get_default_persona()

            # Customize based on user preferences
            customizations = []

            # Communication style
            comm_style = profile.get_value(
                "communication", "style", "professional"
            )
            if comm_style == "casual":
                customizations.append("- Use casual, friendly language and tone")
            elif comm_style == "formal":
                customizations.append(
                    "- Use formal, professional language at all times"
                )

            # Response format preferences  
            format_pref = profile.get_value(
                "preferences", "response_format", "balanced"
            )
            if format_pref == "brief":
                customizations.append("- Keep responses concise and to the point")
            elif format_pref == "detailed":
                customizations.append(
                    "- Provide detailed explanations and context"
                )

            # User name personalization
            name = profile.get_value("personal", "name")
            if name:
                customizations.append(f"- Address the user by their name: {name}")

            # Add customizations to persona
            if customizations:
                base_persona += "\n\nUSER-SPECIFIC PREFERENCES:\n" + "\n".join(
                    customizations
                )

            return base_persona

        except Exception:
            # Fall back to default persona
            return self._get_default_persona()

    def _get_default_persona(self) -> str:
        """Get default persona configuration."""
        return """You are the Headless Governor, a personal AI assistant that operates as an invisible OS layer.

CORE PRINCIPLES:
- You execute tasks autonomously when safe, ask for confirmation when risky
- You maintain context across conversations and remember user preferences
- You prioritize security and never execute dangerous operations without explicit approval
- You are helpful, efficient, and proactive in task completion
- You explain your reasoning when making decisions

INTERACTION STYLE:
- Be concise but thorough in explanations
- Always acknowledge tool execution results
- Ask for clarification when user intent is ambiguous
- Provide status updates for long-running operations"""

    async def _get_environment_context(
        self, user_id: str, proc_store: ProceduralMemoryStore
    ) -> dict[str, Any]:
        """Get current environment context with user timezone preferences.

        Args:
            user_id: User identifier
            proc_store: Procedural memory store

        Returns:
            Environment context dictionary
        """
        try:
            # Use cache for performance (environment context changes slowly)
            cache_key = f"env_{user_id}_{datetime.utcnow().hour}"
            if cache_key in self.environment_cache:
                return self.environment_cache[cache_key]

            now = datetime.utcnow()

            # Get user timezone from profile
            profile = await proc_store.get_user_profile(user_id)
            user_tz = profile.get_value("timezone", "primary", "UTC")

            environment = {
                "current_time": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "timestamp": now.isoformat(),
                "timezone": user_tz,
                "day_of_week": now.strftime("%A"),
                "time_period": self._get_time_period(now.hour),
                "system_load": "normal",  # Could integrate with actual system metrics
            }

            # Add location if available
            location = profile.get_value("location", "current")
            if location:
                environment["location"] = location

            # Cache for 1 hour
            self.environment_cache[cache_key] = environment
            return environment

        except Exception:
            return self._get_default_environment()

    def _get_default_environment(self) -> dict[str, Any]:
        """Get default environment context."""
        now = datetime.utcnow()
        return {
            "current_time": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "timestamp": now.isoformat(),
            "timezone": "UTC",
            "day_of_week": now.strftime("%A"),
            "time_period": self._get_time_period(now.hour),
            "system_load": "normal",
        }

    async def _get_active_tasks(
        self, user_id: str, state: GovernorState
    ) -> list[dict[str, Any]]:
        """Get active tasks and pending operations.

        Args:
            user_id: User identifier
            state: Current state

        Returns:
            List of active tasks
        """
        tasks = []

        # Add pending tool calls from state
        if state.pending_tools:
            for tool in state.pending_tools:
                tasks.append(
                    {
                        "type": "pending_tool_execution",
                        "tool_name": tool.tool_name,
                        "status": "awaiting_confirmation"
                        if state.awaiting_confirmation
                        else "pending",
                        "risk_level": tool.risk_level.value,
                        "execution_id": tool.execution_id,
                    }
                )

        # Add any background operations (future feature)
        # This could include scheduled tasks, file uploads, etc.

        return tasks

    async def _get_relevant_instructions(
        self, user_id: str, current_input: str, proc_store: ProceduralMemoryStore
    ) -> list[Any]:
        """Get relevant behavioral instructions for the current input.

        Args:
            user_id: User identifier
            current_input: Current user input to find relevant instructions for
            proc_store: Procedural memory store

        Returns:
            List of relevant instructions with similarity scores
        """
        try:
            relevant_instructions = await proc_store.get_relevant_instructions(
                user_id=user_id,
                query_text=current_input,
                min_confidence=0.3,
                min_priority=1,
                limit=5,
            )

            return relevant_instructions

        except Exception:
            # Return empty list if instruction retrieval fails
            return []

    def _build_prompt(
        self,
        context_data: ContextData,
        current_input: str,
        state: GovernorState,
        available_tools: list[str],
    ) -> str:
        """Build the final system prompt from context data.

        Args:
            context_data: Gathered context data
            current_input: Current user input
            state: Current state
            available_tools: Available tools list

        Returns:
            Complete system prompt
        """
        prompt_parts = [
            context_data.persona,
            "",
            "CURRENT ENVIRONMENT:",
            f"- Time: {context_data.environment['current_time']}",
            (
                f"- Day: {context_data.environment['day_of_week']} "
                f"({context_data.environment['time_period']})"
            ),
            f"- System Status: {context_data.environment['system_load']}",
            "",
        ]

        # Add relevant behavioral instructions (procedural memory)
        if context_data.memory.get("relevant_instructions"):
            prompt_parts.extend(
                [
                    "RELEVANT BEHAVIORAL INSTRUCTIONS:",
                ]
            )
            for instr in context_data.memory["relevant_instructions"]:
                prompt_parts.append(
                    f"- {instr['title']}: {instr['instruction']}"
                )
                prompt_parts.append(
                    f"  (Category: {instr['category']}, "
                    f"Relevance: {instr['relevance']:.2f})"
                )
            prompt_parts.append("")

        # Add short-term memory context (working memory)
        if context_data.memory.get("recent_conversation"):
            prompt_parts.extend(
                [
                    "RECENT CONVERSATION (Working Memory):",
                    (
                        f"- Session length: "
                        f"{context_data.memory['session_age_minutes']} minutes"
                    ),
                    (
                        f"- Total exchanges: "
                        f"{context_data.memory['conversation_length']}"
                    ),
                    (
                        f"- Topics: "
                        f"{', '.join(context_data.memory['conversation_topics'])}"
                    ),
                    "",
                ]
            )
            
            # Inject actual conversation transcript
            recent_messages = context_data.memory["recent_conversation"]
            if recent_messages:
                prompt_parts.append("CONVERSATION TRANSCRIPT:")
                for message in recent_messages:
                    if message.get("role") == "user":
                        prompt_parts.append(f"User: {message['content']}")
                    elif message.get("role") == "assistant":
                        prompt_parts.append(f"Assistant: {message['content']}")
                prompt_parts.append("")
            
            prompt_parts.extend(
                [
                    "MEMORY TOOLS AVAILABLE:",
                    (
                        "- You have access to 'search_episodic_memory' and "
                        "'query_knowledge_graph' tools"
                    ),
                    (
                        "- If the user refers to past events, people, or facts not in "
                        "the active conversation, YOU MUST USE THESE TOOLS to retrieve "
                        "the information"
                    ),
                    (
                        "- Do not attempt to recall information from outside the "
                        "recent conversation above without using these tools"
                    ),
                    "",
                ]
            )

        # Add active tasks
        if context_data.tasks:
            prompt_parts.extend(
                [
                    "ACTIVE TASKS:",
                ]
            )
            for task in context_data.tasks:
                prompt_parts.append(
                    f"- {task['type']}: {task.get('tool_name', 'N/A')} "
                    f"({task['status']})"
                )
            prompt_parts.append("")

        # Add available tools
        if available_tools:
            prompt_parts.extend(
                ["AVAILABLE TOOLS:", f"- {', '.join(available_tools)}", ""]
            )

        # Add current state and input
        prompt_parts.extend(
            [
                "CURRENT SITUATION:",
                f"- User Input: {current_input}",
                f"- Current State: {state.current_node.value}",
                f"- Session ID: {state.session_id}",
                (
                    f"- Awaiting Confirmation: "
                    f"{'Yes' if state.awaiting_confirmation else 'No'}"
                ),
                "",
            ]
        )

        return "\n".join(prompt_parts)

    def _get_time_period(self, hour: int) -> str:
        """Get descriptive time period for given hour.

        Args:
            hour: Hour of day (0-23)

        Returns:
            Time period description
        """
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    def _extract_topics(self, turns: list[ConversationTurn]) -> list[str]:
        """Extract conversation topics from ConversationTurn objects.

        Args:
            turns: List of ConversationTurn objects

        Returns:
            List of extracted topics
        """
        # Simple topic extraction (could be enhanced with NLP)
        topics = set()

        for turn in turns[-3:]:  # Last 3 conversation turns
            # Extract from both user and assistant messages
            user_content = turn.user_message.lower()
            assistant_content = turn.assistant_response.lower()
            combined_content = f"{user_content} {assistant_content}"

            # Basic keyword extraction
            if "weather" in combined_content:
                topics.add("weather")
            if "email" in combined_content:
                topics.add("email")
            if "search" in combined_content:
                topics.add("search")
            if "file" in combined_content:
                topics.add("files")
            if "calculate" in combined_content or "math" in combined_content:
                topics.add("calculations")
            if "memory" in combined_content or "remember" in combined_content:
                topics.add("memory")
            if "schedule" in combined_content or "appointment" in combined_content:
                topics.add("scheduling")

        return list(topics) if topics else ["general"]

    def clear_cache(self) -> None:
        """Clear all cached context data."""
        self.persona_cache.clear()
        self.environment_cache.clear()