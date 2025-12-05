"""Enhanced Context Assembler with Memory Manager integration.

This module provides memory-aware context assembly using the RAG++ 4-tier memory hierarchy.
It replaces the basic context assembly with sophisticated parallel memory retrieval from:
- Tier 1: Short-term Memory (Redis) - Working conversation context
- Tier 2: Episodic Memory (PostgreSQL+pgvector) - Searchable interaction history  
- Tier 3: Semantic Memory (Neo4j) - Knowledge graph of entities/relationships
- Tier 4: Procedural Memory (PostgreSQL) - User preferences and behavioral rules
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from ...core.domain.state import GovernorState
from ...memory.manager import MemoryManager
from ...memory.base import MemoryQuery
from ...memory.services.entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)


class EnhancedContextAssembler:
    """Memory-aware context assembler with parallel retrieval from all memory tiers.

    This assembler leverages the complete RAG++ memory hierarchy to build rich,
    contextual prompts that include:
    - Recent conversation context (short-term memory)
    - Relevant past interactions (episodic memory)
    - Known entities and relationships (semantic memory)
    - User preferences and behavioral rules (procedural memory)
    """

    def __init__(self, memory_manager: MemoryManager):
        """Initialize enhanced context assembler.

        Args:
            memory_manager: Initialized memory manager for accessing all tiers
        """
        self.memory_manager = memory_manager
        self.entity_extractor = EntityExtractor()

        # Performance configuration
        self.max_retrieval_latency_ms = (
            2000  # Increased to allow for embedding generation and vector search
        )
        self.enable_entity_extraction = True

        logger.info(
            "🧠 Enhanced Context Assembler initialized with RAG++ memory integration"
        )

    async def assemble_context(
        self,
        user_id: str,
        current_input: str,
        state: GovernorState,
        available_tools: List[str] = None,
    ) -> str:
        """Assemble memory-aware context using parallel retrieval from all memory tiers.

        This method:
        1. Extracts entities from current input for targeted retrieval
        2. Queries all 4 memory tiers in parallel
        3. Assembles a sophisticated prompt with weighted memory integration
        4. Falls back gracefully if any memory tier fails

        Args:
            user_id: User identifier for personalized context
            current_input: Current user message/input
            state: Current Governor state
            available_tools: List of available tool names

        Returns:
            Complete memory-aware system prompt
        """
        start_time = datetime.utcnow()
        request_id = f"ctx_{user_id}_{start_time.timestamp():.0f}"

        logger.debug(
            f"🔍 [CTX-{request_id}] Assembling memory-aware context for: {current_input[:50]}..."
        )

        try:
            # Step 1: Extract entities for targeted memory retrieval
            entities = []
            if self.enable_entity_extraction:
                entities = await self._extract_entities_safe(current_input, request_id)

            # Step 2: Parallel memory retrieval from all tiers
            logger.debug(
                f"🧠 [CTX-{request_id}] Starting parallel memory retrieval from 4 tiers..."
            )

            memory_context = await asyncio.wait_for(
                self.memory_manager.recall_context(
                    user_id=user_id,
                    current_input=current_input,
                    max_latency_ms=self.max_retrieval_latency_ms,
                ),
                timeout=self.max_retrieval_latency_ms / 1000.0,
            )

            # Step 3: Build memory-aware prompt
            prompt = await self._build_memory_aware_prompt(
                user_id=user_id,
                current_input=current_input,
                state=state,
                memory_context=memory_context,
                entities=entities,
                available_tools=available_tools or [],
            )

            # Performance metrics
            assembly_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(
                f"✅ [CTX-{request_id}] Context assembled successfully ({assembly_time:.2f}ms)"
            )

            return prompt

        except asyncio.TimeoutError:
            logger.warning(
                f"⏰ [CTX-{request_id}] Context assembly timeout, using fallback"
            )
            return await self._build_fallback_prompt(
                user_id, current_input, state, available_tools
            )

        except Exception as e:
            logger.error(f"❌ [CTX-{request_id}] Context assembly failed: {e}")
            return await self._build_fallback_prompt(
                user_id, current_input, state, available_tools
            )

    async def _extract_entities_safe(self, text: str, request_id: str) -> List[str]:
        """Safely extract entities from input text."""
        try:
            async with self.entity_extractor as extractor:
                extraction_result = await extractor.extract_from_conversation(
                    user_message=text,
                    assistant_response="",
                    user_id="context_extraction",
                )

                entities = [entity.name for entity in extraction_result.entities]
                logger.debug(
                    f"📝 [CTX-{request_id}] Extracted {len(entities)} entities: {entities[:5]}"
                )
                return entities

        except Exception as e:
            logger.warning(
                f"⚠️  [CTX-{request_id}] Entity extraction failed, using keyword fallback: {e}"
            )
            # Fallback to simple keyword extraction
            return self._extract_keywords_simple(text)

    def _extract_keywords_simple(self, text: str) -> List[str]:
        """Simple keyword extraction fallback."""
        words = text.split()
        keywords = []

        for word in words:
            # Extract capitalized words and important terms
            clean_word = word.strip(".,!?").lower()
            if len(clean_word) > 2 and (
                word.istitle()
                or clean_word in ["email", "file", "search", "weather", "calendar"]
            ):
                keywords.append(clean_word)

        return keywords[:10]  # Limit to 10 keywords

    async def _build_memory_aware_prompt(
        self,
        user_id: str,
        current_input: str,
        state: GovernorState,
        memory_context: Dict[str, Any],
        entities: List[str],
        available_tools: List[str],
    ) -> str:
        """Build sophisticated prompt with memory integration."""

        sections = [
            self._build_persona_section(),
            self._build_environment_section(state),
            await self._build_memory_section(memory_context),
            self._build_tools_section(available_tools),
            self._build_situation_section(user_id, current_input, state, entities),
        ]

        # Filter out empty sections and join
        prompt = "\n\n".join(section for section in sections if section.strip())

        return prompt

    def _build_persona_section(self) -> str:
        """Build core persona section."""
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
- Provide status updates for long-running operations

IMPORTANT: The memory context below is for YOUR INTERNAL USE ONLY. Do not repeat or echo memory information back to the user unless they specifically ask about it. Use this knowledge naturally in conversation."""

    def _build_environment_section(self, state: GovernorState) -> str:
        """Build environment context section."""
        now = datetime.utcnow()

        return f"""CURRENT ENVIRONMENT:
- Time: {now.strftime("%Y-%m-%d %H:%M:%S UTC")}
- Day: {now.strftime("%A")} ({self._get_time_period(now.hour)})
- Session: {state.session_id}
- Current State: {state.current_node.value}
- System Status: Operational"""

    async def _build_memory_section(self, memory_context: Dict[str, Any]) -> str:
        """Build memory context section with weighted integration."""
        if memory_context.get("status") == "error":
            return "INTERNAL MEMORY STATUS: Limited (using fallback context)"

        memory_parts = ["=== INTERNAL MEMORY CONTEXT (FOR YOUR USE ONLY) ==="]

        # Short-term memory (working context)
        short_term = memory_context.get("short_term", {})
        if short_term:
            if short_term.get("summary"):
                memory_parts.append(
                    f"RECENT CONVERSATION SUMMARY:\n{short_term['summary']}"
                )

            if short_term.get("recent_messages"):
                recent_messages = short_term["recent_messages"]

                # Include ALL messages from the short-term buffer (it already manages token budget)
                conversation_context = []
                for (
                    msg
                ) in recent_messages:  # Use ALL messages in buffer, not just last 5
                    user_msg = msg.get("user_message", "")
                    assistant_msg = msg.get("assistant_response", "")
                    # Clean system info from assistant response for cleaner context
                    clean_assistant_msg = (
                        assistant_msg.split("---")[0].strip() if assistant_msg else ""
                    )

                    if user_msg:
                        conversation_context.append(f"User: {user_msg}")
                    if clean_assistant_msg:
                        conversation_context.append(f"Assistant: {clean_assistant_msg}")

                if conversation_context:
                    memory_parts.append(
                        f"RECENT CONVERSATION CONTEXT:\n"
                        + "\n".join(conversation_context)
                    )

        # Episodic memory (past interactions)
        episodic = memory_context.get("episodic", [])
        if episodic:
            episode_text = "\n".join(
                [
                    f"- {ep.content[:100]}... (similarity: {ep.metadata.get('similarity', 0):.2f})"
                    for ep in episodic[:3]  # Top 3 most relevant
                    if ep.content
                ]
            )
            if episode_text:
                memory_parts.append(f"RELEVANT PAST INTERACTIONS:\n{episode_text}")

        # Semantic memory (knowledge graph)
        semantic = memory_context.get("semantic", {})
        if semantic and semantic.get("entities"):
            entities = semantic["entities"][:10]  # Limit display
            entity_names = [
                entity.get("name", entity.get("type", "Unknown"))
                for entity in entities
                if entity
            ]
            if entity_names:
                memory_parts.append(f"KNOWN ENTITIES: {', '.join(entity_names)}")

            if semantic.get("relationships"):
                rel_text = "\n".join(
                    [
                        f"- {rel.get('from_entity', '?')} → {rel.get('to_entity', '?')} ({rel.get('relationship_type', 'related')})"
                        for rel in semantic["relationships"][:5]
                        if rel.get("from_entity") and rel.get("to_entity")
                    ]
                )
                if rel_text:
                    memory_parts.append(f"KNOWN RELATIONSHIPS:\n{rel_text}")

        # Procedural memory (preferences and rules)
        procedural = memory_context.get("procedural", [])
        if procedural:
            rules_text = "\n".join(
                [
                    f"- {rule.get('title', 'Preference')}: {rule.get('instruction', '')}"
                    for rule in procedural[:3]  # Top 3 most relevant
                    if rule.get("instruction")
                ]
            )
            if rules_text:
                memory_parts.append(f"USER PREFERENCES & RULES:\n{rules_text}")

        if len(memory_parts) > 1:  # More than just the header
            memory_parts.append("=== END INTERNAL MEMORY CONTEXT ===")
            return "\n\n".join(memory_parts)
        else:
            return "INTERNAL MEMORY STATUS: No relevant context retrieved"

    def _build_tools_section(self, available_tools: List[str]) -> str:
        """Build available tools section."""
        if not available_tools:
            return ""

        return f"""AVAILABLE TOOLS:
- {', '.join(available_tools)}

Use tools when they can help accomplish the user's goals. Always explain what you're doing."""

    def _build_situation_section(
        self,
        user_id: str,
        current_input: str,
        state: GovernorState,
        entities: List[str],
    ) -> str:
        """Build current situation section."""
        situation_parts = [
            "CURRENT SITUATION:",
            f"- Input: {current_input}",
            f"- Conversation Turn: #{state.total_turns}",
        ]

        if state.awaiting_confirmation:
            situation_parts.append(
                "- Status: Awaiting user confirmation for pending action"
            )

        if state.pending_tools:
            tool_names = [tool.tool_name for tool in state.pending_tools]
            situation_parts.append(f"- Pending Tools: {', '.join(tool_names)}")

        if entities:
            situation_parts.append(f"- Detected Entities: {', '.join(entities[:5])}")

        return "\n".join(situation_parts)

    async def _build_fallback_prompt(
        self,
        user_id: str,
        current_input: str,
        state: GovernorState,
        available_tools: List[str],
    ) -> str:
        """Build fallback prompt when memory retrieval fails."""
        logger.warning(f"🔄 Building fallback prompt for user {user_id}")

        fallback_sections = [
            self._build_persona_section(),
            self._build_environment_section(state),
            "MEMORY STATUS: Operating in limited context mode (memory system unavailable)",
            self._build_tools_section(available_tools),
            self._build_situation_section(user_id, current_input, state, []),
        ]

        return "\n\n".join(section for section in fallback_sections if section.strip())

    def _get_time_period(self, hour: int) -> str:
        """Get descriptive time period for given hour."""
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    async def get_assembly_metrics(self) -> Dict[str, Any]:
        """Get context assembly performance metrics."""
        memory_health = await self.memory_manager.get_health_status()

        return {
            "memory_manager_healthy": memory_health.get("manager_initialized", False),
            "tier_status": memory_health.get("tiers", {}),
            "entity_extraction_enabled": self.enable_entity_extraction,
            "max_latency_budget_ms": self.max_retrieval_latency_ms,
            "memory_metrics": memory_health.get("health_metrics", {}),
        }

    def configure_performance(
        self, max_latency_ms: int = None, enable_entity_extraction: bool = None
    ) -> None:
        """Configure performance parameters.

        Args:
            max_latency_ms: Maximum latency budget for memory retrieval
            enable_entity_extraction: Whether to extract entities for targeted retrieval
        """
        if max_latency_ms is not None:
            self.max_retrieval_latency_ms = max_latency_ms
            logger.info(
                f"⚙️  Context assembler latency budget set to {max_latency_ms}ms"
            )

        if enable_entity_extraction is not None:
            self.enable_entity_extraction = enable_entity_extraction
            logger.info(
                f"⚙️  Entity extraction {'enabled' if enable_entity_extraction else 'disabled'}"
            )


# Factory function for easy integration
async def create_enhanced_context_assembler(
    memory_manager: MemoryManager = None,
) -> EnhancedContextAssembler:
    """Create enhanced context assembler with memory manager.

    Args:
        memory_manager: Initialized memory manager, creates new one if None

    Returns:
        Configured enhanced context assembler
    """
    if memory_manager is None:
        from ...memory import get_memory_manager

        memory_manager = await get_memory_manager()

    return EnhancedContextAssembler(memory_manager)
