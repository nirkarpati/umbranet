"""Central Memory Manager for coordinating all 4 memory tiers (RAG++ hierarchy).

This module implements the unified memory interface that orchestrates:
- Tier 1: Short-term Memory (Redis) - Working conversation context
- Tier 2: Episodic Memory (PostgreSQL+pgvector) - Searchable interaction history  
- Tier 3: Semantic Memory (Neo4j) - Knowledge graph of entities/relationships
- Tier 4: Procedural Memory (PostgreSQL) - User preferences and behavioral rules
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core.config import settings
from .base import MemoryConfig, MemoryHealthMetrics, MemoryQuery, MemoryTier
from .tiers.short_term import ShortTermMemoryClient
from .tiers.episodic import EpisodicMemoryStore  
from .tiers.semantic import SemanticMemoryStore
from .tiers.procedural import ProceduralMemoryStore

logger = logging.getLogger(__name__)


class MemoryManagerError(Exception):
    """Exception raised for memory manager operations."""
    pass


class MemoryManager:
    """Central coordinator for all memory operations in the RAG++ hierarchy.
    
    The Memory Manager serves as the single entry point for:
    - Storing interactions across appropriate memory tiers
    - Retrieving contextual memories with parallel tier querying
    - Health monitoring and performance optimization
    - Graceful degradation when memory tiers are unavailable
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize memory manager with configuration.
        
        Args:
            config: Memory configuration, uses defaults from settings if not provided
        """
        self.config = config or self._create_default_config()
        
        # Initialize memory tiers
        self.short_term = ShortTermMemoryClient(max_token_budget=self.config.max_tokens)
        self.episodic = EpisodicMemoryStore()
        self.semantic = SemanticMemoryStore() 
        self.procedural = ProceduralMemoryStore()
        
        # Health monitoring
        self.health_metrics = MemoryHealthMetrics()
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._is_initialized = False
        
        logger.info("Memory Manager initialized with 4-tier RAG++ hierarchy")
    
    def _create_default_config(self) -> MemoryConfig:
        """Create default memory configuration from app settings."""
        return MemoryConfig(
            redis_url=settings.redis_url,
            postgres_url=settings.postgres_url,
            neo4j_uri=settings.neo4j_url,
            neo4j_user=settings.neo4j_user,
            neo4j_password=settings.neo4j_password,
            embedding_model="all-MiniLM-L6-v2",
            max_tokens=2000
        )
    
    async def initialize(self) -> None:
        """Initialize all memory tiers and start health monitoring."""
        if self._is_initialized:
            logger.warning("Memory Manager already initialized")
            return
            
        logger.info("🧠 Initializing Memory Manager - RAG++ 4-Tier System")
        start_time = datetime.utcnow()
        
        # Initialize all tiers in parallel for performance
        init_tasks = [
            self._init_tier_safe("Short-term (Redis)", self.short_term),
            self._init_tier_safe("Episodic (PostgreSQL+pgvector)", self.episodic), 
            self._init_tier_safe("Semantic (Neo4j)", self.semantic),
            self._init_tier_safe("Procedural (PostgreSQL)", self.procedural)
        ]
        
        results = await asyncio.gather(*init_tasks, return_exceptions=True)
        
        # Log initialization results
        successful_tiers = sum(1 for result in results if result is True)
        total_tiers = len(results)
        
        initialization_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(
            f"✅ Memory Manager initialization completed: "
            f"{successful_tiers}/{total_tiers} tiers ready "
            f"({initialization_time:.3f}s)"
        )
        
        # Start health monitoring if enabled
        if self.config.enable_health_monitoring:
            self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
            logger.info("💓 Health monitoring started")
        
        self._is_initialized = True
    
    async def _init_tier_safe(self, tier_name: str, tier: Any) -> bool:
        """Safely initialize a memory tier with error handling."""
        try:
            # Check if tier has initialize method
            if hasattr(tier, 'initialize'):
                await tier.initialize()
            logger.info(f"   ✅ {tier_name} initialized")
            return True
        except Exception as e:
            logger.error(f"   ❌ {tier_name} initialization failed: {e}")
            return False
    
    async def store_interaction(
        self, 
        user_id: str, 
        interaction: Dict[str, Any]
    ) -> Dict[str, str]:
        """Store interaction across all appropriate memory tiers.
        
        This is the main entry point for storing user interactions. The interaction
        is processed and stored in multiple tiers based on its content and type.
        
        Args:
            user_id: User identifier
            interaction: Interaction data containing content, metadata, etc.
            
        Returns:
            Dictionary with storage results and identifiers
        """
        if not self._is_initialized:
            logger.warning("Memory Manager not initialized, skipping storage")
            return {"status": "error", "reason": "not_initialized"}
        
        # Enrich interaction data
        interaction_data = {
            **interaction,
            "timestamp": datetime.utcnow().isoformat(),
            "interaction_id": f"{user_id}_{datetime.utcnow().timestamp()}",
            "user_id": user_id
        }
        
        logger.debug(f"📝 Storing interaction for user {user_id}: {interaction.get('content', '')[:100]}...")
        
        # Use tier router to decide which tiers to store in
        from .services.memory_tier_router import create_memory_tier_router, MemoryTier
        
        async with await create_memory_tier_router() as router:
            routing_result = await router.route_interaction(
                user_message=interaction.get('content', ''),
                assistant_response=interaction.get('assistant_response', ''),
                user_id=user_id
            )
        
        logger.info(f"🧠 Memory tier routing: {[tier.value for tier in routing_result.recommended_tiers]}")
        for tier, reason in routing_result.reasoning.items():
            logger.info(f"   • {tier}: {reason}")
        
        # Storage tasks for recommended tiers only
        storage_tasks = []
        storage_results = {}
        tier_names = []
        
        try:
            for tier in routing_result.recommended_tiers:
                tier_name = tier.value
                tier_names.append(tier_name)
                
                if tier == MemoryTier.SHORT_TERM:
                    storage_tasks.append(
                        self._store_in_tier_safe("short_term", self.short_term, user_id, interaction_data)
                    )
                elif tier == MemoryTier.EPISODIC:
                    storage_tasks.append(
                        self._store_in_tier_safe("episodic", self.episodic, user_id, interaction_data)
                    )
                elif tier == MemoryTier.SEMANTIC and self.config.enable_semantic_extraction:
                    storage_tasks.append(
                        self._store_in_tier_safe("semantic", self.semantic, user_id, interaction_data)
                    )
                elif tier == MemoryTier.PROCEDURAL:
                    storage_tasks.append(
                        self._store_in_tier_safe("procedural", self.procedural, user_id, interaction_data)
                    )
            
            # Execute storage in parallel
            results = await asyncio.gather(*storage_tasks, return_exceptions=True)
            
            # Process results for the tiers we actually used
            for tier_name, result in zip(tier_names, results):
                if isinstance(result, Exception):
                    logger.error(f"Storage failed in {tier_name}: {result}")
                    storage_results[f"{tier_name}_status"] = "error"
                else:
                    storage_results[f"{tier_name}_id"] = result
                    storage_results[f"{tier_name}_status"] = "success"
            
            return {
                "status": "stored",
                "interaction_id": interaction_data["interaction_id"],
                "timestamp": interaction_data["timestamp"],
                **storage_results
            }
            
        except Exception as e:
            logger.error(f"Failed to store interaction for user {user_id}: {e}")
            return {
                "status": "error", 
                "reason": str(e),
                "interaction_id": interaction_data.get("interaction_id")
            }
    
    async def _store_in_tier_safe(
        self, 
        tier_name: str, 
        tier: Any, 
        user_id: str, 
        data: Dict[str, Any]
    ) -> Optional[str]:
        """Safely store data in a memory tier with error handling."""
        try:
            # Different tiers have different storage methods
            if tier_name == "short_term":
                # Short-term memory expects individual parameters
                async with tier:
                    await tier.add_turn(
                        user_id=user_id,
                        user_message=data.get("content", ""),
                        assistant_response=data.get("assistant_response", ""),
                        metadata=data.get("metadata", {})
                    )
                return f"short_term_{user_id}_{data['timestamp']}"
                
            elif tier_name == "episodic":
                # Use curator to decide if/how to store in episodic memory
                from .services.episodic_curator import create_episodic_curator
                
                async with await create_episodic_curator() as curator:
                    curation_result = await curator.curate_interaction(
                        user_message=data.get("content", ""),
                        assistant_response=data.get("assistant_response", ""),
                        user_id=user_id
                    )
                
                if not curation_result.should_store:
                    logger.info(f"📝 Episodic curator decided not to store: {curation_result.reasoning}")
                    return f"episodic_skipped_{user_id}_{data['timestamp']}"
                
                logger.info(f"📝 Episodic curator storing with importance {curation_result.importance_score:.1f}: {curation_result.summary[:50]}...")
                
                # Store the curated content
                async with tier:
                    # Include session_id and curation metadata
                    enhanced_metadata = data.get("metadata", {})
                    if session_id := data.get("session_id"):
                        enhanced_metadata["session_id"] = session_id
                    
                    # Add curation metadata
                    enhanced_metadata.update({
                        "importance_score": curation_result.importance_score,
                        "tags": curation_result.tags,
                        "summary": curation_result.summary,
                        "curation_method": "llm" if settings.openai_api_key else "rule_based"
                    })
                        
                    return await tier.log_interaction(
                        user_id=user_id,
                        user_message=curation_result.content_to_store.split("\nAssistant:")[0].replace("User: ", ""),
                        assistant_response=curation_result.content_to_store.split("\nAssistant: ")[-1] if "\nAssistant: " in curation_result.content_to_store else data.get("assistant_response", ""),
                        metadata=enhanced_metadata
                    )
                    
            elif tier_name == "semantic":
                # Semantic memory extracts entities and relationships
                async with tier:
                    result = await tier.extract_and_store_entities(
                        user_id=user_id,
                        user_message=data.get("content", ""),
                        assistant_response=data.get("assistant_response", "")
                    )
                    return f"semantic_{len(result.entities)}_{len(result.relationships)}"
                    
            elif tier_name == "procedural":
                # Procedural memory stores preferences and rules
                async with tier:
                    # This would involve preference extraction from the interaction
                    # For now, return a placeholder
                    return f"procedural_{user_id}_{data['timestamp']}"
                    
        except Exception as e:
            logger.error(f"Failed to store in {tier_name} tier: {e}")
            raise e
    
    # Removed _should_update_procedural - now using intelligent tier routing
    
    async def recall_context(
        self, 
        user_id: str, 
        current_input: str, 
        max_latency_ms: int = None
    ) -> Dict[str, Any]:
        """Retrieve contextual memories with parallel tier querying and latency budget.
        
        This is the main context retrieval method that queries all memory tiers
        in parallel to assemble a comprehensive context for the AI response.
        
        Args:
            user_id: User identifier  
            current_input: Current user input to contextualize
            max_latency_ms: Maximum latency budget (uses config default if None)
            
        Returns:
            Dictionary containing context from all memory tiers
        """
        if not self._is_initialized:
            logger.warning("Memory Manager not initialized, returning empty context")
            return {"status": "error", "reason": "not_initialized"}
        
        max_latency_ms = max_latency_ms or self.config.default_retrieval_timeout_ms
        start_time = datetime.utcnow()
        
        logger.debug(f"🔍 Recalling context for user {user_id}: {current_input[:50]}...")
        
        try:
            # Use asyncio.wait_for to enforce latency budget
            context = await asyncio.wait_for(
                self._parallel_context_retrieval(user_id, current_input),
                timeout=max_latency_ms / 1000.0
            )
            
            # Record successful retrieval metrics
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.health_metrics.record_retrieval(latency, success=True)
            
            logger.debug(f"✅ Context recall completed in {latency:.2f}ms")
            return context
            
        except asyncio.TimeoutError:
            # Record timeout and return partial context
            self.health_metrics.record_retrieval(max_latency_ms, success=False)
            
            logger.warning(f"⏰ Context recall timeout after {max_latency_ms}ms, returning partial context")
            
            return {
                "status": "partial",
                "reason": "timeout",
                "short_term": await self._quick_short_term_retrieval(user_id),
                "retrieval_timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Context recall failed for user {user_id}: {e}")
            self.health_metrics.record_retrieval(0, success=False)
            
            return {
                "status": "error",
                "reason": str(e),
                "short_term": await self._quick_short_term_retrieval(user_id),
                "retrieval_timestamp": datetime.utcnow().isoformat()
            }
    
    async def _parallel_context_retrieval(self, user_id: str, query: str) -> Dict[str, Any]:
        """Retrieve context from all memory tiers in parallel."""
        
        # Quick entity extraction for semantic/procedural queries
        entities = await self._extract_entities_fast(query)
        
        # Create memory queries for different tiers
        base_query = MemoryQuery(
            user_id=user_id,
            query_text=query,
            entities=entities,
            limit=5
        )
        
        # Create episodic query with lower similarity threshold for more flexible matching
        episodic_query = MemoryQuery(
            user_id=user_id,
            query_text=query,
            entities=entities,
            limit=5,
            similarity_threshold=0.5  # Lower threshold for natural language variations
        )
        
        # Parallel retrieval from all tiers
        retrieval_tasks = [
            self._retrieve_from_tier_safe("short_term", self.short_term, user_id, base_query),
            self._retrieve_from_tier_safe("episodic", self.episodic, user_id, episodic_query),
            self._retrieve_from_tier_safe("semantic", self.semantic, user_id, base_query),
            self._retrieve_from_tier_safe("procedural", self.procedural, user_id, base_query)
        ]
        
        results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
        short_term, episodic, semantic, procedural = results
        
        return {
            "short_term": short_term if not isinstance(short_term, Exception) else {},
            "episodic": episodic if not isinstance(episodic, Exception) else [],
            "semantic": semantic if not isinstance(semantic, Exception) else {},
            "procedural": procedural if not isinstance(procedural, Exception) else [],
            "entities": entities,
            "retrieval_timestamp": datetime.utcnow().isoformat(),
            "status": "success"
        }
    
    async def _retrieve_from_tier_safe(
        self, 
        tier_name: str, 
        tier: Any, 
        user_id: str, 
        query: MemoryQuery
    ) -> Any:
        """Safely retrieve data from a memory tier with error handling."""
        try:
            if tier_name == "short_term":
                async with tier:
                    context = await tier.get_context(user_id)
                    return {
                        "summary": context.summary,
                        "recent_messages": [
                            {
                                "user_message": turn.user_message,
                                "assistant_response": turn.assistant_response, 
                                "timestamp": turn.timestamp.isoformat()
                            }
                            for turn in context.recent_messages
                        ],
                        "token_count": context.token_count
                    }
                    
            elif tier_name == "episodic":
                async with tier:
                    # Use semantic similarity search instead of chronological retrieval
                    episodes = await tier.recall(
                        user_id=user_id, 
                        query_text=query.query_text, 
                        limit=query.limit,
                        similarity_threshold=query.similarity_threshold
                    )
                    return episodes
                    
            elif tier_name == "semantic":
                async with tier:
                    entities = await tier.get_entities_for_user(user_id)
                    relationships = await tier.get_relationships_for_user(user_id) 
                    return {
                        "entities": entities[:10],  # Limit for performance
                        "relationships": relationships[:10],
                        "total_entities": len(entities),
                        "total_relationships": len(relationships)
                    }
                    
            elif tier_name == "procedural":
                async with tier:
                    rules = await tier.get_user_rules(user_id)
                    return rules[:5]  # Top 5 most relevant rules
                    
        except Exception as e:
            logger.error(f"Failed to retrieve from {tier_name} tier: {e}")
            raise e
    
    async def _quick_short_term_retrieval(self, user_id: str) -> Dict[str, Any]:
        """Quick retrieval from short-term memory as fallback."""
        try:
            async with self.short_term:
                context = await self.short_term.get_context(user_id)
                return {
                    "summary": context.summary,
                    "recent_count": len(context.recent_messages),
                    "token_count": context.token_count
                }
        except Exception as e:
            logger.error(f"Quick short-term retrieval failed: {e}")
            return {"summary": None, "recent_count": 0, "token_count": 0}
    
    async def _extract_entities_fast(self, text: str) -> List[str]:
        """Fast entity extraction for query enhancement."""
        # Simple keyword extraction as fallback
        # TODO: Integrate with semantic memory's entity extractor
        words = text.lower().split()
        
        # Filter for potential entities (capitalized words, etc.)
        entities = []
        for word in text.split():
            if word.istitle() and len(word) > 2:
                entities.append(word)
        
        return entities[:10]  # Limit entity count
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of all memory tiers."""
        health_status = {
            "manager_initialized": self._is_initialized,
            "health_metrics": {
                "total_retrievals": self.health_metrics.total_retrievals,
                "success_rate": self.health_metrics.success_rate,
                "average_latency_ms": self.health_metrics.average_latency_ms,
                "last_check": self.health_metrics.last_health_check.isoformat() if self.health_metrics.last_health_check else None
            },
            "tiers": {}
        }
        
        # Check health of each tier
        tier_checks = [
            ("short_term", self.short_term),
            ("episodic", self.episodic), 
            ("semantic", self.semantic),
            ("procedural", self.procedural)
        ]
        
        for tier_name, tier in tier_checks:
            try:
                # Check if tier has health_check method
                if hasattr(tier, 'health_check'):
                    is_healthy = await tier.health_check()
                else:
                    # Basic connectivity check
                    is_healthy = hasattr(tier, '__aenter__')
                    
                health_status["tiers"][tier_name] = {
                    "healthy": is_healthy,
                    "last_check": datetime.utcnow().isoformat()
                }
            except Exception as e:
                health_status["tiers"][tier_name] = {
                    "healthy": False,
                    "error": str(e),
                    "last_check": datetime.utcnow().isoformat()
                }
        
        return health_status
    
    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        logger.info("🩺 Health monitoring loop started")
        
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval_seconds)
                
                health_status = await self.get_health_status()
                self.health_metrics.last_health_check = datetime.utcnow()
                
                # Log unhealthy tiers
                unhealthy_tiers = [
                    tier_name for tier_name, status in health_status["tiers"].items()
                    if not status["healthy"]
                ]
                
                if unhealthy_tiers:
                    logger.warning(f"🚨 Unhealthy memory tiers: {unhealthy_tiers}")
                else:
                    logger.debug("💓 All memory tiers healthy")
                    
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup memory manager resources."""
        logger.info("🧹 Cleaning up Memory Manager resources")
        
        # Cancel health monitoring
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup all tiers
        cleanup_tasks = []
        for tier in [self.short_term, self.episodic, self.semantic, self.procedural]:
            if hasattr(tier, 'cleanup'):
                cleanup_tasks.append(tier.cleanup())
            elif hasattr(tier, 'disconnect'):
                cleanup_tasks.append(tier.disconnect())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self._is_initialized = False
        logger.info("✅ Memory Manager cleanup completed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


async def get_memory_manager() -> MemoryManager:
    """Get or create global memory manager instance."""
    global _memory_manager
    
    if _memory_manager is None:
        _memory_manager = MemoryManager()
        await _memory_manager.initialize()
    
    return _memory_manager