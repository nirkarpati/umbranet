"""RAG++ Memory Hierarchy for the Headless Governor System.

This package implements a sophisticated 4-tier memory system:
- Tier 1: Short-term Memory (Redis) - Working conversation context  
- Tier 2: Episodic Memory (PostgreSQL+pgvector) - Searchable interaction history
- Tier 3: Semantic Memory (Neo4j) - Knowledge graph of entities and relationships
- Tier 4: Procedural Memory (PostgreSQL) - User preferences and behavioral rules

The Memory Manager coordinates all tiers for unified storage and retrieval operations.
"""

from .manager import MemoryManager, get_memory_manager
from .base import MemoryConfig, MemoryQuery, MemoryHealthMetrics

__all__ = [
    "MemoryManager",
    "get_memory_manager", 
    "MemoryConfig",
    "MemoryQuery",
    "MemoryHealthMetrics"
]