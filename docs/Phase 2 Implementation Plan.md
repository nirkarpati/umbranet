# Phase 2 Implementation Plan: RAG++ Memory Hierarchy

**Project:** The Headless Governor System  
**Phase:** 2 (Memory & Intelligence)  
**Version:** 1.0.0  
**Status:** READY FOR IMPLEMENTATION  
**Dependencies:** Phase 1 (The Kernel & Interface) ✅ COMPLETED

## Executive Summary

Phase 2 transforms the Headless Governor from a stateless conversation system into a **memory-enabled AI assistant** that learns, remembers, and adapts to each user's unique context. We implement a sophisticated 4-tier memory hierarchy (RAG++) that enables infinite conversation continuity, relationship understanding, and behavioral learning.

**Objective:** Deploy a memory-aware Governor that recalls past interactions, learns user preferences, and maintains rich contextual understanding across unlimited time spans.

## Architecture Overview

### The RAG++ Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 2 ARCHITECTURE                        │
│                   RAG++ Memory Hierarchy                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Input → [Entity Extraction + Embedding] → Parallel Fetch: │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Tier 1    │  │   Tier 2    │  │   Tier 3    │  │   Tier 4    │ │
│  │ Short-Term  │  │  Episodic   │  │  Semantic   │  │ Procedural  │ │
│  │   Redis     │  │  PGVector   │  │    Neo4j    │  │  Postgres   │ │
│  │ (Working)   │  │ (Episodes)  │  │ (Knowledge) │  │ (Rules)     │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
│           │              │              │              │       │
│           └──────────────┼──────────────┼──────────────┘       │
│                          │              │                      │
│                          ▼              ▼                      ▼
│                 ┌─────────────────────────────────────────────────┐ │
│                 │          Context Assembler                    │ │
│                 │       (Parallel Memory Fusion)               │ │
│                 └─────────────────────────────────────────────────┘ │
│                                      │                             │
│                                      ▼                             │
│                 ┌─────────────────────────────────────────────────┐ │
│                 │         Enhanced State Machine               │ │
│                 │    (Memory-Aware Decision Making)            │ │
│                 └─────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Tier Specifications

#### **Tier 1: Short-Term Memory (Redis Cluster)**
- **Purpose**: Token-managed working context for current conversation
- **Technology**: Redis with distributed locking
- **Data**: Raw message buffer + rolling summary
- **Retention**: Session-scoped with configurable TTL
- **Latency Target**: <10ms

#### **Tier 2: Episodic Memory (PGVector)**
- **Purpose**: Searchable log of all past interactions 
- **Technology**: PostgreSQL with pgvector extension
- **Data**: Embedded conversation snippets with metadata
- **Retention**: Permanent append-only log
- **Latency Target**: <100ms

#### **Tier 3: Semantic Memory (Neo4j)**
- **Purpose**: Probabilistic knowledge graph of relationships
- **Technology**: Neo4j with weighted edges
- **Data**: Entities, relationships, confidence scores
- **Retention**: Permanent with decay algorithms
- **Latency Target**: <200ms

#### **Tier 4: Procedural Memory (PostgreSQL)**
- **Purpose**: User preferences, habits, and behavioral rules
- **Technology**: PostgreSQL with vector search
- **Data**: Preference profiles + embedded behavioral patterns
- **Retention**: User-controlled with versioning
- **Latency Target**: <50ms

## Implementation Steps

### Step 1: Memory Infrastructure (`src/memory/`)

**Goal**: Implement the 4-tier memory storage system with unified interfaces.

**Priority**: CRITICAL - All memory operations depend on this foundation.

#### 1.1 Memory Abstractions

```python
# File: src/memory/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

class MemoryTier(ABC):
    """Base class for all memory tier implementations."""
    
    @abstractmethod
    async def store(self, user_id: str, data: Dict[str, Any]) -> str:
        """Store data and return unique identifier."""
        pass
    
    @abstractmethod 
    async def retrieve(self, user_id: str, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve relevant memories based on query."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if memory tier is healthy."""
        pass

class MemoryQuery(BaseModel):
    """Standardized memory query format."""
    user_id: str
    query_text: Optional[str] = None
    entities: Optional[List[str]] = None
    time_range: Optional[tuple[datetime, datetime]] = None
    limit: int = 10
    similarity_threshold: float = 0.7
```

#### 1.2 Tier 1: Short-Term Memory (Redis)

```python
# File: src/memory/tiers/short_term.py

import redis.asyncio as redis
import json
from typing import List, Dict, Any

class ShortTermMemory(MemoryTier):
    """Redis-based short-term memory with token management."""
    
    def __init__(self, redis_url: str, max_tokens: int = 2000):
        self.redis = redis.from_url(redis_url)
        self.max_tokens = max_tokens
        self.summarizer = LLMSummarizer()  # GPT-4o-mini for cheap summaries
    
    async def store(self, user_id: str, data: Dict[str, Any]) -> str:
        """Store message in buffer, summarize if needed."""
        key = f"session:{user_id}:v1"
        
        async with self.redis.lock(f"{key}:lock", timeout=10):
            # Get current session
            session_data = await self._get_session(key)
            
            # Add new message to buffer
            session_data["buffer"].append(data)
            session_data["token_count"] += self._estimate_tokens(data["content"])
            
            # If over token limit, summarize oldest messages
            if session_data["token_count"] > self.max_tokens:
                session_data = await self._compress_buffer(session_data)
            
            # Save updated session
            await self.redis.hset(key, mapping={
                "buffer": json.dumps(session_data["buffer"]),
                "summary": session_data["summary"],
                "token_count": session_data["token_count"],
                "updated_at": datetime.utcnow().isoformat()
            })
            
            return f"{key}:{len(session_data['buffer'])}"
    
    async def _compress_buffer(self, session_data: Dict) -> Dict:
        """Compress old messages into summary using LLM."""
        messages_to_summarize = session_data["buffer"][:-5]  # Keep last 5 raw
        
        if messages_to_summarize:
            new_summary = await self.summarizer.summarize_messages(
                previous_summary=session_data["summary"],
                messages=messages_to_summarize
            )
            
            return {
                "buffer": session_data["buffer"][-5:],  # Keep last 5 raw
                "summary": new_summary,
                "token_count": self._estimate_tokens(new_summary) + 
                              sum(self._estimate_tokens(msg["content"]) 
                                  for msg in session_data["buffer"][-5:])
            }
        
        return session_data
```

#### 1.3 Tier 2: Episodic Memory (PGVector)

```python
# File: src/memory/tiers/episodic.py

import asyncpg
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class EpisodicMemory(MemoryTier):
    """PostgreSQL + pgvector for episodic memory storage."""
    
    def __init__(self, db_url: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.db_url = db_url
        self.embedding_model = SentenceTransformer(embedding_model)
        self.pool = None
    
    async def initialize(self):
        """Initialize database connection pool and ensure schema."""
        self.pool = await asyncpg.create_pool(self.db_url, min_size=2, max_size=10)
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
                
                CREATE TABLE IF NOT EXISTS episodic_logs (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id TEXT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    content TEXT NOT NULL,
                    embedding vector(384),
                    metadata JSONB DEFAULT '{}',
                    session_id TEXT,
                    message_type TEXT DEFAULT 'conversation'
                );
                
                CREATE INDEX IF NOT EXISTS episodic_logs_user_id_idx ON episodic_logs (user_id);
                CREATE INDEX IF NOT EXISTS episodic_logs_timestamp_idx ON episodic_logs (timestamp);
                CREATE INDEX IF NOT EXISTS episodic_logs_embedding_idx ON episodic_logs 
                    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
            """)
    
    async def store(self, user_id: str, data: Dict[str, Any]) -> str:
        """Store interaction with vector embedding."""
        content = data.get("content", "")
        embedding = self.embedding_model.encode(content).tolist()
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO episodic_logs (user_id, content, embedding, metadata, session_id, message_type)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
            """, user_id, content, embedding, 
                json.dumps(data.get("metadata", {})),
                data.get("session_id"),
                data.get("message_type", "conversation"))
            
            return str(row["id"])
    
    async def retrieve(self, user_id: str, query: MemoryQuery) -> List[Dict[str, Any]]:
        """Retrieve similar episodes using vector similarity."""
        if not query.query_text:
            return []
        
        query_embedding = self.embedding_model.encode(query.query_text).tolist()
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, content, metadata, timestamp, session_id,
                       1 - (embedding <=> $1) as similarity
                FROM episodic_logs
                WHERE user_id = $2 
                  AND 1 - (embedding <=> $1) > $3
                  AND ($4::timestamptz IS NULL OR timestamp >= $4)
                  AND ($5::timestamptz IS NULL OR timestamp <= $5)
                ORDER BY similarity DESC
                LIMIT $6
            """, query_embedding, user_id, query.similarity_threshold,
                query.time_range[0] if query.time_range else None,
                query.time_range[1] if query.time_range else None,
                query.limit)
            
            return [
                {
                    "id": str(row["id"]),
                    "content": row["content"],
                    "metadata": json.loads(row["metadata"]),
                    "timestamp": row["timestamp"],
                    "session_id": row["session_id"],
                    "similarity": float(row["similarity"])
                }
                for row in rows
            ]
```

#### 1.4 Tier 3: Semantic Memory (Neo4j)

```python
# File: src/memory/tiers/semantic.py

from neo4j import AsyncGraphDatabase
from typing import List, Dict, Any, Tuple
import spacy

class SemanticMemory(MemoryTier):
    """Neo4j-based semantic knowledge graph."""
    
    def __init__(self, neo4j_uri: str, username: str, password: str):
        self.driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(username, password))
        self.nlp = spacy.load("en_core_web_sm")
        self.confidence_decay_rate = 0.95  # Daily decay factor
    
    async def store(self, user_id: str, data: Dict[str, Any]) -> str:
        """Extract entities and relationships, store in graph."""
        content = data.get("content", "")
        entities = self._extract_entities(content)
        relationships = self._extract_relationships(content, entities)
        
        async with self.driver.session() as session:
            # Create user node if not exists
            await session.write_transaction(self._create_user_node, user_id)
            
            # Store entities and relationships
            for entity in entities:
                await session.write_transaction(self._create_entity, user_id, entity, data)
            
            for rel in relationships:
                await session.write_transaction(self._create_relationship, user_id, rel, data)
            
            return f"semantic_{user_id}_{data.get('timestamp', 'now')}"
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities using spaCy."""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "PRODUCT"]:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        
        return entities
    
    async def _create_relationship(self, tx, user_id: str, relationship: Dict, context: Dict):
        """Create weighted relationship between entities."""
        await tx.run("""
            MATCH (u:User {id: $user_id})
            MERGE (e1:Entity {name: $entity1, user_id: $user_id})
            MERGE (e2:Entity {name: $entity2, user_id: $user_id})
            MERGE (e1)-[r:RELATES_TO {type: $rel_type}]->(e2)
            SET r.weight = COALESCE(r.weight, 0) + $weight_increment,
                r.last_mentioned = datetime($timestamp),
                r.confidence = COALESCE(r.confidence, 0.5) + $confidence_boost,
                r.context = $context
            WITH u, e1, e2
            MERGE (u)-[:KNOWS]->(e1)
            MERGE (u)-[:KNOWS]->(e2)
        """, 
            user_id=user_id,
            entity1=relationship["entity1"],
            entity2=relationship["entity2"], 
            rel_type=relationship["type"],
            weight_increment=1.0,
            confidence_boost=0.1,
            timestamp=context.get("timestamp", datetime.utcnow().isoformat()),
            context=json.dumps(context.get("metadata", {}))
        )
```

#### 1.5 Tier 4: Procedural Memory (PostgreSQL)

```python
# File: src/memory/tiers/procedural.py

import asyncpg
from typing import Dict, Any, List

class ProceduralMemory(MemoryTier):
    """PostgreSQL-based user preferences and behavioral rules."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.pool = None
    
    async def initialize(self):
        """Initialize procedural memory schema."""
        self.pool = await asyncpg.create_pool(self.db_url)
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    preference_key TEXT NOT NULL,
                    preference_value JSONB NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(user_id, category, preference_key)
                );
                
                CREATE TABLE IF NOT EXISTS behavioral_rules (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id TEXT NOT NULL,
                    rule_text TEXT NOT NULL,
                    rule_embedding vector(384),
                    trigger_contexts TEXT[] DEFAULT '{}',
                    activation_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE INDEX behavioral_rules_embedding_idx ON behavioral_rules 
                    USING ivfflat (rule_embedding vector_cosine_ops) WITH (lists = 50);
            """)
    
    async def store_preference(self, user_id: str, category: str, 
                             key: str, value: Any, confidence: float = 0.8) -> str:
        """Store or update user preference."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO user_preferences (user_id, category, preference_key, preference_value, confidence)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (user_id, category, preference_key) 
                DO UPDATE SET 
                    preference_value = $4,
                    confidence = GREATEST(EXCLUDED.confidence, user_preferences.confidence),
                    updated_at = NOW()
                RETURNING id
            """, user_id, category, key, json.dumps(value), confidence)
            
            return str(row["id"])
```

---

### Step 2: Enhanced Context Assembly (`src/memory/context/`)

**Goal**: Replace mock context assembly with parallel memory retrieval system.

#### 2.1 Memory-Aware Context Assembler

```python
# File: src/memory/context/enhanced_assembler.py

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

class EnhancedContextAssembler:
    """Memory-aware context assembler with parallel retrieval."""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.entity_extractor = EntityExtractor()
    
    async def assemble_context(
        self, 
        user_id: str, 
        current_input: str, 
        state: GovernorState,
        available_tools: List[str] = None
    ) -> str:
        """Assemble context using parallel memory retrieval."""
        
        # Extract entities and generate embeddings
        entities = await self.entity_extractor.extract(current_input)
        
        # Parallel memory retrieval
        memory_results = await asyncio.gather(
            self._get_short_term_context(user_id, state),
            self._get_episodic_context(user_id, current_input, entities),
            self._get_semantic_context(user_id, entities),
            self._get_procedural_context(user_id, current_input),
            return_exceptions=True
        )
        
        short_term, episodic, semantic, procedural = memory_results
        
        # Construct enhanced prompt
        return self._build_memory_aware_prompt(
            user_id=user_id,
            current_input=current_input,
            state=state,
            short_term_context=short_term if not isinstance(short_term, Exception) else {},
            episodic_memories=episodic if not isinstance(episodic, Exception) else [],
            semantic_knowledge=semantic if not isinstance(semantic, Exception) else {},
            procedural_rules=procedural if not isinstance(procedural, Exception) else [],
            available_tools=available_tools or []
        )
    
    async def _get_episodic_context(
        self, 
        user_id: str, 
        query_text: str, 
        entities: List[str]
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant past episodes."""
        memory_query = MemoryQuery(
            user_id=user_id,
            query_text=query_text,
            entities=entities,
            limit=5,
            similarity_threshold=0.75
        )
        
        return await self.memory_manager.episodic.retrieve(user_id, memory_query)
    
    def _build_memory_aware_prompt(self, **context_data) -> str:
        """Build sophisticated prompt with memory integration."""
        sections = [
            self._build_persona_section(),
            self._build_memory_section(context_data),
            self._build_tools_section(context_data.get("available_tools", [])),
            self._build_situation_section(context_data)
        ]
        
        return "\n\n".join(section for section in sections if section)
    
    def _build_memory_section(self, context_data: Dict) -> str:
        """Build memory context section with weighted integration."""
        memory_parts = []
        
        # Short-term (working memory)
        if context_data.get("short_term_context"):
            memory_parts.append(f"RECENT CONVERSATION:\n{context_data['short_term_context']['summary']}")
        
        # Episodic memories (past interactions)
        if context_data.get("episodic_memories"):
            episodes = context_data["episodic_memories"][:3]  # Top 3 most relevant
            episode_text = "\n".join([
                f"- {ep['content']} (relevance: {ep['similarity']:.2f})"
                for ep in episodes
            ])
            memory_parts.append(f"RELEVANT PAST INTERACTIONS:\n{episode_text}")
        
        # Semantic knowledge (relationships and facts)
        if context_data.get("semantic_knowledge"):
            knowledge = context_data["semantic_knowledge"]
            if knowledge.get("entities"):
                entities_text = ", ".join(knowledge["entities"])
                memory_parts.append(f"KNOWN ENTITIES: {entities_text}")
            
            if knowledge.get("relationships"):
                rel_text = "\n".join([
                    f"- {rel['entity1']} {rel['type']} {rel['entity2']} (confidence: {rel['confidence']:.2f})"
                    for rel in knowledge["relationships"][:5]
                ])
                memory_parts.append(f"KNOWN RELATIONSHIPS:\n{rel_text}")
        
        # Procedural rules (preferences and habits)
        if context_data.get("procedural_rules"):
            rules = context_data["procedural_rules"][:3]  # Top 3 most relevant
            rules_text = "\n".join([
                f"- {rule['description']} (relevance: {rule['relevance']:.2f})"
                for rule in rules
            ])
            memory_parts.append(f"USER PREFERENCES & HABITS:\n{rules_text}")
        
        return "\n\n".join(memory_parts) if memory_parts else ""
```

---

### Step 3: O-E-R Learning Loop (`src/memory/learning/`)

**Goal**: Implement the Observe-Extract-Reflect learning cycle for continuous adaptation.

#### 3.1 Learning Loop Components

```python
# File: src/memory/learning/oer_loop.py

from typing import Dict, List, Any
import asyncio
from datetime import datetime, timedelta

class OERLearningLoop:
    """Observe-Extract-Reflect learning loop for continuous adaptation."""
    
    def __init__(self, memory_manager: MemoryManager, llm_client: LLMClient):
        self.memory_manager = memory_manager
        self.llm_client = llm_client
        self.entity_extractor = EntityExtractor()
        self.relationship_extractor = RelationshipExtractor()
    
    async def observe(self, interaction_data: Dict[str, Any]) -> str:
        """OBSERVE: Real-time processing of new interactions."""
        user_id = interaction_data["user_id"]
        
        # Store in all appropriate memory tiers
        storage_tasks = [
            self.memory_manager.short_term.store(user_id, interaction_data),
            self.memory_manager.episodic.store(user_id, interaction_data),
        ]
        
        # Extract entities and relationships for semantic memory
        entities = await self.entity_extractor.extract(interaction_data["content"])
        if entities:
            semantic_data = {**interaction_data, "entities": entities}
            storage_tasks.append(
                self.memory_manager.semantic.store(user_id, semantic_data)
            )
        
        # Execute storage in parallel
        results = await asyncio.gather(*storage_tasks, return_exceptions=True)
        
        return f"observed_{user_id}_{datetime.utcnow().timestamp()}"
    
    async def extract(self, user_id: str, time_window: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """EXTRACT: Pattern extraction from recent interactions."""
        end_time = datetime.utcnow()
        start_time = end_time - time_window
        
        # Get recent episodes for pattern analysis
        query = MemoryQuery(
            user_id=user_id,
            time_range=(start_time, end_time),
            limit=50
        )
        
        recent_episodes = await self.memory_manager.episodic.retrieve(user_id, query)
        
        if not recent_episodes:
            return {"patterns": [], "preferences": [], "relationships": []}
        
        # Extract patterns using LLM analysis
        patterns = await self._extract_behavioral_patterns(recent_episodes)
        preferences = await self._extract_preferences(recent_episodes)
        relationships = await self._extract_relationship_updates(recent_episodes)
        
        return {
            "patterns": patterns,
            "preferences": preferences, 
            "relationships": relationships,
            "extracted_at": datetime.utcnow().isoformat()
        }
    
    async def reflect(self, user_id: str, daily_reflection: bool = False) -> Dict[str, Any]:
        """REFLECT: Nightly consolidation and graph updates."""
        if daily_reflection:
            # Full daily reflection - more comprehensive
            return await self._daily_reflection(user_id)
        else:
            # Incremental reflection - lighter weight
            return await self._incremental_reflection(user_id)
    
    async def _daily_reflection(self, user_id: str) -> Dict[str, Any]:
        """Comprehensive daily reflection and memory consolidation."""
        yesterday = datetime.utcnow() - timedelta(days=1)
        today = datetime.utcnow()
        
        # Get all interactions from the past day
        daily_query = MemoryQuery(
            user_id=user_id,
            time_range=(yesterday, today),
            limit=200
        )
        
        daily_episodes = await self.memory_manager.episodic.retrieve(user_id, daily_query)
        
        if not daily_episodes:
            return {"status": "no_interactions", "date": today.date().isoformat()}
        
        # Comprehensive analysis
        reflection_tasks = [
            self._analyze_daily_themes(daily_episodes),
            self._update_relationship_weights(user_id, daily_episodes),
            self._consolidate_preferences(user_id, daily_episodes),
            self._identify_new_behavioral_rules(daily_episodes)
        ]
        
        results = await asyncio.gather(*reflection_tasks, return_exceptions=True)
        themes, relationship_updates, preference_updates, new_rules = results
        
        # Apply updates to memory systems
        await self._apply_reflection_updates(user_id, {
            "themes": themes,
            "relationships": relationship_updates,
            "preferences": preference_updates,
            "new_rules": new_rules
        })
        
        return {
            "status": "completed",
            "date": today.date().isoformat(),
            "interactions_analyzed": len(daily_episodes),
            "themes_identified": len(themes) if isinstance(themes, list) else 0,
            "relationships_updated": len(relationship_updates) if isinstance(relationship_updates, list) else 0,
            "preferences_updated": len(preference_updates) if isinstance(preference_updates, list) else 0,
            "new_rules_learned": len(new_rules) if isinstance(new_rules, list) else 0
        }
```

---

### Step 4: Memory Manager Integration (`src/memory/manager.py`)

**Goal**: Unified memory interface with health monitoring and performance optimization.

#### 4.1 Central Memory Manager

```python
# File: src/memory/manager.py

from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime

class MemoryManager:
    """Central coordinator for all memory operations."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.short_term = ShortTermMemory(config.redis_url, config.max_tokens)
        self.episodic = EpisodicMemory(config.postgres_url, config.embedding_model)
        self.semantic = SemanticMemory(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
        self.procedural = ProceduralMemory(config.postgres_url)
        self.learning_loop = OERLearningLoop(self, config.llm_client)
        
        self.health_metrics = MemoryHealthMetrics()
    
    async def initialize(self):
        """Initialize all memory tiers."""
        init_tasks = [
            self.episodic.initialize(),
            self.semantic.initialize(),
            self.procedural.initialize()
        ]
        
        await asyncio.gather(*init_tasks, return_exceptions=True)
        
        # Start health monitoring
        asyncio.create_task(self._health_monitor_loop())
    
    async def store_interaction(
        self, 
        user_id: str, 
        interaction: Dict[str, Any]
    ) -> Dict[str, str]:
        """Store interaction across all appropriate memory tiers."""
        # Add timestamp and interaction ID
        interaction_data = {
            **interaction,
            "timestamp": datetime.utcnow().isoformat(),
            "interaction_id": f"{user_id}_{datetime.utcnow().timestamp()}"
        }
        
        # Trigger O-E-R observation phase
        observation_id = await self.learning_loop.observe(interaction_data)
        
        return {
            "status": "stored",
            "observation_id": observation_id,
            "timestamp": interaction_data["timestamp"]
        }
    
    async def recall_context(
        self, 
        user_id: str, 
        current_input: str, 
        max_latency_ms: int = 500
    ) -> Dict[str, Any]:
        """Retrieve contextual memories with latency budget."""
        start_time = datetime.utcnow()
        
        try:
            # Use asyncio.wait_for to enforce latency budget
            context = await asyncio.wait_for(
                self._parallel_context_retrieval(user_id, current_input),
                timeout=max_latency_ms / 1000.0
            )
            
            # Record successful retrieval metrics
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.health_metrics.record_retrieval(latency, success=True)
            
            return context
            
        except asyncio.TimeoutError:
            # Record timeout and return partial context
            self.health_metrics.record_retrieval(max_latency_ms, success=False)
            
            return {
                "status": "partial",
                "reason": "timeout",
                "short_term": await self._quick_short_term_retrieval(user_id)
            }
    
    async def _parallel_context_retrieval(self, user_id: str, query: str) -> Dict[str, Any]:
        """Retrieve context from all memory tiers in parallel."""
        entities = await self._extract_entities_fast(query)
        
        # Parallel retrieval from all tiers
        retrieval_tasks = [
            self.short_term.retrieve(user_id, {"query": query}),
            self.episodic.retrieve(user_id, MemoryQuery(
                user_id=user_id, 
                query_text=query, 
                entities=entities,
                limit=5
            )),
            self.semantic.retrieve(user_id, {"entities": entities}),
            self.procedural.retrieve(user_id, {"query": query, "context": entities})
        ]
        
        results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
        short_term, episodic, semantic, procedural = results
        
        return {
            "short_term": short_term if not isinstance(short_term, Exception) else {},
            "episodic": episodic if not isinstance(episodic, Exception) else [],
            "semantic": semantic if not isinstance(semantic, Exception) else {},
            "procedural": procedural if not isinstance(procedural, Exception) else [],
            "retrieval_timestamp": datetime.utcnow().isoformat()
        }
```

---

### Step 5: Integration & Performance Optimization

**Goal**: Integrate memory system with existing Governor components and optimize for production.

#### 5.1 Enhanced State Machine Integration

```python
# File: src/governor/state_machine/enhanced_graph.py

class MemoryAwareGovernorGraph(GovernorGraph):
    """Enhanced state machine with memory integration."""
    
    def __init__(self, memory_manager: MemoryManager):
        super().__init__()
        self.memory_manager = memory_manager
        self.enhanced_assembler = EnhancedContextAssembler(memory_manager)
    
    async def invoke(self, graph_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process with memory-aware context assembly."""
        state = graph_input.get("state")
        event = graph_input.get("event")
        
        # Store interaction in memory
        await self.memory_manager.store_interaction(event.user_id, {
            "content": event.content,
            "message_type": event.message_type.value,
            "channel": event.channel.value,
            "session_id": event.session_id,
            "metadata": event.metadata
        })
        
        # Assemble memory-aware context
        context_prompt = await self.enhanced_assembler.assemble_context(
            user_id=event.user_id,
            current_input=event.content,
            state=state,
            available_tools=graph_input.get("available_tools", [])
        )
        
        # Process with enhanced context
        response = await self._process_with_memory_context(
            state, event, context_prompt
        )
        
        # Store response in memory
        await self.memory_manager.store_interaction(event.user_id, {
            "content": response.content,
            "message_type": "assistant_response",
            "channel": response.channel.value,
            "session_id": response.session_id,
            "metadata": response.metadata
        })
        
        return {"response": response}
```

---

## Implementation Timeline

### **Phase 2.1: Memory Infrastructure (Weeks 1-2)**
- [ ] Memory tier abstractions and base classes
- [ ] Redis short-term memory with token management
- [ ] PostgreSQL episodic memory with pgvector
- [ ] Neo4j semantic memory with weighted graphs
- [ ] PostgreSQL procedural memory with preferences

### **Phase 2.2: Context Assembly Enhancement (Week 3)**
- [ ] Enhanced context assembler with parallel retrieval
- [ ] Memory query optimization and caching
- [ ] Latency budgeting and fallback mechanisms
- [ ] Performance monitoring and metrics

### **Phase 2.3: O-E-R Learning Loop (Week 4)**
- [ ] Real-time observation and storage pipeline
- [ ] Pattern extraction and preference learning
- [ ] Daily reflection and memory consolidation
- [ ] Graph dynamics and relationship updates

### **Phase 2.4: Integration & Optimization (Week 5)**
- [ ] State machine integration with memory awareness
- [ ] Performance optimization and caching strategies
- [ ] Health monitoring and alerting
- [ ] Memory cleanup and retention policies

### **Phase 2.5: Testing & Validation (Week 6)**
- [ ] Comprehensive testing suite for all memory operations
- [ ] Performance benchmarking and load testing
- [ ] Memory accuracy and recall validation
- [ ] End-to-end conversation continuity testing

## Success Criteria

**Phase 2 is complete when:**

✅ **Infinite Context**: Users can reference conversations from weeks/months ago  
✅ **Relationship Memory**: System remembers who users know and their relationships  
✅ **Preference Learning**: Adaptive behavior based on learned user patterns  
✅ **Performance**: <500ms memory retrieval latency for context assembly  
✅ **Accuracy**: >85% relevance score for retrieved memories  
✅ **Scalability**: Support for 10,000+ users with efficient memory management  

## Risk Mitigation

**High Risk**: Memory retrieval latency
- *Mitigation*: Parallel retrieval with latency budgets and graceful degradation
- *Fallback*: Short-term memory only if other tiers timeout

**Medium Risk**: Graph knowledge accuracy
- *Mitigation*: Confidence scoring and decay mechanisms
- *Fallback*: Conservative confidence thresholds for high-stakes decisions

**Medium Risk**: Storage costs and scaling
- *Mitigation*: Tiered retention policies and compression strategies
- *Fallback*: Configurable memory limits per user tier

**Low Risk**: Entity extraction accuracy
- *Mitigation*: Multiple extraction methods and human-in-the-loop validation
- *Fallback*: Keyword-based extraction as backup

## Phase 3 Preparation

Phase 2 establishes the memory foundation for Phase 3's advanced capabilities:

- **Memory systems** will support advanced reasoning and planning
- **Learning loops** will enable meta-learning and transfer learning
- **Knowledge graphs** will support multi-user collaboration and shared knowledge
- **Performance optimizations** will enable real-time multi-modal processing

---

**Next Steps**: Begin with **Step 1: Memory Infrastructure** implementation, starting with the memory abstractions and Redis short-term memory system.