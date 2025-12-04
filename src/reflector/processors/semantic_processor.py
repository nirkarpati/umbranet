"""Processor for Tier 3: Semantic Memory operations."""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from ...memory.tiers.semantic import SemanticMemoryStore
from ..queue.schemas import MemoryReflectionJob

logger = logging.getLogger(__name__)

class SemanticProcessor:
    """Processor for Tier 3: Semantic Memory operations."""
    
    def __init__(self):
        self.semantic_store: Optional[SemanticMemoryStore] = None
        self.processing_count = 0
        self.success_count = 0
        self.error_count = 0
    
    async def initialize(self) -> None:
        """Initialize semantic memory store."""
        try:
            self.semantic_store = SemanticMemoryStore()
            # Test connection
            async with self.semantic_store:
                pass  # Connection test
            logger.info("✅ Semantic processor initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize semantic processor: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Cleanup semantic processor resources."""
        if self.semantic_store:
            # SemanticMemoryStore cleanup handled by context manager
            pass
        logger.info("🧹 Semantic processor cleaned up")
    
    async def process_job(self, job: MemoryReflectionJob) -> Dict[str, Any]:
        """Process semantic memory for reflection job with intelligent pattern detection."""
        start_time = datetime.utcnow()
        self.processing_count += 1
        
        try:
            logger.debug(f"🕸️ Processing semantic memory for job {job.job_id}")
            
            # Use intelligent extraction with pattern detection
            async with self.semantic_store:
                extraction_result = await self._extract_with_pattern_detection(
                    user_id=job.user_id,
                    user_message=job.user_message,
                    assistant_response=job.assistant_response
                )
                
                if extraction_result.get("skip_semantic"):
                    # Decision to skip semantic storage entirely
                    self.success_count += 1
                    processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    logger.info(f"📋 Semantic storage skipped for job {job.job_id}: {extraction_result.get('reasoning', 'Pattern-based decision')} in {processing_time:.1f}ms")
                    
                    return {
                        "status": "skipped",
                        "result_id": f"semantic_skip_{job.job_id}",
                        "entities_extracted": 0,
                        "relationships_extracted": 0,
                        "reasoning": extraction_result.get("reasoning"),
                        "processing_time_ms": processing_time
                    }
                
                # Extract approved entities and relationships
                entities = extraction_result.get("entities", [])
                relationships = extraction_result.get("relationships", [])
                
                # Store in knowledge graph using existing interface
                if entities or relationships:
                    final_result = await self._store_approved_entities_and_relationships(
                        entities=entities,
                        relationships=relationships,
                        user_id=job.user_id
                    )
                else:
                    final_result = {"entities": [], "relationships": []}
                
                entity_count = len(final_result.get("entities", []))
                relationship_count = len(final_result.get("relationships", []))
                
                self.success_count += 1
                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                logger.info(f"✅ Semantic processing completed for job {job.job_id}: {entity_count} entities, {relationship_count} relationships in {processing_time:.1f}ms")
                
                return {
                    "status": "stored",
                    "result_id": f"semantic_{entity_count}_{relationship_count}",
                    "entities_extracted": entity_count,
                    "relationships_extracted": relationship_count,
                    "entities": [{"name": e.get("name"), "type": e.get("entity_type")} for e in final_result.get("entities", [])],
                    "relationships": [{"from": r.get("from_entity_id"), "to": r.get("to_entity_id"), "type": r.get("relationship_type")} for r in final_result.get("relationships", [])],
                    "reasoning": extraction_result.get("reasoning"),
                    "processing_time_ms": processing_time
                }
                
        except Exception as e:
            self.error_count += 1
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.error(f"❌ Semantic processing failed for job {job.job_id} in {processing_time:.1f}ms: {e}")
            raise
    
    async def _extract_with_pattern_detection(
        self, 
        user_id: str, 
        user_message: str, 
        assistant_response: str
    ) -> Dict[str, Any]:
        """Use LLM to intelligently decide what should go to semantic memory."""
        
        # Get existing context for pattern detection
        existing_context = await self._get_existing_graph_context(user_id)
        recent_episodes = await self._get_recent_episodes(user_id)
        
        # Enhanced LLM prompt for semantic memory curation
        pattern_detection_prompt = f"""
You are a knowledge graph curator for a personal AI assistant. Analyze this conversation and decide what should be stored as SEMANTIC KNOWLEDGE (permanent facts in knowledge graph) vs kept only as EPISODIC MEMORY (temporary experiences).

CONVERSATION:
User: {user_message}
Assistant: {assistant_response}

EXISTING SEMANTIC KNOWLEDGE:
{existing_context}

RECENT EPISODES (for pattern detection):
{recent_episodes}

DECISION RULES:
✅ SEMANTIC (store in knowledge graph):
- Personal identifiers: names, relationships, demographic facts
- Established preferences with clear evidence or explicit statements  
- Factual information about people, places, organizations
- Confirmed patterns (3+ mentions of same preference/behavior)

❌ EPISODIC ONLY (skip semantic storage):
- Single events/activities without preference patterns
- Unconfirmed preferences from isolated mentions
- Temporary states or one-off experiences
- Activities that don't reveal established patterns

EXAMPLES:
"My mom's name is Varda" → SEMANTIC (family relationship fact)
"I love Italian food" → SEMANTIC (explicit preference)  
"Had lunch at Italian restaurant downtown" → EPISODIC (single event, no pattern evidence)
Third mention of enjoying Italian cuisine → SEMANTIC (pattern confirmed)

Analyze the conversation and return JSON:
{{
    "skip_semantic": true/false,
    "entities": [
        {{
            "name": "entity_name",
            "type": "entity_type", 
            "confidence": 0.8,
            "properties": {{"key": "value"}}
        }}
    ],
    "relationships": [
        {{
            "from_entity": "entity_name",
            "to_entity": "entity_name",
            "relationship": "relationship_type",
            "confidence": 0.9
        }}
    ],
    "reasoning": "Explanation of decision - why semantic storage or episodic only"
}}

If skip_semantic is true, return empty entities and relationships arrays.
Focus on permanent facts and established patterns, not temporary experiences.
"""
        
        try:
            # Import here to avoid circular imports
            from ...memory.services.summarizer import ConversationSummarizer
            
            # Use existing summarizer for LLM calls
            summarizer = ConversationSummarizer()
            response = await summarizer._call_openai(pattern_detection_prompt, max_tokens=1000)
            
            # Parse JSON response
            import json
            import re
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    # Fallback to processing everything
                    logger.warning(f"Could not parse LLM pattern detection response, falling back to full extraction")
                    return await self._fallback_to_full_extraction(user_id, user_message, assistant_response)
            
            logger.info(f"🧠 Pattern detection decision: {data.get('reasoning', 'No reasoning provided')}")
            return data
            
        except Exception as e:
            logger.warning(f"Pattern detection failed: {e}, falling back to full extraction")
            return await self._fallback_to_full_extraction(user_id, user_message, assistant_response)
    
    async def _get_existing_graph_context(self, user_id: str) -> str:
        """Get existing knowledge graph context for pattern detection."""
        try:
            # Get sample of existing entities and relationships
            entities = await self.semantic_store.get_entities_for_user(user_id)
            relationships = await self.semantic_store.get_relationships_for_user(user_id)
            
            if not entities and not relationships:
                return "No existing semantic knowledge - this is the first analysis for this user."
            
            # Format for context
            context_parts = []
            
            if entities:
                context_parts.append("EXISTING ENTITIES:")
                for entity in entities[:5]:  # Show up to 5 entities
                    entity_type = entity.get('type', 'Unknown')
                    entity_name = entity.get('name', 'Unknown')
                    context_parts.append(f"- {entity_type}({entity_name})")
            
            if relationships:
                context_parts.append("\nEXISTING RELATIONSHIPS:")
                for rel in relationships[:5]:  # Show up to 5 relationships
                    from_entity = rel.get('from_entity', 'Unknown')
                    to_entity = rel.get('to_entity', 'Unknown')
                    rel_type = rel.get('relationship_type', 'UNKNOWN')
                    context_parts.append(f"- ({from_entity})-[{rel_type}]->({to_entity})")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.warning(f"Failed to get existing graph context: {e}")
            return "Unable to retrieve existing knowledge graph context."
    
    async def _get_recent_episodes(self, user_id: str) -> str:
        """Get recent episodic memories for pattern detection."""
        try:
            from ...memory.tiers.episodic import EpisodicMemoryStore
            
            async with EpisodicMemoryStore() as episodic_store:
                episodes = await episodic_store.get_recent_episodes(user_id, limit=5)
            
            if not episodes:
                return "No recent episodes found."
            
            context_parts = ["RECENT EPISODES:"]
            for i, episode in enumerate(episodes[:3], 1):  # Show 3 most recent
                content = episode.get('content', 'Unknown')[:100] + "..."  # Truncate for context
                timestamp = episode.get('timestamp', 'Unknown time')
                context_parts.append(f"{i}. ({timestamp[:10]}) {content}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.warning(f"Failed to get recent episodes: {e}")
            return "Unable to retrieve recent episodic context."
    
    async def _fallback_to_full_extraction(self, user_id: str, user_message: str, assistant_response: str) -> Dict[str, Any]:
        """Fallback to original extraction method if pattern detection fails."""
        try:
            extraction_result = await self.semantic_store.extract_and_store_entities(
                user_id=user_id,
                user_message=user_message,
                assistant_response=assistant_response
            )
            
            return {
                "skip_semantic": False,
                "entities": [{"name": e.name, "type": e.entity_type.value, "confidence": 0.7} for e in extraction_result.entities],
                "relationships": [{"from_entity": e.from_entity_id, "to_entity": e.to_entity_id, "relationship": e.relationship_type.value, "confidence": 0.7} for e in extraction_result.relationships],
                "reasoning": "Fallback to full extraction due to pattern detection failure"
            }
        except Exception as e:
            logger.error(f"Fallback extraction also failed: {e}")
            return {
                "skip_semantic": True,
                "entities": [],
                "relationships": [],
                "reasoning": f"Both pattern detection and fallback extraction failed: {e}"
            }
    
    async def _store_approved_entities_and_relationships(
        self, 
        entities: list[dict], 
        relationships: list[dict], 
        user_id: str
    ) -> Dict[str, Any]:
        """Store approved entities and relationships to knowledge graph."""
        
        from ...core.domain.semantic import EntityType, RelationshipType, GraphEntity, GraphRelationship
        
        stored_entities = []
        stored_relationships = []
        
        try:
            # Ensure user entity exists
            user_entity_id = await self.semantic_store.upsert_entity(
                user_id=user_id,
                entity_type=EntityType.USER,
                entity_name="User", 
                properties={"is_primary_user": True}
            )
            
            # Store entities
            for entity_data in entities:
                entity_type_str = entity_data.get("type", "ENTITY")
                
                # Map to enum
                try:
                    entity_type = EntityType(entity_type_str.upper())
                except ValueError:
                    entity_type = EntityType.ENTITY
                
                entity_id = await self.semantic_store.upsert_entity(
                    user_id=user_id,
                    entity_type=entity_type,
                    entity_name=entity_data["name"],
                    properties=entity_data.get("properties", {})
                )
                
                stored_entities.append({
                    "name": entity_data["name"],
                    "entity_type": entity_type.value,
                    "entity_id": entity_id
                })
            
            # Store relationships  
            for rel_data in relationships:
                relationship_type_str = rel_data.get("relationship", "RELATED_TO")
                
                # Map to enum
                try:
                    relationship_type = RelationshipType(relationship_type_str.upper())
                except ValueError:
                    relationship_type = RelationshipType.RELATED_TO
                
                # Find entity IDs
                from_name = rel_data["from_entity"]
                to_name = rel_data["to_entity"]
                
                # Use user entity ID if from_entity is "User"
                if from_name.lower() in ["user", "i", "me"]:
                    from_entity_id = user_entity_id
                else:
                    # Find stored entity by name
                    from_entity = next(
                        (e for e in stored_entities if e["name"] == from_name), 
                        None
                    )
                    if not from_entity:
                        # Create entity if not found
                        from_entity_id = await self.semantic_store.upsert_entity(
                            user_id=user_id,
                            entity_type=EntityType.ENTITY,
                            entity_name=from_name,
                            properties={}
                        )
                    else:
                        from_entity_id = from_entity["entity_id"]
                
                # Same for to_entity
                to_entity = next((e for e in stored_entities if e["name"] == to_name), None)
                if not to_entity:
                    to_entity_id = await self.semantic_store.upsert_entity(
                        user_id=user_id,
                        entity_type=EntityType.ENTITY,
                        entity_name=to_name,
                        properties={}
                    )
                else:
                    to_entity_id = to_entity["entity_id"]
                
                # Store relationship
                await self.semantic_store.upsert_relationship(
                    from_entity_id=from_entity_id,
                    to_entity_id=to_entity_id,
                    relationship_type=relationship_type,
                    weight=rel_data.get("confidence", 0.8),
                    decay_rate=0.01,
                    properties=rel_data.get("properties", {})
                )
                
                stored_relationships.append({
                    "from_entity_id": from_entity_id,
                    "to_entity_id": to_entity_id,
                    "relationship_type": relationship_type.value
                })
            
            return {
                "entities": stored_entities,
                "relationships": stored_relationships
            }
            
        except Exception as e:
            logger.error(f"Failed to store approved entities/relationships: {e}")
            raise

    def get_health(self) -> Dict[str, Any]:
        """Get processor health metrics."""
        success_rate = self.success_count / max(self.processing_count, 1)
        return {
            "processor": "semantic",
            "healthy": True,
            "total_processed": self.processing_count,
            "successful": self.success_count,
            "errors": self.error_count,
            "success_rate": success_rate,
            "store_initialized": self.semantic_store is not None
        }