"""Entity extraction service for building semantic knowledge graphs."""

import json
import logging
import re
import uuid
from typing import Any

from ...core.config import settings
from ...core.domain.semantic import (
    EntityExtractionResult,
    EntityType,
    GraphEntity,
    GraphRelationship,
    RelationshipType,
)
from ...core.embeddings.base import EmbeddingProvider
from ...core.embeddings.provider_factory import get_embedding_provider
from ..services.summarizer import ConversationSummarizer

logger = logging.getLogger(__name__)


class EntityExtractionError(Exception):
    """Exception raised during entity extraction."""

    pass


class EntityExtractor:
    """Service for extracting entities and relationships from conversation text."""

    def __init__(self, embedding_provider: EmbeddingProvider | None = None):
        """Initialize the entity extractor.

        Args:
            embedding_provider: Optional embedding provider, uses default if None
        """
        self.summarizer = ConversationSummarizer()
        self.embedding_provider = embedding_provider or get_embedding_provider()

    async def __aenter__(self) -> "EntityExtractor":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        pass

    def _generate_entity_id(
        self, user_id: str, name: str, entity_type: EntityType
    ) -> str:
        """Generate a consistent entity ID."""
        # Create deterministic ID based on user, name, and type
        key = f"{user_id}:{entity_type.value}:{name.lower().strip()}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

    async def _generate_entity_embedding(self, entity: GraphEntity) -> list[float]:
        """Generate embedding for an entity using its name and properties.

        Args:
            entity: Entity to generate embedding for

        Returns:
            Embedding vector
        """
        # Create text representation: "name: properties"
        properties_text = ", ".join(
            [
                f"{k}: {v}"
                for k, v in entity.properties.items()
                if k not in ["extraction_method", "confidence", "original_type"]
            ]
        )

        embedding_text = (
            f"{entity.name}: {properties_text}" if properties_text else entity.name
        )

        try:
            embedding = await self.embedding_provider.embed_text(embedding_text)
            return embedding
        except Exception as e:
            logger.warning(
                f"Failed to generate embedding for entity {entity.name}: {e}"
            )
            # Return empty list if embedding generation fails
            return []

    async def _get_existing_graph_context(self, user_id: str) -> str:
        """Get a sample of existing entities and relationships for context consistency."""
        try:
            from ..tiers.semantic import SemanticMemoryStore

            # Get a sample of existing graph data
            async with SemanticMemoryStore() as semantic_store:
                entities = await semantic_store.get_entities_for_user(user_id)
                relationships = await semantic_store.get_relationships_for_user(user_id)

            if not entities and not relationships:
                return "No existing graph data - this is the first extraction for this user."

            # Format sample data for context
            context_parts = []

            if entities:
                sample_entities = entities[:5]  # Show up to 5 entities
                context_parts.append("EXISTING ENTITIES:")
                for entity in sample_entities:
                    entity_type = entity.get("type", "Unknown")
                    entity_name = entity.get("name", "Unknown")
                    properties = entity.get("properties", {})
                    prop_str = ", ".join(
                        [
                            f"{k}:{v}"
                            for k, v in properties.items()
                            if k not in ["extraction_method", "confidence"]
                        ]
                    )
                    context_parts.append(
                        f"- {entity_type}({entity_name}"
                        + (f", {prop_str}" if prop_str else "")
                        + ")"
                    )

            if relationships:
                sample_rels = relationships[:5]  # Show up to 5 relationships
                context_parts.append("\nEXISTING RELATIONSHIPS:")
                for rel in sample_rels:
                    from_entity = rel.get("from_entity", "Unknown")
                    to_entity = rel.get("to_entity", "Unknown")
                    rel_type = rel.get("relationship_type", "UNKNOWN")
                    context_parts.append(
                        f"- ({from_entity})-[{rel_type}]->({to_entity})"
                    )

            return "\n".join(context_parts)

        except Exception as e:
            logger.warning(
                f"Failed to get existing graph context for user {user_id}: {e}"
            )
            return "Unable to retrieve existing graph context - proceeding with fresh extraction."

    async def _extract_entities_llm(
        self, combined_text: str, user_id: str
    ) -> EntityExtractionResult:
        """Extract entities using LLM with context-aware structured prompting."""
        if not settings.openai_api_key:
            logger.error(
                "OpenAI API key not configured - cannot perform entity extraction"
            )
            raise EntityExtractionError(
                "Entity extraction requires OpenAI API key for embedding-based architecture. "
                "Rule-based fallbacks have been removed to maintain GraphRAG quality."
            )

        # Get existing graph context for consistency
        existing_context = await self._get_existing_graph_context(user_id)

        # Enhanced LLM-based extraction prompt with User-Property model
        extraction_prompt = f"""
You are an expert knowledge graph extractor. Analyze the conversation and extract meaningful semantic entities and relationships.

CONVERSATION TEXT: {combined_text}

EXISTING GRAPH CONTEXT (learn from these patterns for consistency):
{existing_context}

CRITICAL IDENTITY RULES:
1. IDENTITY RULE: The speaker is ALWAYS the entity named 'User'. Do NOT create a separate entity for the user's name.
2. NAME EXTRACTION RULE: If the user explicitly states their name (e.g., 'I am Nir', 'My name is...'), add this name as a property to the 'User' entity (e.g., User {{name: 'Nir'}}).
3. CONTEXT RULE: Use the provided CONVERSATION TEXT (which includes both User and Assistant) to resolve references, but prioritize User statements for new facts.

EXTRACTION GUIDELINES:
1. Create SPECIFIC and MEANINGFUL relationship types - avoid generic terms like "RELATED_TO"
2. Use descriptive entity types that capture the essence of what you're describing
3. Include relevant properties/attributes for entities (age, role, location, etc.)
4. Make relationships bidirectional when it makes sense
5. Use natural, human-readable relationship names
6. NEVER create separate entities for user names - always use the 'User' entity with name as a property

GOOD EXTRACTION EXAMPLES:
- "My mom Varda is 63" → Entities: Person(Varda, age:63), Person(User) + Relationship: (User)-[HAS_MOTHER]->(Varda)
- "I am Nir and I work at Google as a software engineer" → Entities: Person(User, name:"Nir", role:"software engineer"), Organization(Google) + Relationship: (User)-[WORKS_AT]->(Google)
- "I love pizza and hate broccoli" → Entities: Food(Pizza), Food(Broccoli), Person(User) + Relationships: (User)-[LOVES]->(Pizza), (User)-[HATES]->(Broccoli)
- "My dog Max is a Golden Retriever" → Entities: Pet(Max, breed:"Golden Retriever"), Person(User) + Relationship: (User)-[OWNS]->(Max)

Be creative but consistent with existing patterns. Entity types can be: Person, Location, Organization, Food, Pet, Hobby, Skill, Goal, Event, Product, Concept, etc.
Relationship types can be: HAS_MOTHER, WORKS_AT, LIVES_IN, LOVES, HATES, OWNS, FRIEND_OF, STUDIED_AT, BORN_IN, etc.

Return JSON format:
{{
    "entities": [
        {{
            "name": "entity_name",
            "type": "flexible_entity_type", 
            "confidence": 0.7-1.0,
            "properties": {{"key": "value", "age": "number", "description": "string", "name": "user_actual_name" (ONLY for User entity when explicitly stated)}}
        }}
    ],
    "relationships": [
        {{
            "from_entity": "entity_name",
            "to_entity": "entity_name", 
            "relationship": "meaningful_relationship_name",
            "confidence": 0.7-1.0,
            "properties": {{"since": "date", "strength": "string", "notes": "string"}}
        }}
    ]
}}

Focus on creating a rich, meaningful knowledge graph. ALWAYS use "User" as the entity name for the speaker, with their actual name stored in properties when explicitly mentioned.
"""

        try:
            # Use the summarizer's OpenAI client for consistency
            response = await self.summarizer._call_openai(
                extraction_prompt, max_tokens=800
            )

            # Parse JSON response
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise EntityExtractionError("Could not parse LLM response as JSON")

            # Convert to domain objects
            entities: list[GraphEntity] = []
            relationships = []

            # Process entities with flexible typing
            for entity_data in data.get("entities", []):
                entity_type_str = entity_data.get("type", "Entity")

                # Try to match to existing EntityType enum, otherwise use ENTITY as fallback
                try:
                    entity_type = EntityType(entity_type_str)
                except ValueError:
                    # For flexible entity types, try common mappings
                    type_mapping = {
                        "pet": EntityType.ENTITY,
                        "hobby": EntityType.CONCEPT,
                        "food": EntityType.PRODUCT,
                        "animal": EntityType.ENTITY,
                        "vehicle": EntityType.PRODUCT,
                        "book": EntityType.PRODUCT,
                        "movie": EntityType.PRODUCT,
                        "song": EntityType.PRODUCT,
                        "company": EntityType.ORGANIZATION,
                        "school": EntityType.ORGANIZATION,
                        "university": EntityType.ORGANIZATION,
                        "city": EntityType.LOCATION,
                        "country": EntityType.LOCATION,
                        "state": EntityType.LOCATION,
                    }
                    entity_type = type_mapping.get(
                        entity_type_str.lower(), EntityType.ENTITY
                    )

                entity_id = self._generate_entity_id(
                    user_id, entity_data["name"], entity_type
                )

                # Store original type string in properties for flexibility
                properties = {
                    **entity_data.get("properties", {}),
                    "extraction_method": "llm",
                    "confidence": entity_data.get("confidence", 0.7),
                    "original_type": entity_type_str,  # Preserve LLM's original type
                }

                entity = GraphEntity(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    name=entity_data["name"],
                    user_id=user_id,
                    properties=properties,
                )
                entities.append(entity)

            # Process relationships
            user_entity_id = self._generate_entity_id(user_id, "User", EntityType.USER)

            for rel_data in data.get("relationships", []):
                relationship_str = rel_data.get("relationship", "RELATED_TO")

                # Try to match to existing RelationshipType enum, otherwise use RELATED_TO as fallback
                try:
                    relationship_type = RelationshipType(relationship_str)
                except ValueError:
                    # For flexible relationship types, try common mappings
                    relationship_mapping = {
                        # Family
                        "has_mother": RelationshipType.HAS_MOTHER,
                        "has_father": RelationshipType.HAS_FATHER,
                        "has_child": RelationshipType.HAS_CHILD,
                        "has_sibling": RelationshipType.HAS_SIBLING,
                        # Work
                        "works_at": RelationshipType.WORKS_AT,
                        "colleague_of": RelationshipType.COLLEAGUE_OF,
                        # Preferences
                        "likes": RelationshipType.LIKES,
                        "loves": RelationshipType.LOVES,
                        "hates": RelationshipType.HATES,
                        "dislikes": RelationshipType.DISLIKES,
                        # Locations
                        "lives_in": RelationshipType.LIVES_IN,
                        "born_in": RelationshipType.BORN_IN,
                        # Social
                        "friend_of": RelationshipType.FRIEND_OF,
                        "knows": RelationshipType.KNOWS,
                        # Skills
                        "expert_in": RelationshipType.EXPERT_IN,
                        "has_skill": RelationshipType.HAS_SKILL,
                        # Ownership
                        "owns": RelationshipType.OWNS,
                    }
                    relationship_type = relationship_mapping.get(
                        relationship_str.lower(), RelationshipType.RELATED_TO
                    )

                # Find entity IDs
                from_name = rel_data["from_entity"]
                to_name = rel_data["to_entity"]

                # If from_entity is "User", use user entity ID
                if from_name.lower() in ["user", "i", "me"]:
                    from_entity_id = user_entity_id
                else:
                    # Find matching entity or create new one
                    from_entity = next(
                        (e for e in entities if e.name == from_name), None
                    )
                    if not from_entity:
                        # Create entity if not found
                        from_entity_id = self._generate_entity_id(
                            user_id, from_name, EntityType.ENTITY
                        )
                        new_entity = GraphEntity(
                            entity_id=from_entity_id,
                            entity_type=EntityType.ENTITY,
                            name=from_name,
                            user_id=user_id,
                        )
                        entities.append(new_entity)
                    else:
                        from_entity_id = from_entity.entity_id

                # Same for to_entity
                to_entity = next((e for e in entities if e.name == to_name), None)
                if not to_entity:
                    to_entity_id = self._generate_entity_id(
                        user_id, to_name, EntityType.ENTITY
                    )
                    new_entity = GraphEntity(
                        entity_id=to_entity_id,
                        entity_type=EntityType.ENTITY,
                        name=to_name,
                        user_id=user_id,
                    )
                    entities.append(new_entity)
                else:
                    to_entity_id = to_entity.entity_id

                confidence = rel_data.get("confidence", 0.7)

                # Preserve the original relationship string for flexibility
                relationship_properties = {
                    **rel_data.get("properties", {}),
                    "extraction_method": "llm",
                    "original_relationship": relationship_str,  # Store LLM's original relationship name
                }

                relationships.append(
                    GraphRelationship(
                        from_entity_id=from_entity_id,
                        to_entity_id=to_entity_id,
                        relationship_type=relationship_type,
                        weight=confidence,
                        decay_rate=0.01
                        if relationship_type
                        in [
                            RelationshipType.HAS_ALLERGY,
                            RelationshipType.DISLIKES,
                            RelationshipType.HATES,
                        ]
                        else 0.02,
                        user_id=user_id,
                        properties=relationship_properties,
                    )
                )

            # Generate embeddings for all entities
            for entity in entities:
                entity.embedding = await self._generate_entity_embedding(entity)

            return EntityExtractionResult(
                entities=entities,
                relationships=relationships,
                confidence=0.8,  # Higher confidence for LLM extraction
            )

        except Exception as e:
            logger.error(f"LLM entity extraction failed: {e}")
            raise EntityExtractionError(
                f"Entity extraction failed: {str(e)}. "
                "Rule-based fallbacks have been removed to maintain GraphRAG quality."
            ) from e

    async def extract_from_conversation(
        self, user_message: str, assistant_response: str, user_id: str
    ) -> EntityExtractionResult:
        """Extract entities and relationships from a conversation turn.

        Args:
            user_message: User's message
            assistant_response: Assistant's response
            user_id: User identifier

        Returns:
            Extraction results with entities and relationships
        """
        # Combine both messages for full context extraction
        combined_text = f"User: {user_message}\nAssistant: {assistant_response}"

        # Pass combined text to enable full conversation context
        return await self._extract_entities_llm(combined_text, user_id)


# Convenience function
async def extract_entities(
    user_message: str,
    assistant_response: str,
    user_id: str,
    embedding_provider: EmbeddingProvider | None = None,
) -> EntityExtractionResult:
    """Extract entities from conversation text.

    Args:
        user_message: User's message
        assistant_response: Assistant's response
        user_id: User identifier
        embedding_provider: Optional embedding provider
    """
    async with EntityExtractor(embedding_provider=embedding_provider) as extractor:
        return await extractor.extract_from_conversation(
            user_message, assistant_response, user_id
        )
