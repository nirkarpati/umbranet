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
from ..services.summarizer import ConversationSummarizer

logger = logging.getLogger(__name__)


class EntityExtractionError(Exception):
    """Exception raised during entity extraction."""
    pass


class EntityExtractor:
    """Service for extracting entities and relationships from conversation text."""
    
    def __init__(self):
        """Initialize the entity extractor."""
        self.summarizer = ConversationSummarizer()
    
    async def __aenter__(self) -> "EntityExtractor":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        pass
    
    def _generate_entity_id(self, user_id: str, name: str, entity_type: EntityType) -> str:
        """Generate a consistent entity ID."""
        # Create deterministic ID based on user, name, and type
        key = f"{user_id}:{entity_type.value}:{name.lower().strip()}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))
    
    def _extract_entities_rule_based(
        self, 
        text: str, 
        user_id: str
    ) -> list[GraphEntity]:
        """Extract entities using rule-based pattern matching.
        
        This is a fallback when LLM extraction is not available.
        """
        entities = []
        text_lower = text.lower()
        
        # Pattern-based extraction
        patterns = {
            EntityType.PERSON: [
                r'\b(?:my|his|her)\s+(?:friend|colleague|boss|manager)\s+(\w+)',
                r'\b(\w+)(?:\s+\w+)?\s+(?:said|told|mentioned)',
                r'\bI\s+know\s+(\w+)',
            ],
            EntityType.LOCATION: [
                r'\bin\s+([A-Z][a-zA-Z\s]+)(?:\s|,|$)',
                r'\bfrom\s+([A-Z][a-zA-Z\s]+)(?:\s|,|$)',
                r'\bto\s+([A-Z][a-zA-Z\s]+)(?:\s|,|$)',
                r'\b(Seattle|Portland|San Francisco|New York|London|Tokyo)\b',
            ],
            EntityType.ORGANIZATION: [
                r'\b(Google|Microsoft|Apple|Amazon|Meta|Netflix|Tesla)\b',
                r'\bat\s+([A-Z][a-zA-Z\s]+(?:Inc|Corp|LLC|Ltd))',
                r'\bwork(?:ing)?\s+at\s+([A-Z][a-zA-Z\s]+)',
            ],
            EntityType.PROJECT: [
                r'\b(?:project|initiative)\s+([a-zA-Z][a-zA-Z0-9\s]+)',
                r'\bworking\s+on\s+([a-zA-Z][a-zA-Z0-9\s]+)',
            ]
        }
        
        for entity_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    
                    name = match.strip().title()
                    if len(name) > 2 and name not in [e.name for e in entities]:
                        entity_id = self._generate_entity_id(user_id, name, entity_type)
                        entities.append(GraphEntity(
                            entity_id=entity_id,
                            entity_type=entity_type,
                            name=name,
                            user_id=user_id,
                            properties={"extraction_method": "rule_based"}
                        ))
        
        return entities
    
    def _extract_preferences_rule_based(
        self, 
        text: str, 
        user_id: str
    ) -> list[GraphRelationship]:
        """Extract user preferences using rule-based matching."""
        relationships = []
        text_lower = text.lower()
        
        # Preference patterns
        like_patterns = [
            r'I\s+(?:love|like|enjoy|prefer)\s+([^.!?]+)',
            r'(?:love|like|enjoy)\s+([^.!?]+)',
            r'I\'m\s+a\s+fan\s+of\s+([^.!?]+)',
        ]
        
        dislike_patterns = [
            r'I\s+(?:hate|dislike|avoid)\s+([^.!?]+)',
            r'(?:don\'t|do not)\s+like\s+([^.!?]+)',
            r'I\'m\s+not\s+a\s+fan\s+of\s+([^.!?]+)',
        ]
        
        # Extract likes
        for pattern in like_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                preference_name = match.strip().title()
                if len(preference_name) > 2:
                    entity_id = self._generate_entity_id(
                        user_id, preference_name, EntityType.PREFERENCE
                    )
                    user_entity_id = self._generate_entity_id(
                        user_id, "User", EntityType.USER
                    )
                    
                    relationships.append(GraphRelationship(
                        from_entity_id=user_entity_id,
                        to_entity_id=entity_id,
                        relationship_type=RelationshipType.LIKES,
                        weight=0.7,  # Moderate confidence for rule-based
                        decay_rate=0.01,  # Slow decay
                        user_id=user_id,
                        properties={"extraction_method": "rule_based"}
                    ))
        
        # Extract dislikes
        for pattern in dislike_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                preference_name = match.strip().title()
                if len(preference_name) > 2:
                    entity_id = self._generate_entity_id(
                        user_id, preference_name, EntityType.PREFERENCE
                    )
                    user_entity_id = self._generate_entity_id(
                        user_id, "User", EntityType.USER
                    )
                    
                    relationships.append(GraphRelationship(
                        from_entity_id=user_entity_id,
                        to_entity_id=entity_id,
                        relationship_type=RelationshipType.DISLIKES,
                        weight=0.8,  # Higher confidence for negative preferences
                        decay_rate=0.005,  # Very slow decay for dislikes
                        user_id=user_id,
                        properties={"extraction_method": "rule_based"}
                    ))
        
        return relationships
    
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
                    entity_type = entity.get('type', 'Unknown')
                    entity_name = entity.get('name', 'Unknown')
                    properties = entity.get('properties', {})
                    prop_str = ", ".join([f"{k}:{v}" for k, v in properties.items() if k not in ['extraction_method', 'confidence']])
                    context_parts.append(f"- {entity_type}({entity_name}" + (f", {prop_str}" if prop_str else "") + ")")
            
            if relationships:
                sample_rels = relationships[:5]  # Show up to 5 relationships
                context_parts.append("\nEXISTING RELATIONSHIPS:")
                for rel in sample_rels:
                    from_entity = rel.get('from_entity', 'Unknown')
                    to_entity = rel.get('to_entity', 'Unknown')
                    rel_type = rel.get('relationship_type', 'UNKNOWN')
                    context_parts.append(f"- ({from_entity})-[{rel_type}]->({to_entity})")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.warning(f"Failed to get existing graph context for user {user_id}: {e}")
            return "Unable to retrieve existing graph context - proceeding with fresh extraction."
    
    async def _extract_entities_llm(
        self, 
        text: str, 
        user_id: str
    ) -> EntityExtractionResult:
        """Extract entities using LLM with context-aware structured prompting."""
        if not settings.openai_api_key:
            # Fall back to rule-based extraction
            entities = self._extract_entities_rule_based(text, user_id)
            relationships = self._extract_preferences_rule_based(text, user_id)
            return EntityExtractionResult(
                entities=entities,
                relationships=relationships,
                confidence=0.6
            )
        
        # Get existing graph context for consistency
        existing_context = await self._get_existing_graph_context(user_id)
        
        # Enhanced LLM-based extraction prompt with context awareness
        extraction_prompt = f"""
You are an expert knowledge graph extractor. Analyze the conversation and extract meaningful semantic entities and relationships.

CONVERSATION TEXT: {text}

EXISTING GRAPH CONTEXT (learn from these patterns for consistency):
{existing_context}

EXTRACTION GUIDELINES:
1. Create SPECIFIC and MEANINGFUL relationship types - avoid generic terms like "RELATED_TO"
2. Use descriptive entity types that capture the essence of what you're describing
3. Include relevant properties/attributes for entities (age, role, location, etc.)
4. Make relationships bidirectional when it makes sense
5. Use natural, human-readable relationship names

GOOD EXTRACTION EXAMPLES:
- "My mom Varda is 63" → Entities: Person(Varda, age:63), Person(User) + Relationship: (User)-[HAS_MOTHER]->(Varda)
- "I work at Google as a software engineer" → Entities: Person(User, role:"software engineer"), Organization(Google) + Relationship: (User)-[WORKS_AT]->(Google)
- "I love pizza and hate broccoli" → Entities: Food(Pizza), Food(Broccoli) + Relationships: (User)-[LOVES]->(Pizza), (User)-[HATES]->(Broccoli)
- "My dog Max is a Golden Retriever" → Entities: Pet(Max, breed:"Golden Retriever") + Relationship: (User)-[OWNS]->(Max)

Be creative but consistent with existing patterns. Entity types can be: Person, Location, Organization, Food, Pet, Hobby, Skill, Goal, Event, Product, Concept, etc.
Relationship types can be: HAS_MOTHER, WORKS_AT, LIVES_IN, LOVES, HATES, OWNS, FRIEND_OF, STUDIED_AT, BORN_IN, etc.

Return JSON format:
{{
    "entities": [
        {{
            "name": "entity_name",
            "type": "flexible_entity_type",
            "confidence": 0.7-1.0,
            "properties": {{"key": "value", "age": "number", "description": "string"}}
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

Focus on creating a rich, meaningful knowledge graph. Use "User" as the entity name for the speaker.
"""
        
        try:
            # Use the summarizer's OpenAI client for consistency
            response = await self.summarizer._call_openai(extraction_prompt, max_tokens=800)
            
            # Parse JSON response
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise EntityExtractionError("Could not parse LLM response as JSON")
            
            # Convert to domain objects
            entities = []
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
                        "state": EntityType.LOCATION
                    }
                    entity_type = type_mapping.get(entity_type_str.lower(), EntityType.ENTITY)
                
                entity_id = self._generate_entity_id(
                    user_id, entity_data["name"], entity_type
                )
                
                # Store original type string in properties for flexibility
                properties = {
                    **entity_data.get("properties", {}),
                    "extraction_method": "llm",
                    "confidence": entity_data.get("confidence", 0.7),
                    "original_type": entity_type_str  # Preserve LLM's original type
                }
                
                entities.append(GraphEntity(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    name=entity_data["name"],
                    user_id=user_id,
                    properties=properties
                ))
            
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
                        "owns": RelationshipType.OWNS
                    }
                    relationship_type = relationship_mapping.get(relationship_str.lower(), RelationshipType.RELATED_TO)
                
                # Find entity IDs
                from_name = rel_data["from_entity"]
                to_name = rel_data["to_entity"]
                
                # If from_entity is "User", use user entity ID
                if from_name.lower() in ["user", "i", "me"]:
                    from_entity_id = user_entity_id
                else:
                    # Find matching entity or create new one
                    from_entity = next(
                        (e for e in entities if e.name == from_name), 
                        None
                    )
                    if not from_entity:
                        # Create entity if not found
                        from_entity_id = self._generate_entity_id(
                            user_id, from_name, EntityType.ENTITY
                        )
                        entities.append(GraphEntity(
                            entity_id=from_entity_id,
                            entity_type=EntityType.ENTITY,
                            name=from_name,
                            user_id=user_id
                        ))
                    else:
                        from_entity_id = from_entity.entity_id
                
                # Same for to_entity
                to_entity = next((e for e in entities if e.name == to_name), None)
                if not to_entity:
                    to_entity_id = self._generate_entity_id(
                        user_id, to_name, EntityType.ENTITY
                    )
                    entities.append(GraphEntity(
                        entity_id=to_entity_id,
                        entity_type=EntityType.ENTITY,
                        name=to_name,
                        user_id=user_id
                    ))
                else:
                    to_entity_id = to_entity.entity_id
                
                confidence = rel_data.get("confidence", 0.7)
                
                # Preserve the original relationship string for flexibility
                relationship_properties = {
                    **rel_data.get("properties", {}),
                    "extraction_method": "llm",
                    "original_relationship": relationship_str  # Store LLM's original relationship name
                }
                
                relationships.append(GraphRelationship(
                    from_entity_id=from_entity_id,
                    to_entity_id=to_entity_id,
                    relationship_type=relationship_type,
                    weight=confidence,
                    decay_rate=0.01 if relationship_type in [
                        RelationshipType.HAS_ALLERGY, 
                        RelationshipType.DISLIKES,
                        RelationshipType.HATES
                    ] else 0.02,
                    user_id=user_id,
                    properties=relationship_properties
                ))
            
            return EntityExtractionResult(
                entities=entities,
                relationships=relationships,
                confidence=0.8  # Higher confidence for LLM extraction
            )
            
        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}, falling back to rules")
            # Fall back to rule-based extraction
            entities = self._extract_entities_rule_based(text, user_id)
            relationships = self._extract_preferences_rule_based(text, user_id)
            return EntityExtractionResult(
                entities=entities,
                relationships=relationships,
                confidence=0.6
            )
    
    async def extract_from_conversation(
        self, 
        user_message: str, 
        assistant_response: str, 
        user_id: str
    ) -> EntityExtractionResult:
        """Extract entities and relationships from a conversation turn.
        
        Args:
            user_message: User's message
            assistant_response: Assistant's response
            user_id: User identifier
            
        Returns:
            Extraction results with entities and relationships
        """
        # Combine both messages for context
        combined_text = f"User: {user_message}\nAssistant: {assistant_response}"
        
        # Focus primarily on user message for entity extraction
        return await self._extract_entities_llm(user_message, user_id)


# Convenience function
async def extract_entities(
    user_message: str, 
    assistant_response: str, 
    user_id: str
) -> EntityExtractionResult:
    """Extract entities from conversation text."""
    async with EntityExtractor() as extractor:
        return await extractor.extract_from_conversation(
            user_message, assistant_response, user_id
        )