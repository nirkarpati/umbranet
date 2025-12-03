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
    
    async def _extract_entities_llm(
        self, 
        text: str, 
        user_id: str
    ) -> EntityExtractionResult:
        """Extract entities using LLM with structured prompting."""
        if not settings.openai_api_key:
            # Fall back to rule-based extraction
            entities = self._extract_entities_rule_based(text, user_id)
            relationships = self._extract_preferences_rule_based(text, user_id)
            return EntityExtractionResult(
                entities=entities,
                relationships=relationships,
                confidence=0.6
            )
        
        # LLM-based extraction prompt
        extraction_prompt = f"""
Analyze the following conversation text and extract semantic entities and relationships for knowledge graph construction.

TEXT: {text}

Extract:
1. ENTITIES: People, places, organizations, concepts, preferences, skills, goals
2. RELATIONSHIPS: How entities connect (likes, works_at, lives_in, knows, etc.)

For the user (speaker), create relationships showing their preferences, connections, and attributes.

Return JSON format:
{{
    "entities": [
        {{
            "name": "entity_name",
            "type": "Person|Location|Organization|Concept|Preference|Skill|Goal|Product",
            "confidence": 0.0-1.0,
            "properties": {{"key": "value"}}
        }}
    ],
    "relationships": [
        {{
            "from_entity": "entity_name",
            "to_entity": "entity_name", 
            "relationship": "LIKES|DISLIKES|WORKS_AT|LIVES_IN|KNOWS|HAS_SKILL|etc",
            "confidence": 0.0-1.0,
            "properties": {{"key": "value"}}
        }}
    ]
}}

Focus on factual, actionable information. Avoid speculation.
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
            
            # Process entities
            for entity_data in data.get("entities", []):
                try:
                    entity_type = EntityType(entity_data["type"])
                except ValueError:
                    entity_type = EntityType.ENTITY  # Default fallback
                
                entity_id = self._generate_entity_id(
                    user_id, entity_data["name"], entity_type
                )
                
                entities.append(GraphEntity(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    name=entity_data["name"],
                    user_id=user_id,
                    properties={
                        **entity_data.get("properties", {}),
                        "extraction_method": "llm",
                        "confidence": entity_data.get("confidence", 0.7)
                    }
                ))
            
            # Process relationships
            user_entity_id = self._generate_entity_id(user_id, "User", EntityType.USER)
            
            for rel_data in data.get("relationships", []):
                try:
                    relationship_type = RelationshipType(rel_data["relationship"])
                except ValueError:
                    relationship_type = RelationshipType.RELATED_TO  # Default fallback
                
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
                relationships.append(GraphRelationship(
                    from_entity_id=from_entity_id,
                    to_entity_id=to_entity_id,
                    relationship_type=relationship_type,
                    weight=confidence,
                    decay_rate=0.01 if relationship_type in [
                        RelationshipType.HAS_ALLERGY, 
                        RelationshipType.DISLIKES
                    ] else 0.02,
                    user_id=user_id,
                    properties={
                        **rel_data.get("properties", {}),
                        "extraction_method": "llm"
                    }
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