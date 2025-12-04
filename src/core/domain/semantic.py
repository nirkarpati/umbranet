"""Semantic memory domain models for weighted probabilistic knowledge graphs."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, validator


class EntityType(str, Enum):
    """Core entity types for the semantic graph."""
    
    # Fixed core nodes (always present)
    USER = "User"
    SYSTEM_AGENT = "SystemAgent"
    
    # Dynamic entity types (learned from interactions)
    PERSON = "Person"
    LOCATION = "Location"
    PROJECT = "Project"
    EVENT = "Event"
    ORGANIZATION = "Organization"
    PRODUCT = "Product"
    CONCEPT = "Concept"
    PREFERENCE = "Preference"
    SKILL = "Skill"
    GOAL = "Goal"
    TASK = "Task"
    FOOD = "Food"
    
    # Flexible catch-all for new entity types
    ENTITY = "Entity"


class RelationshipType(str, Enum):
    """Relationship types with semantic meaning."""
    
    # Family relationships
    HAS_MOTHER = "HAS_MOTHER"
    HAS_FATHER = "HAS_FATHER"
    HAS_SIBLING = "HAS_SIBLING"
    HAS_CHILD = "HAS_CHILD"
    HAS_SPOUSE = "HAS_SPOUSE"
    HAS_PARTNER = "HAS_PARTNER"
    
    # Work relationships
    WORKS_AT = "WORKS_AT"
    MANAGES = "MANAGES"
    REPORTS_TO = "REPORTS_TO"
    COLLEAGUE_OF = "COLLEAGUE_OF"
    
    # Location relationships
    LIVES_IN = "LIVES_IN"
    BORN_IN = "BORN_IN"
    VISITED = "VISITED"
    WORKS_IN = "WORKS_IN"
    STUDIED_IN = "STUDIED_IN"
    
    # Preferences and interests
    LIKES = "LIKES"
    DISLIKES = "DISLIKES"
    PREFERS = "PREFERS"
    ENJOYS = "ENJOYS"
    AVOIDS = "AVOIDS"
    LOVES = "LOVES"
    HATES = "HATES"
    
    # Skills and knowledge
    HAS_SKILL = "HAS_SKILL"
    KNOWS_ABOUT = "KNOWS_ABOUT"
    EXPERT_IN = "EXPERT_IN"
    LEARNING = "LEARNING"
    TAUGHT_BY = "TAUGHT_BY"
    
    # Social connections
    KNOWS = "KNOWS"
    FRIEND_OF = "FRIEND_OF"
    MENTOR_OF = "MENTOR_OF"
    STUDENT_OF = "STUDENT_OF"
    
    # Ownership and possession
    OWNS = "OWNS"
    HAS_GOAL = "HAS_GOAL"
    HAS_ALLERGY = "HAS_ALLERGY"
    
    # Entity relationships
    RELATED_TO = "RELATED_TO"
    PART_OF = "PART_OF"
    SIMILAR_TO = "SIMILAR_TO"
    DEPENDS_ON = "DEPENDS_ON"
    LOCATED_IN = "LOCATED_IN"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    
    # Temporal relationships
    HAPPENED_ON = "HAPPENED_ON"
    SCHEDULED_FOR = "SCHEDULED_FOR"
    
    # System relationships
    MENTIONED_IN = "MENTIONED_IN"
    USED_FOR = "USED_FOR"


class GraphEntity(BaseModel):
    """Represents a node in the semantic knowledge graph."""
    
    entity_id: str = Field(..., description="Unique identifier for the entity")
    entity_type: EntityType = Field(..., description="Type of the entity")
    name: str = Field(..., description="Display name of the entity")
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional properties of the entity"
    )
    user_id: str = Field(..., description="User this entity belongs to")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this entity was first created"
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow, 
        description="When this entity was last updated"
    )
    
    @validator('entity_id', 'name')
    def validate_non_empty(cls, v: str) -> str:
        """Ensure ID and name are not empty."""
        if not v or not v.strip():
            raise ValueError("Entity ID and name cannot be empty")
        return v.strip()


class GraphRelationship(BaseModel):
    """Represents a weighted relationship between entities."""
    
    from_entity_id: str = Field(..., description="Source entity ID")
    to_entity_id: str = Field(..., description="Target entity ID")
    relationship_type: RelationshipType = Field(..., description="Type of relationship")
    weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence weight (0.0 to 1.0)"
    )
    decay_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How fast this relationship decays over time"
    )
    last_verified: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this relationship was last verified"
    )
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional relationship properties"
    )
    user_id: str = Field(..., description="User this relationship belongs to")
    source: str = Field(
        default="user_interaction",
        description="Source of this relationship (e.g., 'user_interaction', 'inference')"
    )


class EntityExtractionResult(BaseModel):
    """Result of extracting entities from text."""
    
    entities: list[GraphEntity] = Field(
        default_factory=list,
        description="Extracted entities"
    )
    relationships: list[GraphRelationship] = Field(
        default_factory=list,
        description="Extracted relationships"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the extraction"
    )


class GraphQuery(BaseModel):
    """Query for semantic graph traversal."""
    
    user_id: str = Field(..., description="User to query for")
    entity_names: list[str] = Field(
        default_factory=list,
        description="Entity names to search for"
    )
    entity_types: list[EntityType] = Field(
        default_factory=list,
        description="Entity types to filter by"
    )
    relationship_types: list[RelationshipType] = Field(
        default_factory=list,
        description="Relationship types to follow"
    )
    max_depth: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum traversal depth"
    )
    min_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum relationship weight to consider"
    )


class SemanticFact(BaseModel):
    """A semantic fact retrieved from the knowledge graph."""
    
    subject: GraphEntity = Field(..., description="Subject entity")
    relationship: GraphRelationship = Field(..., description="Relationship")
    object: GraphEntity = Field(..., description="Object entity")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in this fact"
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context about this fact"
    )


class GraphStats(BaseModel):
    """Statistics about a user's semantic graph."""
    
    total_entities: int = Field(default=0, description="Total number of entities")
    total_relationships: int = Field(default=0, description="Total number of relationships")
    entity_types_count: dict[str, int] = Field(
        default_factory=dict,
        description="Count of entities by type"
    )
    relationship_types_count: dict[str, int] = Field(
        default_factory=dict,
        description="Count of relationships by type"
    )
    average_weight: float = Field(
        default=0.0,
        description="Average relationship weight"
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="When these stats were computed"
    )