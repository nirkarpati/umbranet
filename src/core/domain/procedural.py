"""Procedural memory domain models for user profiles and behavioral instructions."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, validator


class ProfileCategory(str, Enum):
    """Categories for user profile data."""
    
    # Core identity
    PERSONAL = "personal"
    PREFERENCES = "preferences"
    SETTINGS = "settings"
    
    # Location and timezone
    LOCATION = "location"
    TIMEZONE = "timezone"
    
    # Communication style
    COMMUNICATION = "communication"
    LANGUAGE = "language"
    
    # Work and professional
    PROFESSIONAL = "professional"
    
    # Health and dietary
    HEALTH = "health"
    DIETARY = "dietary"
    
    # Technical preferences
    TECHNICAL = "technical"
    
    # Custom categories
    CUSTOM = "custom"


class InstructionCategory(str, Enum):
    """Categories for behavioral instructions."""
    
    # Communication behavior
    COMMUNICATION_STYLE = "communication_style"
    RESPONSE_FORMAT = "response_format"
    
    # Task execution
    TASK_EXECUTION = "task_execution"
    TOOL_PREFERENCES = "tool_preferences"
    
    # Decision making
    DECISION_CRITERIA = "decision_criteria"
    RISK_TOLERANCE = "risk_tolerance"
    
    # Scheduling and time
    SCHEDULING = "scheduling"
    TIME_MANAGEMENT = "time_management"
    
    # Travel and booking
    TRAVEL_BOOKING = "travel_booking"
    ACCOMMODATION = "accommodation"
    
    # Shopping and purchases
    SHOPPING = "shopping"
    FINANCIAL = "financial"
    
    # Content and entertainment
    CONTENT_CURATION = "content_curation"
    ENTERTAINMENT = "entertainment"
    
    # Learning and development
    LEARNING = "learning"
    SKILL_DEVELOPMENT = "skill_development"
    
    # General behavioral rules
    GENERAL = "general"


class ProfileEntry(BaseModel):
    """A single profile entry (key-value pair)."""
    
    user_id: str = Field(..., description="User this profile entry belongs to")
    category: ProfileCategory = Field(..., description="Category of the profile data")
    key: str = Field(..., description="Profile key (e.g., 'name', 'timezone', 'home_address')")
    value: str = Field(..., description="Profile value")
    value_type: str = Field(
        default="string",
        description="Type of the value (string, number, boolean, json)"
    )
    is_sensitive: bool = Field(
        default=False,
        description="Whether this data is sensitive and requires special handling"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this entry was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this entry was last updated"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about this profile entry"
    )
    
    @validator('key')
    def validate_key(cls, v: str) -> str:
        """Ensure key is not empty and uses valid format."""
        if not v or not v.strip():
            raise ValueError("Profile key cannot be empty")
        # Convert to lowercase with underscores for consistency
        return v.strip().lower().replace(' ', '_').replace('-', '_')
    
    @validator('value')
    def validate_value(cls, v: str) -> str:
        """Ensure value is not empty."""
        if not v or not v.strip():
            raise ValueError("Profile value cannot be empty")
        return v.strip()


class BehavioralInstruction(BaseModel):
    """A behavioral instruction or rule for the AI system."""
    
    user_id: str = Field(..., description="User this instruction belongs to")
    instruction_id: str = Field(..., description="Unique identifier for this instruction")
    category: InstructionCategory = Field(..., description="Category of behavioral instruction")
    title: str = Field(..., description="Short title describing the instruction")
    instruction: str = Field(
        ...,
        description="The actual instruction text for the AI system"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in this instruction (0.0 to 1.0)"
    )
    priority: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Priority level (1=lowest, 10=highest)"
    )
    is_active: bool = Field(
        default=True,
        description="Whether this instruction is currently active"
    )
    embedding: list[float] | None = Field(
        None,
        description="Vector embedding of the instruction for similarity search"
    )
    examples: list[str] = Field(
        default_factory=list,
        description="Example scenarios where this instruction applies"
    )
    exceptions: list[str] = Field(
        default_factory=list,
        description="Exceptions or conditions where this instruction doesn't apply"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this instruction was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this instruction was last updated"
    )
    last_used: datetime | None = Field(
        None,
        description="When this instruction was last used/referenced"
    )
    usage_count: int = Field(
        default=0,
        description="How many times this instruction has been used"
    )
    source: str = Field(
        default="user_explicit",
        description="Source of this instruction (user_explicit, inferred, system)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about this instruction"
    )
    
    @validator('instruction')
    def validate_instruction(cls, v: str) -> str:
        """Ensure instruction is not empty and meaningful."""
        if not v or not v.strip():
            raise ValueError("Instruction text cannot be empty")
        if len(v.strip()) < 10:
            raise ValueError("Instruction must be at least 10 characters long")
        return v.strip()
    
    @validator('title')
    def validate_title(cls, v: str) -> str:
        """Ensure title is not empty."""
        if not v or not v.strip():
            raise ValueError("Instruction title cannot be empty")
        return v.strip()


class UserProfile(BaseModel):
    """Complete user profile composed of multiple profile entries."""
    
    user_id: str = Field(..., description="User identifier")
    entries: dict[str, ProfileEntry] = Field(
        default_factory=dict,
        description="Profile entries keyed by 'category.key'"
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the profile was last updated"
    )
    
    def get_value(self, category: ProfileCategory, key: str, default: str | None = None) -> str | None:
        """Get a profile value by category and key."""
        entry_key = f"{category.value}.{key}"
        entry = self.entries.get(entry_key)
        return entry.value if entry else default
    
    def set_value(
        self, 
        category: ProfileCategory, 
        key: str, 
        value: str,
        value_type: str = "string",
        is_sensitive: bool = False
    ) -> None:
        """Set a profile value."""
        entry_key = f"{category.value}.{key}"
        
        if entry_key in self.entries:
            # Update existing entry
            self.entries[entry_key].value = value
            self.entries[entry_key].value_type = value_type
            self.entries[entry_key].updated_at = datetime.utcnow()
        else:
            # Create new entry
            self.entries[entry_key] = ProfileEntry(
                user_id=self.user_id,
                category=category,
                key=key,
                value=value,
                value_type=value_type,
                is_sensitive=is_sensitive
            )
        
        self.last_updated = datetime.utcnow()
    
    def remove_value(self, category: ProfileCategory, key: str) -> bool:
        """Remove a profile value. Returns True if removed, False if not found."""
        entry_key = f"{category.value}.{key}"
        if entry_key in self.entries:
            del self.entries[entry_key]
            self.last_updated = datetime.utcnow()
            return True
        return False
    
    def get_by_category(self, category: ProfileCategory) -> dict[str, str]:
        """Get all values for a specific category."""
        prefix = f"{category.value}."
        return {
            key.replace(prefix, ""): entry.value
            for key, entry in self.entries.items()
            if key.startswith(prefix)
        }


class InstructionQuery(BaseModel):
    """Query for retrieving relevant behavioral instructions."""
    
    user_id: str = Field(..., description="User to query for")
    query_text: str = Field(..., description="Query text to find relevant instructions")
    categories: list[InstructionCategory] = Field(
        default_factory=list,
        description="Specific categories to search in (empty = all categories)"
    )
    min_confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold"
    )
    min_priority: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Minimum priority level"
    )
    include_inactive: bool = Field(
        default=False,
        description="Whether to include inactive instructions"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of instructions to return"
    )


class RelevantInstruction(BaseModel):
    """A behavioral instruction with relevance scoring."""
    
    instruction: BehavioralInstruction = Field(..., description="The instruction")
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Relevance score for the query"
    )
    match_reason: str = Field(
        default="",
        description="Explanation of why this instruction matches"
    )


class ProceduralMemoryStats(BaseModel):
    """Statistics about a user's procedural memory."""
    
    user_id: str = Field(..., description="User identifier")
    total_profile_entries: int = Field(default=0, description="Number of profile entries")
    total_instructions: int = Field(default=0, description="Number of behavioral instructions")
    profile_categories: dict[str, int] = Field(
        default_factory=dict,
        description="Count of profile entries by category"
    )
    instruction_categories: dict[str, int] = Field(
        default_factory=dict,
        description="Count of instructions by category"
    )
    active_instructions: int = Field(default=0, description="Number of active instructions")
    average_instruction_confidence: float = Field(
        default=0.0,
        description="Average confidence of all instructions"
    )
    last_profile_update: datetime | None = Field(
        None,
        description="When profile was last updated"
    )
    last_instruction_update: datetime | None = Field(
        None,
        description="When instructions were last updated"
    )