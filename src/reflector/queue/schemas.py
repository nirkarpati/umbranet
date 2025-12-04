"""Schema definitions for memory reflection queue messages."""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
import json
import uuid

class ReflectionPriority(int, Enum):
    """Priority levels for memory reflection jobs."""
    LOW = 0      # System messages, routine conversations
    NORMAL = 1   # Regular user interactions  
    HIGH = 2     # Important conversations, explicit memory requests
    URGENT = 3   # Critical user data, error recovery

@dataclass
class MemoryReflectionJob:
    """Schema for memory reflection queue messages."""
    
    # Core identifiers
    job_id: str
    user_id: str
    session_id: str
    
    # Conversation data
    user_message: str
    assistant_response: str
    timestamp: datetime
    
    # Processing metadata
    priority: ReflectionPriority = ReflectionPriority.NORMAL
    metadata: Dict[str, Any] = None
    
    # Retry logic
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    @classmethod
    def from_interaction(
        cls, 
        user_id: str, 
        interaction: Dict[str, Any], 
        priority: ReflectionPriority = ReflectionPriority.NORMAL
    ) -> 'MemoryReflectionJob':
        """Create reflection job from interaction data."""
        return cls(
            job_id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=interaction.get('session_id', 'unknown'),
            user_message=interaction.get('content', ''),
            assistant_response=interaction.get('assistant_response', ''),
            timestamp=datetime.fromisoformat(interaction.get('timestamp', datetime.utcnow().isoformat())),
            priority=priority,
            metadata=interaction.get('metadata', {})
        )
    
    def to_json(self) -> str:
        """Serialize to JSON for queue transmission."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data['timestamp'] = self.timestamp.isoformat()
        data['created_at'] = self.created_at.isoformat()
        data['priority'] = self.priority.value  # Convert enum to int value
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MemoryReflectionJob':
        """Deserialize from JSON."""
        data = json.loads(json_str)
        # Convert ISO strings back to datetime objects
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['priority'] = ReflectionPriority(data['priority'])
        return cls(**data)
    
    def should_retry(self) -> bool:
        """Check if job should be retried on failure."""
        return self.retry_count < self.max_retries
    
    def increment_retry(self) -> 'MemoryReflectionJob':
        """Create new job instance with incremented retry count."""
        self.retry_count += 1
        return self