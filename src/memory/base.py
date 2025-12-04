"""Memory tier abstractions and base classes for the RAG++ memory hierarchy."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MemoryQuery(BaseModel):
    """Standardized memory query format across all tiers."""
    user_id: str = Field(..., description="User identifier")
    query_text: Optional[str] = Field(None, description="Natural language query")
    entities: Optional[List[str]] = Field(None, description="Entity filter list")
    time_range: Optional[tuple[datetime, datetime]] = Field(None, description="Time range filter")
    limit: int = Field(default=10, description="Maximum results to return")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity for retrieval")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional query metadata")


class MemoryTier(ABC):
    """Base class for all memory tier implementations."""
    
    @abstractmethod
    async def store(self, user_id: str, data: Dict[str, Any]) -> str:
        """Store data and return unique identifier.
        
        Args:
            user_id: User identifier
            data: Data to store
            
        Returns:
            Unique identifier for stored data
        """
        pass
    
    @abstractmethod 
    async def retrieve(self, user_id: str, query: MemoryQuery) -> List[Dict[str, Any]]:
        """Retrieve relevant memories based on query.
        
        Args:
            user_id: User identifier
            query: Memory query parameters
            
        Returns:
            List of relevant memory items
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if memory tier is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the memory tier (connections, schema, etc.)."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources (connections, locks, etc.)."""
        pass


class MemoryHealthMetrics(BaseModel):
    """Health metrics for memory operations."""
    
    total_retrievals: int = Field(default=0, description="Total retrieval operations")
    successful_retrievals: int = Field(default=0, description="Successful retrievals")
    failed_retrievals: int = Field(default=0, description="Failed retrievals")
    average_latency_ms: float = Field(default=0.0, description="Average retrieval latency")
    last_health_check: Optional[datetime] = Field(None, description="Last health check timestamp")
    
    def record_retrieval(self, latency_ms: float, success: bool = True) -> None:
        """Record a retrieval operation."""
        self.total_retrievals += 1
        
        if success:
            self.successful_retrievals += 1
        else:
            self.failed_retrievals += 1
        
        # Update rolling average
        total_latency = self.average_latency_ms * (self.total_retrievals - 1)
        self.average_latency_ms = (total_latency + latency_ms) / self.total_retrievals
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_retrievals == 0:
            return 0.0
        return (self.successful_retrievals / self.total_retrievals) * 100.0
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as percentage."""
        return 100.0 - self.success_rate


class MemoryConfig(BaseModel):
    """Configuration for memory manager and tiers."""
    
    # Redis Configuration (Tier 1)
    redis_url: str = Field(..., description="Redis connection URL")
    max_tokens: int = Field(default=2000, description="Maximum tokens in short-term buffer")
    
    # PostgreSQL Configuration (Tiers 2 & 4)
    postgres_url: str = Field(..., description="PostgreSQL connection URL") 
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence transformer model")
    
    # Neo4j Configuration (Tier 3)
    neo4j_uri: str = Field(..., description="Neo4j connection URI")
    neo4j_user: str = Field(..., description="Neo4j username")
    neo4j_password: str = Field(..., description="Neo4j password")
    
    # Performance Configuration
    default_retrieval_timeout_ms: int = Field(default=500, description="Default retrieval timeout")
    health_check_interval_seconds: int = Field(default=60, description="Health check interval")
    enable_health_monitoring: bool = Field(default=True, description="Enable health monitoring")
    
    # Feature Flags
    enable_oer_learning: bool = Field(default=False, description="Enable O-E-R learning loop")
    enable_semantic_extraction: bool = Field(default=True, description="Enable semantic entity extraction")