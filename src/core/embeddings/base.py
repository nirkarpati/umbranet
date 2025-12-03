"""Base embedding provider interface using strategy pattern."""

from abc import ABC, abstractmethod
from typing import Any


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of dimensions in the embedding vectors."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name/identifier of the embedding model."""
        pass
    
    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text string.
        
        Args:
            text: The text to embed
            
        Returns:
            List of float values representing the embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple text strings.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors, one for each input text
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass
    
    async def __aenter__(self) -> "EmbeddingProvider":
        """Async context manager entry."""
        return self
    
    @abstractmethod
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        pass


class EmbeddingError(Exception):
    """Exception raised for embedding-related errors."""
    pass