"""Local sentence-transformers embedding provider implementation."""

import logging
from typing import Any

from .base import EmbeddingError, EmbeddingProvider

logger = logging.getLogger(__name__)


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize local embedding provider.
        
        Args:
            model_name: Sentence-transformers model to use
        """
        self.model_name_str = model_name
        self.model: Any = None
        self._dimension_map = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "all-distilroberta-v1": 768,
        }
    
    @property
    def dimension(self) -> int:
        """Number of dimensions in the embedding vectors."""
        return self._dimension_map.get(self.model_name_str, 384)
    
    @property
    def model_name(self) -> str:
        """Name/identifier of the embedding model."""
        return f"local:{self.model_name_str}"
    
    async def __aenter__(self) -> "LocalEmbeddingProvider":
        """Async context manager entry."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading sentence-transformers model: {self.model_name_str}")
            self.model = SentenceTransformer(self.model_name_str)
            logger.info("Model loaded successfully")
            return self
        except ImportError as e:
            raise EmbeddingError(
                "sentence-transformers not available. "
                "Install with: pip install sentence-transformers"
            ) from e
        except Exception as e:
            raise EmbeddingError(
                f"Failed to load model {self.model_name_str}: {str(e)}"
            ) from e
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        # sentence-transformers models don't need explicit cleanup
        self.model = None
    
    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text string."""
        if not self.model:
            raise EmbeddingError("Model not initialized - use async context manager")
        
        if not text or not text.strip():
            return [0.0] * self.dimension
        
        try:
            # sentence-transformers encode returns numpy array
            embedding = self.model.encode([text.strip()])[0]
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Local embedding failed: {str(e)}")
            raise EmbeddingError(f"Local embedding error: {str(e)}") from e
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple text strings."""
        if not self.model:
            raise EmbeddingError("Model not initialized - use async context manager")
        
        if not texts:
            return []
        
        # Filter and clean texts
        clean_texts = [text.strip() if text and text.strip() else "" for text in texts]
        
        try:
            # sentence-transformers can handle batch processing efficiently
            embeddings = self.model.encode(clean_texts)
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            logger.error(f"Local batch embedding failed: {str(e)}")
            raise EmbeddingError(f"Local batch embedding error: {str(e)}") from e