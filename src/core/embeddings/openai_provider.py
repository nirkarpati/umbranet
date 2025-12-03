"""OpenAI embedding provider implementation."""

import logging
from typing import Any

from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ...core.config import settings
from .base import EmbeddingError, EmbeddingProvider

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider using text-embedding-3-small model."""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        """Initialize OpenAI embedding provider.
        
        Args:
            model: OpenAI embedding model to use
        """
        self.model = model
        self.client: AsyncOpenAI | None = None
        self._dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
    
    @property
    def dimension(self) -> int:
        """Number of dimensions in the embedding vectors."""
        return self._dimension_map.get(self.model, 1536)
    
    @property
    def model_name(self) -> str:
        """Name/identifier of the embedding model."""
        return f"openai:{self.model}"
    
    async def __aenter__(self) -> "OpenAIEmbeddingProvider":
        """Async context manager entry."""
        if not settings.openai_api_key:
            raise EmbeddingError("OpenAI API key not configured")
        
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self.client:
            await self.client.close()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def _call_openai_embeddings(
        self, 
        texts: list[str]
    ) -> list[list[float]]:
        """Call OpenAI embeddings API with retry logic."""
        if not self.client:
            raise EmbeddingError("Client not initialized - use async context manager")
        
        try:
            # Filter out empty texts
            non_empty_texts = [text.strip() for text in texts if text.strip()]
            if not non_empty_texts:
                return [[0.0] * self.dimension for _ in texts]
            
            response = await self.client.embeddings.create(
                input=non_empty_texts,
                model=self.model
            )
            
            embeddings = [item.embedding for item in response.data]
            
            # Handle case where some texts were empty
            if len(embeddings) != len(texts):
                result = []
                empty_embedding = [0.0] * self.dimension
                non_empty_idx = 0
                
                for text in texts:
                    if text.strip():
                        result.append(embeddings[non_empty_idx])
                        non_empty_idx += 1
                    else:
                        result.append(empty_embedding)
                
                return result
            
            return embeddings
            
        except Exception as e:
            logger.error(f"OpenAI embedding API failed: {str(e)}")
            if "rate_limit" in str(e).lower():
                logger.warning("Rate limit hit, retrying...")
                raise  # Will be retried by tenacity
            else:
                raise EmbeddingError(f"OpenAI API error: {str(e)}") from e
    
    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text string."""
        if not text or not text.strip():
            return [0.0] * self.dimension
        
        embeddings = await self._call_openai_embeddings([text])
        return embeddings[0]
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple text strings."""
        if not texts:
            return []
        
        # OpenAI has batch size limits, chunk if necessary
        chunk_size = 100  # Conservative batch size
        all_embeddings = []
        
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            chunk_embeddings = await self._call_openai_embeddings(chunk)
            all_embeddings.extend(chunk_embeddings)
        
        return all_embeddings