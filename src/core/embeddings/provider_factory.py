"""Factory for creating embedding providers."""

from enum import Enum
from typing import Any

from ...core.config import settings
from .base import EmbeddingError, EmbeddingProvider
from .local_provider import LocalEmbeddingProvider
from .openai_provider import OpenAIEmbeddingProvider


class ProviderType(str, Enum):
    """Supported embedding provider types."""
    
    OPENAI = "openai"
    LOCAL = "local"


class EmbeddingProviderFactory:
    """Factory for creating embedding providers."""
    
    @staticmethod
    def create_provider(
        provider_type: ProviderType = ProviderType.OPENAI,
        **kwargs: Any
    ) -> EmbeddingProvider:
        """Create an embedding provider instance.
        
        Args:
            provider_type: Type of provider to create
            **kwargs: Provider-specific configuration
            
        Returns:
            Embedding provider instance
            
        Raises:
            EmbeddingError: If provider creation fails
        """
        if provider_type == ProviderType.OPENAI:
            if not settings.openai_api_key:
                raise EmbeddingError(
                    "OpenAI API key required but not configured. "
                    "Set OPENAI_API_KEY environment variable."
                )
            
            model = kwargs.get("model", "text-embedding-3-small")
            return OpenAIEmbeddingProvider(model=model)
        
        elif provider_type == ProviderType.LOCAL:
            model_name = kwargs.get("model_name", "all-MiniLM-L6-v2")
            return LocalEmbeddingProvider(model_name=model_name)
        
        else:
            raise EmbeddingError(f"Unsupported provider type: {provider_type}")
    
    @staticmethod
    def get_default_provider() -> EmbeddingProvider:
        """Get the default embedding provider based on configuration.
        
        Returns:
            Default embedding provider
        """
        # Prefer OpenAI if API key is available, otherwise use local
        if settings.openai_api_key:
            return EmbeddingProviderFactory.create_provider(ProviderType.OPENAI)
        else:
            return EmbeddingProviderFactory.create_provider(ProviderType.LOCAL)


# Convenience function for getting the default provider
def get_embedding_provider(
    provider_type: ProviderType | None = None,
    **kwargs: Any
) -> EmbeddingProvider:
    """Get an embedding provider instance.
    
    Args:
        provider_type: Optional provider type, uses default if None
        **kwargs: Provider-specific configuration
        
    Returns:
        Embedding provider instance
    """
    if provider_type:
        return EmbeddingProviderFactory.create_provider(provider_type, **kwargs)
    else:
        return EmbeddingProviderFactory.get_default_provider()