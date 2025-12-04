"""Application configuration using pydantic-settings."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )

    # Application Configuration
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    debug: bool = True
    secret_key: str = Field(
        default="dev-secret-key-change-in-production",
        description="Secret key for application security",
    )

    # Database Configuration - Redis (Short-term Memory)
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL",
    )

    # Database Configuration - PostgreSQL (Episodic Memory + Procedural Rules)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "governor"
    postgres_password: str = "dev_password"
    postgres_db: str = "governor_memory"

    @property
    def postgres_url(self) -> str:
        """Construct PostgreSQL connection URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Database Configuration - Neo4j (Semantic Memory)
    neo4j_url: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j connection URL",
    )
    neo4j_user: str = "neo4j"
    neo4j_password: str = "dev_password"

    @property
    def neo4j_auth(self) -> tuple[str, str]:
        """Neo4j authentication tuple."""
        return (self.neo4j_user, self.neo4j_password)

    # API Keys (External Services)
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key for LLM inference",
    )
    telegram_bot_token: str = Field(
        default="",
        description="Telegram bot token for messaging interface",
    )
    twilio_account_sid: str = Field(
        default="",
        description="Twilio account SID for voice/SMS",
    )
    twilio_auth_token: str = Field(
        default="",
        description="Twilio authentication token",
    )

    # RabbitMQ Configuration (Message Queue for Memory Reflection)
    rabbitmq_url: str = Field(
        default="amqp://umbranet:reflection123@rabbitmq:5672/",
        description="RabbitMQ connection URL",
    )
    rabbitmq_exchange: str = Field(
        default="memory_reflection_exchange",
        description="RabbitMQ exchange for memory reflection jobs",
    )
    rabbitmq_queue: str = Field(
        default="memory_reflection_queue",
        description="Main queue for memory reflection jobs",
    )
    rabbitmq_dead_letter_queue: str = Field(
        default="memory_reflection_dlq",
        description="Dead letter queue for failed reflection jobs",
    )

    # Memory Reflection Service Configuration
    reflection_enabled: bool = Field(
        default=True,
        description="Enable asynchronous memory reflection processing",
    )
    reflection_batch_size: int = Field(
        default=10,
        description="Number of reflection jobs to process in batch",
    )
    reflection_workers: int = Field(
        default=2,
        description="Number of concurrent reflection workers",
    )
    reflection_max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for failed reflection jobs",
    )
    reflection_retry_delay: int = Field(
        default=30,
        description="Base delay in seconds between retry attempts",
    )

    # Performance Configuration
    memory_fast_mode: bool = Field(
        default=True,
        description="Enable fast mode: only Tier 1 during chat, queue reflection",
    )
    reflection_timeout: int = Field(
        default=300,
        description="Maximum time in seconds for reflection processing",
    )

    # FastAPI Configuration
    fastapi_host: str = "0.0.0.0"
    fastapi_port: int = 8000
    fastapi_reload: bool = True

    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"


# Global settings instance
settings = Settings()
