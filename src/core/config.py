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
