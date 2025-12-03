"""Unit tests for configuration module."""

from unittest.mock import patch

from src.core.config import Settings


class TestSettings:
    """Test cases for application settings."""

    def test_default_settings(self) -> None:
        """Test default settings values."""
        settings = Settings()
        
        assert settings.environment == "development"
        assert settings.log_level == "INFO"
        assert settings.debug is True
        assert settings.fastapi_host == "0.0.0.0"
        assert settings.fastapi_port == 8000

    def test_postgres_url_construction(self) -> None:
        """Test PostgreSQL URL construction."""
        settings = Settings()
        
        # Should use the default host from environment (.env file has vector_db)
        expected_url = (
            "postgresql://governor:dev_password"
            "@vector_db:5432/governor_memory"
        )
        assert settings.postgres_url == expected_url

    def test_neo4j_auth_tuple(self) -> None:
        """Test Neo4j authentication tuple."""
        settings = Settings()
        
        auth_tuple = settings.neo4j_auth
        assert auth_tuple == ("neo4j", "dev_password")

    def test_environment_detection_development(self) -> None:
        """Test development environment detection."""
        settings = Settings()
        
        assert settings.is_development() is True
        assert settings.is_production() is False

    def test_environment_detection_production(self) -> None:
        """Test production environment detection."""
        with patch.dict("os.environ", {"ENVIRONMENT": "production"}):
            settings = Settings()
            
            assert settings.is_production() is True
            assert settings.is_development() is False

    @patch.dict("os.environ", {
        "REDIS_URL": "redis://custom:6380",
        "POSTGRES_HOST": "custom-postgres",
        "POSTGRES_PORT": "5433",
        "NEO4J_URL": "bolt://custom-neo4j:7688"
    })
    def test_custom_environment_variables(self) -> None:
        """Test custom environment variable loading."""
        settings = Settings()
        
        assert settings.redis_url == "redis://custom:6380"
        assert settings.postgres_host == "custom-postgres"
        assert settings.postgres_port == 5433
        assert settings.neo4j_url == "bolt://custom-neo4j:7688"

    def test_postgres_url_with_custom_values(self) -> None:
        """Test PostgreSQL URL with custom values."""
        with patch.dict("os.environ", {
            "POSTGRES_HOST": "custom-host",
            "POSTGRES_PORT": "5433",
            "POSTGRES_USER": "custom-user",
            "POSTGRES_PASSWORD": "custom-pass",
            "POSTGRES_DB": "custom-db"
        }):
            settings = Settings()
            
            expected_url = (
                "postgresql://custom-user:custom-pass"
                "@custom-host:5433/custom-db"
            )
            assert settings.postgres_url == expected_url