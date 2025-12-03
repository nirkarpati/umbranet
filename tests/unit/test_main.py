"""Unit tests for main application module."""

import pytest
from fastapi.testclient import TestClient

from src.main import app


class TestMainApplication:
    """Test cases for the main FastAPI application."""

    @pytest.fixture
    def client(self) -> TestClient:
        """Create a test client for the FastAPI application."""
        return TestClient(app)

    def test_root_endpoint(self, client: TestClient) -> None:
        """Test the root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Headless Governor System"
        assert data["status"] == "operational"
        assert data["version"] == "1.0.1"
        assert data["environment"] == "development"

    def test_health_check_endpoint(self, client: TestClient) -> None:
        """Test the health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["environment"] == "development"
        assert data["debug"] == "True"

    def test_config_endpoint_development(self, client: TestClient) -> None:
        """Test the config endpoint in development mode."""
        response = client.get("/config")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "redis_url" in data
        assert "postgres_host" in data
        assert "neo4j_url" in data
        assert "log_level" in data

    def test_openapi_docs_available(self, client: TestClient) -> None:
        """Test that OpenAPI documentation is available."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_json_available(self, client: TestClient) -> None:
        """Test that OpenAPI JSON schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        data = response.json()
        assert data["info"]["title"] == "Headless Governor System"
        assert data["info"]["version"] == "1.0.1"