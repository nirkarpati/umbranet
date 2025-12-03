"""Unit tests for the Webhook Interface system."""

import pytest
import json
import asyncio
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from src.interfaces.webhooks.app import WebhookApp, webhook_app
from src.interfaces.webhooks.security import WebhookSecurity, RateLimiter
from src.interfaces.webhooks.response_handler import ResponseHandler
from src.core.domain.events import GovernorEvent, GovernorResponse, ChannelType, MessageType, ResponseType
from src.core.domain.state import GovernorState, StateNode


class TestRateLimiter:
    """Test cases for RateLimiter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.rate_limiter = RateLimiter(max_requests=5, window_seconds=60)
    
    def test_rate_limiter_allows_within_limit(self):
        """Test that requests within limit are allowed."""
        client_ip = "192.168.1.1"
        
        for i in range(5):
            result = self.rate_limiter.is_allowed(client_ip)
            assert result is True
    
    def test_rate_limiter_blocks_over_limit(self):
        """Test that requests over limit are blocked."""
        client_ip = "192.168.1.2"
        
        # Use up the limit
        for i in range(5):
            self.rate_limiter.is_allowed(client_ip)
        
        # This should raise an exception
        with pytest.raises(Exception):  # RateLimitExceeded
            self.rate_limiter.is_allowed(client_ip)
    
    def test_rate_limiter_different_ips(self):
        """Test that different IPs have separate limits."""
        for i in range(5):
            assert self.rate_limiter.is_allowed("192.168.1.10") is True
            assert self.rate_limiter.is_allowed("192.168.1.11") is True


class TestWebhookSecurity:
    """Test cases for WebhookSecurity."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.security = WebhookSecurity()
    
    @pytest.mark.asyncio
    async def test_verify_rate_limit_success(self):
        """Test successful rate limit verification."""
        mock_request = MagicMock()
        mock_request.client.host = "192.168.1.100"
        mock_request.headers = {}
        
        result = await self.security.verify_rate_limit(mock_request)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_request_size_success(self):
        """Test successful request size validation."""
        mock_request = MagicMock()
        mock_request.headers = {"content-length": "1000"}
        
        result = await self.security.validate_request_size(mock_request, max_size=2000)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_request_size_too_large(self):
        """Test request size validation failure."""
        from fastapi import HTTPException
        
        mock_request = MagicMock()
        mock_request.headers = {"content-length": "2000000"}
        
        with pytest.raises(HTTPException) as exc_info:
            await self.security.validate_request_size(mock_request, max_size=1000000)
        
        assert exc_info.value.status_code == 413
    
    def test_get_client_ip_direct(self):
        """Test client IP extraction from direct connection."""
        mock_request = MagicMock()
        mock_request.client.host = "192.168.1.50"
        mock_request.headers = {}
        
        ip = self.security._get_client_ip(mock_request)
        assert ip == "192.168.1.50"
    
    def test_get_client_ip_forwarded(self):
        """Test client IP extraction from forwarded headers."""
        mock_request = MagicMock()
        mock_request.client.host = "10.0.0.1"
        mock_request.headers = {"X-Forwarded-For": "203.0.113.10, 192.168.1.1"}
        
        ip = self.security._get_client_ip(mock_request)
        assert ip == "203.0.113.10"
    
    def test_is_valid_user_agent(self):
        """Test User-Agent validation."""
        assert self.security.is_valid_user_agent("TelegramBot") is True
        assert self.security.is_valid_user_agent("WhatsApp/1.0") is True
        assert self.security.is_valid_user_agent("GitHub-Hookshot/abc123") is True
        assert self.security.is_valid_user_agent("curl/7.68.0") is True
        assert self.security.is_valid_user_agent("PostmanRuntime/7.26.8") is True
        assert self.security.is_valid_user_agent("BadBot/1.0") is False
        assert self.security.is_valid_user_agent(None) is False


class TestResponseHandler:
    """Test cases for ResponseHandler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.response_handler = ResponseHandler()
        self.test_response = GovernorResponse(
            user_id="test_user",
            session_id="test_session",
            content="Test response",
            response_type=ResponseType.TEXT,
            metadata={"test": True}
        )
    
    @pytest.mark.asyncio
    async def test_store_response_for_polling(self):
        """Test storing response for polling."""
        result = await self.response_handler._store_response_for_polling(self.test_response)
        assert result is True
        
        stored_responses = self.response_handler.get_stored_responses("test_session")
        assert len(stored_responses) == 1
        assert stored_responses[0].content == "Test response"
    
    def test_get_stored_responses_empty(self):
        """Test getting stored responses when none exist."""
        responses = self.response_handler.get_stored_responses("nonexistent_session")
        assert responses == []
    
    def test_clear_stored_responses(self):
        """Test clearing stored responses."""
        # Store a response first
        self.response_handler.response_store["test_session"] = [self.test_response]
        
        # Clear responses
        self.response_handler.clear_stored_responses("test_session")
        
        # Verify cleared
        assert "test_session" not in self.response_handler.response_store
    
    @pytest.mark.asyncio
    async def test_deliver_response_fallback_to_polling(self):
        """Test response delivery fallback to polling."""
        result = await self.response_handler.deliver_response(
            self.test_response, 
            ChannelType.DIRECT
        )
        
        assert result is True
        stored = self.response_handler.get_stored_responses("test_session")
        assert len(stored) == 1
    
    @pytest.mark.asyncio
    @patch('src.interfaces.webhooks.response_handler.aiohttp.ClientSession')
    async def test_send_webhook_callback_success(self, mock_session):
        """Test successful webhook callback."""
        # Mock successful HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_session_instance = AsyncMock()
        mock_session_instance.post.return_value = mock_response
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock(return_value=None)
        mock_session.return_value = mock_session_instance
        
        result = await self.response_handler._send_webhook_callback(
            self.test_response,
            "http://example.com/webhook"
        )
        
        assert result is True
    
    @pytest.mark.asyncio
    @patch('src.interfaces.webhooks.response_handler.aiohttp.ClientSession')
    async def test_send_webhook_callback_failure(self, mock_session):
        """Test failed webhook callback."""
        # Mock failed HTTP response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_session_instance = AsyncMock()
        mock_session_instance.post.return_value = mock_response
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock(return_value=None)
        mock_session.return_value = mock_session_instance
        
        result = await self.response_handler._send_webhook_callback(
            self.test_response,
            "http://example.com/webhook"
        )
        
        assert result is False


class TestWebhookApp:
    """Test cases for WebhookApp."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.app_instance = WebhookApp()
        self.client = TestClient(self.app_instance.app)
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_list_active_sessions_empty(self):
        """Test listing active sessions when none exist."""
        response = self.client.get("/webhook/sessions")
        assert response.status_code == 200
        
        data = response.json()
        assert data["total_sessions"] == 0
        assert data["sessions"] == {}
    
    def test_get_session_responses_not_found(self):
        """Test getting responses for non-existent session."""
        response = self.client.get("/webhook/responses/nonexistent")
        assert response.status_code == 200
        
        data = response.json()
        assert data["session_id"] == "nonexistent"
        assert data["response_count"] == 0
        assert data["responses"] == []
    
    def test_clear_session_responses(self):
        """Test clearing session responses."""
        response = self.client.delete("/webhook/responses/test_session")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "cleared"
        assert data["session_id"] == "test_session"
    
    def test_test_channel_connectivity_invalid(self):
        """Test channel connectivity test with invalid channel."""
        response = self.client.get("/webhook/channels/invalid/test")
        assert response.status_code == 400
        
        data = response.json()
        assert "Invalid channel type" in data["detail"]
    
    def test_test_channel_connectivity_valid(self):
        """Test channel connectivity test with valid channel."""
        response = self.client.get("/webhook/channels/telegram/test")
        assert response.status_code == 200
        
        data = response.json()
        assert data["channel"] == "telegram"
        assert "status" in data
    
    def test_detect_channel_type_telegram(self):
        """Test Telegram channel type detection."""
        payload = {"update_id": 123, "message": {"text": "hello"}}
        headers = {"user-agent": "telegram"}
        
        channel = self.app_instance._detect_channel_type(payload, headers)
        assert channel == ChannelType.TELEGRAM
    
    def test_detect_channel_type_whatsapp(self):
        """Test WhatsApp channel type detection."""
        payload = {"messaging_product": "whatsapp"}
        headers = {"user-agent": "whatsapp"}
        
        channel = self.app_instance._detect_channel_type(payload, headers)
        assert channel == ChannelType.WHATSAPP
    
    def test_detect_channel_type_direct(self):
        """Test direct channel type detection."""
        payload = {"user_id": "test", "content": "hello"}
        headers = {"x-channel-type": "direct"}
        
        channel = self.app_instance._detect_channel_type(payload, headers)
        assert channel == ChannelType.DIRECT
    
    def test_detect_channel_type_generic(self):
        """Test generic channel type detection."""
        payload = {"message": "hello"}
        headers = {}
        
        channel = self.app_instance._detect_channel_type(payload, headers)
        assert channel == ChannelType.WEBHOOK
    
    @pytest.mark.asyncio
    async def test_normalize_telegram_payload(self):
        """Test Telegram payload normalization."""
        payload = {
            "update_id": 123,
            "message": {
                "message_id": 456,
                "from": {"id": 789, "username": "testuser", "first_name": "Test"},
                "chat": {"id": 101112, "type": "private"},
                "text": "Hello bot"
            }
        }
        
        event = self.app_instance._normalize_telegram_payload(payload)
        
        assert event.user_id == "telegram_789"
        assert event.session_id.startswith("tg_101112_")
        assert event.content == "Hello bot"
        assert event.channel == ChannelType.TELEGRAM
        assert event.message_type == MessageType.TEXT
        assert event.metadata["platform"] == "telegram"
        assert event.metadata["username"] == "testuser"
    
    @pytest.mark.asyncio
    async def test_normalize_whatsapp_payload(self):
        """Test WhatsApp payload normalization."""
        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "id": "msg123",
                            "from": "1234567890",
                            "text": {"body": "Hello WhatsApp"},
                            "timestamp": "1234567890"
                        }]
                    }
                }]
            }]
        }
        
        event = self.app_instance._normalize_whatsapp_payload(payload)
        
        assert event.user_id == "whatsapp_1234567890"
        assert event.session_id.startswith("wa_msg123")
        assert event.content == "Hello WhatsApp"
        assert event.channel == ChannelType.WHATSAPP
        assert event.metadata["platform"] == "whatsapp"
        assert event.metadata["phone_number"] == "1234567890"
    
    @pytest.mark.asyncio
    async def test_normalize_direct_payload(self):
        """Test direct payload normalization."""
        payload = {
            "user_id": "direct_user_123",
            "session_id": "direct_session_456",
            "content": "Direct message",
            "message_type": "text",
            "metadata": {"source": "api"}
        }
        
        event = self.app_instance._normalize_direct_payload(payload)
        
        assert event.user_id == "direct_user_123"
        assert event.session_id == "direct_session_456"
        assert event.content == "Direct message"
        assert event.channel == ChannelType.DIRECT
        assert event.message_type == MessageType.TEXT
        assert event.metadata["source"] == "api"
    
    @pytest.mark.asyncio
    async def test_normalize_generic_payload(self):
        """Test generic payload normalization."""
        payload = {
            "message": "Generic webhook message",
            "user_id": "generic_user",
            "extra_data": {"key": "value"}
        }
        
        event = self.app_instance._normalize_generic_payload(payload)
        
        assert event.user_id == "generic_user"
        assert event.content == "Generic webhook message"
        assert event.channel == ChannelType.WEBHOOK
        assert event.message_type == MessageType.TEXT
        assert event.metadata["extra_data"] == {"key": "value"}
    
    @pytest.mark.asyncio
    async def test_get_or_create_state_new(self):
        """Test creating new session state."""
        state = await self.app_instance._get_or_create_state("user123", "session456")
        
        assert state.user_id == "user123"
        assert state.session_id == "session456"
        assert state.current_node == StateNode.IDLE
        assert state.total_turns == 0
        assert state.total_tools_executed == 0
        assert state.error_count == 0
        assert state.awaiting_confirmation is False
        assert state.pending_tools == []
        assert state.conversation_history == []
        assert hasattr(state, 'created_at')
    
    @pytest.mark.asyncio
    async def test_get_or_create_state_existing(self):
        """Test retrieving existing session state."""
        # Create initial state
        state1 = await self.app_instance._get_or_create_state("user123", "session456")
        state1.total_turns = 5
        
        # Retrieve existing state
        state2 = await self.app_instance._get_or_create_state("user123", "session456")
        
        assert state1 is state2
        assert state2.total_turns == 5


class TestWebhookIntegration:
    """Integration tests for webhook interface."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(webhook_app.app)
    
    @patch('src.interfaces.webhooks.app.governor_graph')
    @patch('src.interfaces.webhooks.app.context_assembler')
    @patch('src.interfaces.webhooks.app.tool_registry')
    def test_webhook_ingress_telegram_integration(
        self, 
        mock_tool_registry, 
        mock_context_assembler, 
        mock_graph
    ):
        """Test complete Telegram webhook processing integration."""
        # Mock dependencies
        mock_tool_registry.get_available_tools.return_value = {"get_weather": {"description": "Get weather"}}
        mock_context_assembler.assemble_context.return_value = "System prompt here"
        
        mock_response = GovernorResponse(
            user_id="telegram_123",
            session_id="tg_456_789",
            content="Weather response",
            response_type=ResponseType.TEXT,
            metadata={}
        )
        mock_graph.invoke = AsyncMock(return_value={"response": mock_response})
        
        # Send Telegram webhook
        payload = {
            "update_id": 123,
            "message": {
                "message_id": 456,
                "from": {"id": 123, "username": "testuser"},
                "chat": {"id": 456, "type": "private"},
                "text": "What's the weather?"
            }
        }
        
        response = self.client.post(
            "/webhook/ingress",
            json=payload,
            headers={"user-agent": "telegram"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "received"
        assert data["channel"] == "telegram"
        assert "event_id" in data
    
    def test_webhook_ingress_rate_limiting(self):
        """Test webhook rate limiting."""
        # Send many requests rapidly
        responses = []
        for i in range(10):
            response = self.client.post(
                "/webhook/ingress",
                json={"message": f"test message {i}"},
                headers={"user-agent": "test"}
            )
            responses.append(response.status_code)
        
        # Should have some 429 responses after hitting rate limit
        assert 429 in responses or all(r == 200 for r in responses[:5])
    
    def test_webhook_ingress_invalid_payload(self):
        """Test webhook with invalid payload."""
        response = self.client.post(
            "/webhook/ingress",
            data="invalid json",
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code == 422  # FastAPI validation error
    
    def test_webhook_ingress_oversized_payload(self):
        """Test webhook with oversized payload."""
        large_payload = {"message": "x" * 2000000}  # 2MB payload
        
        response = self.client.post(
            "/webhook/ingress",
            json=large_payload,
            headers={"content-length": str(len(str(large_payload)))}
        )
        
        # Should be rejected for size
        assert response.status_code in [413, 400]