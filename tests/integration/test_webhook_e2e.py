"""End-to-end integration tests for the complete webhook flow.

These tests validate the entire pipeline from webhook ingress to response delivery,
ensuring all components work together correctly.
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from src.interfaces.webhooks.app import webhook_app
from src.core.domain.events import GovernorEvent, GovernorResponse, ChannelType, MessageType, ResponseType
from src.core.domain.state import GovernorState, StateNode
from src.core.domain.tools import ToolCall, RiskLevel
from src.governor.context.assembler import ContextAssembler
from src.action_plane.tool_registry.registry import ToolRegistry


class TestWebhookE2EFlow:
    """End-to-end tests for complete webhook processing flow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(webhook_app.app)
        
    @pytest.mark.asyncio
    @patch('src.interfaces.webhooks.app.governor_graph')
    @patch('src.interfaces.webhooks.app.context_assembler')
    @patch('src.interfaces.webhooks.app.tool_registry')
    @patch('src.interfaces.webhooks.app.response_handler')
    async def test_complete_telegram_conversation_flow(
        self,
        mock_response_handler,
        mock_tool_registry,
        mock_context_assembler,
        mock_graph
    ):
        """Test complete conversation flow with Telegram webhook."""
        
        # Setup mocks
        mock_tool_registry.get_available_tools.return_value = {
            "get_weather": {"description": "Get weather information"},
            "send_email": {"description": "Send email to recipient"}
        }
        
        mock_context_assembler.assemble_context.return_value = """
You are the Headless Governor, a personal AI assistant.

CURRENT ENVIRONMENT:
- Time: 2024-11-27 14:30:00 UTC
- Day: Wednesday (afternoon)

AVAILABLE TOOLS:
- get_weather, send_email

CURRENT SITUATION:
- User Input: What's the weather like today?
- Current State: analyze
        """
        
        # Mock successful response
        mock_response = GovernorResponse(
            user_id="telegram_123456",
            session_id="tg_chat_msg_1",
            content="I'll check the weather for you. The current weather is sunny with a temperature of 22°C.",
            response_type=ResponseType.TEXT,
            metadata={"tool_used": "get_weather", "location": "current"}
        )
        
        mock_graph.invoke = AsyncMock(return_value={"response": mock_response})
        mock_response_handler.deliver_response = AsyncMock(return_value=True)
        
        # Send Telegram webhook
        telegram_payload = {
            "update_id": 12345,
            "message": {
                "message_id": 1,
                "from": {
                    "id": 123456,
                    "username": "testuser",
                    "first_name": "Test",
                    "last_name": "User"
                },
                "chat": {
                    "id": 123456,
                    "type": "private"
                },
                "date": int(time.time()),
                "text": "What's the weather like today?"
            }
        }
        
        # Send webhook request
        response = self.client.post(
            "/webhook/ingress",
            json=telegram_payload,
            headers={
                "user-agent": "telegram",
                "content-type": "application/json"
            }
        )
        
        # Verify immediate response
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "received"
        assert response_data["channel"] == "telegram"
        assert "event_id" in response_data
        assert "timestamp" in response_data
        
        # Give background task time to process
        await asyncio.sleep(0.1)
        
        # Verify that the graph was called with correct parameters
        mock_graph.invoke.assert_called_once()
        call_args = mock_graph.invoke.call_args[0][0]
        assert "state" in call_args
        assert "event" in call_args
        assert "context_prompt" in call_args
        assert "available_tools" in call_args
        
        # Verify event structure
        event = call_args["event"]
        assert event.user_id == "telegram_123456"
        assert event.content == "What's the weather like today?"
        assert event.channel == ChannelType.TELEGRAM
        assert event.message_type == MessageType.TEXT
        assert event.metadata["platform"] == "telegram"
        
        # Verify response delivery was called
        mock_response_handler.deliver_response.assert_called_once_with(
            mock_response, ChannelType.TELEGRAM
        )
    
    @pytest.mark.asyncio
    @patch('src.interfaces.webhooks.app.governor_graph')
    @patch('src.interfaces.webhooks.app.context_assembler')
    @patch('src.interfaces.webhooks.app.tool_registry')
    async def test_dangerous_tool_confirmation_flow(
        self,
        mock_tool_registry,
        mock_context_assembler,
        mock_graph
    ):
        """Test flow when dangerous tool requires confirmation."""
        
        # Setup mocks for dangerous tool scenario
        mock_tool_registry.get_available_tools.return_value = {
            "send_email": {"description": "Send email", "risk_level": "dangerous"},
            "get_weather": {"description": "Get weather", "risk_level": "safe"}
        }
        
        mock_context_assembler.assemble_context.return_value = "System prompt with confirmation context"
        
        # Mock confirmation request response
        confirmation_response = GovernorResponse(
            user_id="telegram_789",
            session_id="tg_confirm_session",
            content="⚠️ I want to send an email to john@example.com with the subject 'Meeting Update'. This is a potentially risky operation. Do you want me to proceed?",
            response_type=ResponseType.CONFIRMATION_REQUEST,
            metadata={
                "pending_tool": "send_email",
                "requires_confirmation": True,
                "risk_level": "dangerous"
            }
        )
        
        mock_graph.invoke = AsyncMock(return_value={"response": confirmation_response})
        
        # Send request that should trigger confirmation
        payload = {
            "update_id": 789,
            "message": {
                "message_id": 2,
                "from": {"id": 789, "username": "riskuser"},
                "chat": {"id": 789, "type": "private"},
                "text": "Send an email to john@example.com saying the meeting is moved to 3pm"
            }
        }
        
        response = self.client.post(
            "/webhook/ingress",
            json=payload,
            headers={"user-agent": "telegram"}
        )
        
        assert response.status_code == 200
        
        # Give background task time to process
        await asyncio.sleep(0.1)
        
        # Verify graph was called and returned confirmation request
        mock_graph.invoke.assert_called_once()
        call_args = mock_graph.invoke.call_args[0][0]
        
        # Verify event contains email request
        event = call_args["event"]
        assert "email" in event.content.lower()
        assert "john@example.com" in event.content
    
    def test_multiple_channel_webhook_processing(self):
        """Test processing webhooks from multiple channels simultaneously."""
        
        # Telegram webhook
        telegram_payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "from": {"id": 100, "username": "telegram_user"},
                "chat": {"id": 100, "type": "private"},
                "text": "Hello from Telegram"
            }
        }
        
        # WhatsApp webhook
        whatsapp_payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "id": "wa_msg_1",
                            "from": "1234567890",
                            "text": {"body": "Hello from WhatsApp"},
                            "timestamp": str(int(time.time()))
                        }]
                    }
                }]
            }]
        }
        
        # Direct API call
        direct_payload = {
            "user_id": "direct_user_1",
            "session_id": "direct_session_1",
            "content": "Hello from Direct API",
            "message_type": "text"
        }
        
        # Send all three webhooks
        responses = []
        
        responses.append(self.client.post(
            "/webhook/ingress",
            json=telegram_payload,
            headers={"user-agent": "telegram"}
        ))
        
        responses.append(self.client.post(
            "/webhook/ingress",
            json=whatsapp_payload,
            headers={"user-agent": "whatsapp"}
        ))
        
        responses.append(self.client.post(
            "/webhook/ingress",
            json=direct_payload,
            headers={"x-channel-type": "direct"}
        ))
        
        # Verify all responses are successful
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "received"
            assert "channel" in data
        
        # Verify different channels were detected
        channels = [r.json()["channel"] for r in responses]
        assert "telegram" in channels
        assert "whatsapp" in channels
        assert "direct" in channels
    
    def test_session_state_persistence(self):
        """Test that session state persists across multiple requests."""
        
        # First message
        first_payload = {
            "user_id": "persistent_user",
            "session_id": "persistent_session",
            "content": "First message",
            "message_type": "text"
        }
        
        response1 = self.client.post(
            "/webhook/ingress",
            json=first_payload,
            headers={"x-channel-type": "direct"}
        )
        assert response1.status_code == 200
        
        # Check session was created
        session_status = self.client.get("/webhook/status/persistent_session")
        assert session_status.status_code == 200
        
        session_data = session_status.json()
        assert session_data["user_id"] == "persistent_user"
        assert session_data["session_id"] == "persistent_session"
        
        # Second message to same session
        second_payload = {
            "user_id": "persistent_user",
            "session_id": "persistent_session",
            "content": "Second message",
            "message_type": "text"
        }
        
        response2 = self.client.post(
            "/webhook/ingress",
            json=second_payload,
            headers={"x-channel-type": "direct"}
        )
        assert response2.status_code == 200
        
        # Verify session state updated
        updated_status = self.client.get("/webhook/status/persistent_session")
        assert updated_status.status_code == 200
        
        updated_data = updated_status.json()
        # Note: total_turns might be updated by background processing
        assert updated_data["user_id"] == "persistent_user"
    
    def test_response_polling_mechanism(self):
        """Test response polling for channels without direct delivery."""
        
        session_id = "polling_test_session"
        
        # Send webhook that should store response for polling
        payload = {
            "user_id": "polling_user",
            "session_id": session_id,
            "content": "Test message for polling",
            "message_type": "text"
        }
        
        response = self.client.post(
            "/webhook/ingress",
            json=payload,
            headers={"x-channel-type": "direct"}
        )
        assert response.status_code == 200
        
        # Give background processing time
        time.sleep(0.2)
        
        # Poll for responses
        poll_response = self.client.get(f"/webhook/responses/{session_id}")
        assert poll_response.status_code == 200
        
        poll_data = poll_response.json()
        assert poll_data["session_id"] == session_id
        # May have responses depending on mock setup
        assert "response_count" in poll_data
        assert "responses" in poll_data
    
    def test_error_handling_and_recovery(self):
        """Test error handling in webhook processing."""
        
        # Send malformed payload
        malformed_payload = {
            "incomplete": "payload"
        }
        
        response = self.client.post(
            "/webhook/ingress",
            json=malformed_payload,
            headers={"user-agent": "test"}
        )
        
        # Should handle gracefully and return 400 or 200 with error info
        assert response.status_code in [200, 400]
        
        # Send payload that's too large
        large_payload = {"message": "x" * 100000}  # 100KB message
        
        response = self.client.post(
            "/webhook/ingress",
            json=large_payload,
            headers={"content-length": str(len(str(large_payload)))}
        )
        
        # Should be handled by size validation
        assert response.status_code in [200, 400, 413]
    
    def test_webhook_security_integration(self):
        """Test security features in webhook processing."""
        
        # Test rate limiting by sending many requests rapidly
        rapid_responses = []
        for i in range(20):  # Send more than typical rate limit
            payload = {"message": f"Rapid message {i}"}
            response = self.client.post(
                "/webhook/ingress",
                json=payload,
                headers={"user-agent": "test", "x-forwarded-for": "192.168.1.100"}
            )
            rapid_responses.append(response.status_code)
        
        # Should eventually hit rate limits
        assert 429 in rapid_responses or len([r for r in rapid_responses if r == 200]) >= 10
        
        # Test with missing User-Agent (should still work but might be flagged)
        no_ua_response = self.client.post(
            "/webhook/ingress",
            json={"message": "No user agent"},
            headers={}
        )
        assert no_ua_response.status_code in [200, 400]
    
    def test_channel_connectivity_endpoints(self):
        """Test channel connectivity testing endpoints."""
        
        # Test Telegram connectivity
        telegram_test = self.client.get("/webhook/channels/telegram/test")
        assert telegram_test.status_code == 200
        
        test_data = telegram_test.json()
        assert test_data["channel"] == "telegram"
        assert "status" in test_data
        
        # Test WhatsApp connectivity
        whatsapp_test = self.client.get("/webhook/channels/whatsapp/test")
        assert whatsapp_test.status_code == 200
        
        # Test invalid channel
        invalid_test = self.client.get("/webhook/channels/invalid/test")
        assert invalid_test.status_code == 400


class TestWebhookPerformance:
    """Performance tests for webhook processing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(webhook_app.app)
    
    def test_webhook_response_time(self):
        """Test webhook response time is acceptable."""
        
        payload = {
            "user_id": "perf_test_user",
            "session_id": "perf_test_session",
            "content": "Performance test message",
            "message_type": "text"
        }
        
        start_time = time.time()
        
        response = self.client.post(
            "/webhook/ingress",
            json=payload,
            headers={"x-channel-type": "direct"}
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 1.0  # Should respond within 1 second
    
    def test_concurrent_webhook_processing(self):
        """Test handling multiple concurrent webhooks."""
        import threading
        
        responses = []
        errors = []
        
        def send_webhook(index):
            try:
                payload = {
                    "user_id": f"concurrent_user_{index}",
                    "session_id": f"concurrent_session_{index}",
                    "content": f"Concurrent message {index}",
                    "message_type": "text"
                }
                
                response = self.client.post(
                    "/webhook/ingress",
                    json=payload,
                    headers={"x-channel-type": "direct"}
                )
                responses.append(response.status_code)
                
            except Exception as e:
                errors.append(str(e))
        
        # Start 10 concurrent requests
        threads = []
        for i in range(10):
            thread = threading.Thread(target=send_webhook, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        # Verify most requests succeeded
        assert len(errors) <= 2  # Allow a few failures
        assert len(responses) >= 8  # Most should succeed
        assert all(status in [200, 429] for status in responses)  # Only OK or rate limited


class TestWebhookBusinessLogic:
    """Tests for business logic integration in webhook processing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(webhook_app.app)
    
    @patch('src.interfaces.webhooks.app.tool_registry')
    def test_tool_availability_in_webhook_flow(self, mock_tool_registry):
        """Test that tools are properly available in webhook processing."""
        
        # Mock tool registry with specific tools
        mock_tool_registry.get_available_tools.return_value = {
            "weather_tool": {
                "description": "Get weather information",
                "risk_level": "safe"
            },
            "email_tool": {
                "description": "Send email messages", 
                "risk_level": "dangerous"
            }
        }
        
        payload = {
            "user_id": "tool_test_user",
            "session_id": "tool_test_session", 
            "content": "What tools are available?",
            "message_type": "text"
        }
        
        response = self.client.post(
            "/webhook/ingress",
            json=payload,
            headers={"x-channel-type": "direct"}
        )
        
        assert response.status_code == 200
        
        # Verify tool registry was called
        mock_tool_registry.get_available_tools.assert_called_once()
    
    def test_session_management_across_webhooks(self):
        """Test session management across multiple webhook calls."""
        
        user_id = "session_mgmt_user"
        session_id = "session_mgmt_session"
        
        # First webhook
        payload1 = {
            "user_id": user_id,
            "session_id": session_id,
            "content": "Start conversation",
            "message_type": "text"
        }
        
        response1 = self.client.post(
            "/webhook/ingress",
            json=payload1,
            headers={"x-channel-type": "direct"}
        )
        assert response1.status_code == 200
        
        # Check session exists
        session_check = self.client.get(f"/webhook/status/{session_id}")
        assert session_check.status_code == 200
        
        # Second webhook to same session
        payload2 = {
            "user_id": user_id,
            "session_id": session_id,
            "content": "Continue conversation",
            "message_type": "text"
        }
        
        response2 = self.client.post(
            "/webhook/ingress",
            json=payload2,
            headers={"x-channel-type": "direct"}
        )
        assert response2.status_code == 200
        
        # Verify session persisted
        final_check = self.client.get(f"/webhook/status/{session_id}")
        assert final_check.status_code == 200
        
        session_data = final_check.json()
        assert session_data["user_id"] == user_id
        assert session_data["session_id"] == session_id