"""Unit tests for domain event models."""

from datetime import datetime
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.core.domain.events import (
    ChannelType,
    GovernorEvent,
    GovernorResponse,
    MessageType,
    ResponseType,
)


class TestGovernorEvent:
    """Test cases for GovernorEvent model."""

    def test_valid_governor_event(self) -> None:
        """Test creating a valid GovernorEvent."""
        event = GovernorEvent(
            user_id="user_123",
            session_id="session_456",
            message_type=MessageType.TEXT,
            content="Hello, world!",
            channel=ChannelType.TELEGRAM,
            metadata={"telegram_message_id": 12345}
        )
        
        assert event.user_id == "user_123"
        assert event.session_id == "session_456"
        assert event.message_type == MessageType.TEXT
        assert event.content == "Hello, world!"
        assert event.channel == ChannelType.TELEGRAM
        assert event.metadata == {"telegram_message_id": 12345}
        assert isinstance(event.timestamp, datetime)

    def test_governor_event_validation_empty_content(self) -> None:
        """Test that empty content is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            GovernorEvent(
                user_id="user_123",
                session_id="session_456",
                message_type=MessageType.TEXT,
                content="   ",  # Only whitespace
                channel=ChannelType.TELEGRAM
            )
        
        assert "Content cannot be empty" in str(exc_info.value)

    def test_governor_event_validation_invalid_user_id(self) -> None:
        """Test that invalid user IDs are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            GovernorEvent(
                user_id="user@123!",  # Invalid characters
                session_id="session_456",
                message_type=MessageType.TEXT,
                content="Hello",
                channel=ChannelType.TELEGRAM
            )
        
        assert "alphanumeric with optional hyphens/underscores" in str(exc_info.value)

    def test_governor_event_content_stripping(self) -> None:
        """Test that content is properly stripped of whitespace."""
        event = GovernorEvent(
            user_id="user_123",
            session_id="session_456", 
            message_type=MessageType.TEXT,
            content="  Hello, world!  ",  # Leading/trailing spaces
            channel=ChannelType.TELEGRAM
        )
        
        assert event.content == "Hello, world!"

    def test_governor_event_all_channels(self) -> None:
        """Test that all channel types are supported."""
        for channel in ChannelType:
            event = GovernorEvent(
                user_id="user_123",
                session_id="session_456",
                message_type=MessageType.TEXT,
                content="Test message",
                channel=channel
            )
            assert event.channel == channel

    def test_governor_event_all_message_types(self) -> None:
        """Test that all message types are supported."""
        for msg_type in MessageType:
            event = GovernorEvent(
                user_id="user_123",
                session_id="session_456",
                message_type=msg_type,
                content="Test content",
                channel=ChannelType.API
            )
            assert event.message_type == msg_type


class TestGovernorResponse:
    """Test cases for GovernorResponse model."""

    def test_valid_governor_response(self) -> None:
        """Test creating a valid GovernorResponse."""
        response = GovernorResponse(
            user_id="user_123",
            session_id="session_456",
            content="Hello! How can I help?",
            channel=ChannelType.TELEGRAM,
            metadata={"parse_mode": "markdown"}
        )
        
        assert response.user_id == "user_123"
        assert response.session_id == "session_456"
        assert response.content == "Hello! How can I help?"
        assert response.response_type == ResponseType.TEXT  # Default
        assert response.channel == ChannelType.TELEGRAM
        assert response.requires_confirmation is False  # Default
        assert response.urgency_level == "normal"  # Default
        assert isinstance(response.timestamp, datetime)

    def test_governor_response_with_confirmation(self) -> None:
        """Test response with confirmation requirement."""
        response = GovernorResponse(
            user_id="user_123",
            session_id="session_456",
            content="Do you want me to send this email?",
            channel=ChannelType.TELEGRAM,
            response_type=ResponseType.CONFIRMATION_REQUEST,
            requires_confirmation=True,
            confirmation_context={"tool": "send_email", "recipient": "test@example.com"},
            urgency_level="high"
        )
        
        assert response.response_type == ResponseType.CONFIRMATION_REQUEST
        assert response.requires_confirmation is True
        assert response.confirmation_context == {
            "tool": "send_email", 
            "recipient": "test@example.com"
        }
        assert response.urgency_level == "high"

    def test_governor_response_invalid_urgency(self) -> None:
        """Test that invalid urgency levels are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            GovernorResponse(
                user_id="user_123",
                session_id="session_456",
                content="Test response",
                channel=ChannelType.API,
                urgency_level="super_urgent"  # Invalid
            )
        
        assert "Urgency level must be one of" in str(exc_info.value)

    def test_governor_response_all_types(self) -> None:
        """Test that all response types are supported."""
        for response_type in ResponseType:
            response = GovernorResponse(
                user_id="user_123",
                session_id="session_456",
                content="Test content",
                channel=ChannelType.API,
                response_type=response_type
            )
            assert response.response_type == response_type

    def test_governor_response_urgency_levels(self) -> None:
        """Test all valid urgency levels."""
        valid_levels = ["low", "normal", "high", "urgent"]
        
        for level in valid_levels:
            response = GovernorResponse(
                user_id="user_123",
                session_id="session_456",
                content="Test content",
                channel=ChannelType.API,
                urgency_level=level
            )
            assert response.urgency_level == level