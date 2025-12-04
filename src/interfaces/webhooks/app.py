"""FastAPI Webhook Application - Unified ingress for all communication channels.

This module provides the main webhook application that serves as the unified
entry point for all external communication channels (Telegram, WhatsApp, etc.).
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from ...core.domain.events import GovernorEvent, GovernorResponse, MessageType, ChannelType, ResponseType
from ...core.domain.state import GovernorState, StateNode
from ...governor.state_machine.graph import GovernorGraph
from ...governor.context.assembler import ContextAssembler
from ...action_plane.tool_registry.registry import ToolRegistry
from .response_handler import response_handler
from .security import webhook_security

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize core components
governor_graph = GovernorGraph()
context_assembler = ContextAssembler()
tool_registry = ToolRegistry()

# Global state store (in production, this would be Redis/database)
session_states: Dict[str, GovernorState] = {}


class WebhookApp:
    """Main webhook application class."""
    
    def __init__(self):
        """Initialize the FastAPI webhook application."""
        self.app = self._create_app()
        self._setup_routes()
        self._setup_middleware()
    
    def _create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        return FastAPI(
            title="Headless Governor Webhook Interface",
            description="Unified ingress for all communication channels",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
    
    def _setup_middleware(self) -> None:
        """Set up middleware for the application."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self) -> None:
        """Set up all application routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0"
            }
        
        @self.app.post("/webhook/ingress")
        async def webhook_ingress(
            request: Request,
            background_tasks: BackgroundTasks,
            authorization: Optional[str] = Header(None),
            x_webhook_signature: Optional[str] = Header(None),
            x_telegram_bot_api_secret_token: Optional[str] = Header(None),
            x_hub_signature_256: Optional[str] = Header(None)
        ) -> Dict[str, Any]:
            """Generic webhook ingress for all channels.
            
            This endpoint receives webhooks from various communication platforms
            and processes them asynchronously through the Governor state machine.
            
            Args:
                request: FastAPI request object containing the webhook payload
                background_tasks: FastAPI background tasks for async processing
                authorization: Optional authorization header
                x_webhook_signature: Optional webhook signature for verification
                x_telegram_bot_api_secret_token: Telegram webhook signature
                x_hub_signature_256: GitHub webhook signature
                
            Returns:
                Immediate response acknowledging receipt of the webhook
                
            Raises:
                HTTPException: If payload processing fails or security checks fail
            """
            try:
                # Security checks
                await webhook_security.verify_rate_limit(request)
                await webhook_security.validate_request_size(request)
                
                # Get raw payload for signature verification
                payload_body = await request.body()
                payload = await request.json()
                headers = dict(request.headers)
                
                # Detect channel type for appropriate security handling
                channel_type = self._detect_channel_type(payload, headers)
                
                # Choose appropriate signature header based on channel
                signature_header = None
                if channel_type == ChannelType.TELEGRAM and x_telegram_bot_api_secret_token:
                    signature_header = x_telegram_bot_api_secret_token
                    channel_str = "telegram"
                elif x_hub_signature_256:  # GitHub
                    signature_header = x_hub_signature_256
                    channel_str = "github"
                elif x_webhook_signature:  # Generic
                    signature_header = x_webhook_signature
                    channel_str = "generic"
                else:
                    channel_str = "generic"
                
                # Verify webhook signature if configured
                await webhook_security.verify_webhook_signature(
                    request=request,
                    payload_body=payload_body,
                    signature_header=signature_header,
                    channel_type=channel_str
                )
                
                # API key authentication for sensitive endpoints (optional)
                api_key_info = await webhook_security.verify_api_key(authorization)
                
                # Normalize to GovernorEvent
                event = await self._normalize_payload(payload, headers)
                
                # Log incoming event with security info
                logger.info(
                    f"Received webhook event: {event.user_id} / {event.session_id} "
                    f"(channel: {channel_type.value}, auth: {bool(api_key_info)})"
                )
                
                # Process asynchronously
                background_tasks.add_task(self._process_governor_event, event)
                
                # Return immediate 200 OK response
                return {
                    "status": "received",
                    "event_id": event.session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "channel": channel_type.value
                }
                
            except HTTPException:
                # Re-raise HTTP exceptions (security failures)
                raise
            except Exception as e:
                logger.error(f"Webhook ingress error: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")
        
        @self.app.get("/webhook/status/{session_id}")
        async def get_session_status(session_id: str) -> Dict[str, Any]:
            """Get status of a specific session.
            
            Args:
                session_id: Session identifier
                
            Returns:
                Session status information
            """
            state = session_states.get(session_id)
            if not state:
                raise HTTPException(status_code=404, detail="Session not found")
            
            return {
                "session_id": session_id,
                "user_id": state.user_id,
                "current_node": state.current_node.value,
                "awaiting_confirmation": state.awaiting_confirmation,
                "total_turns": state.total_turns,
                "tools_executed": state.total_tools_executed,
                "pending_tools": len(state.pending_tools),
                "created_at": state.created_at.isoformat() if hasattr(state, 'created_at') else None
            }
        
        @self.app.get("/webhook/sessions")
        async def list_active_sessions() -> Dict[str, Any]:
            """List all active sessions.
            
            Returns:
                Dictionary of active sessions with basic info
            """
            sessions = {}
            for session_id, state in session_states.items():
                sessions[session_id] = {
                    "user_id": state.user_id,
                    "current_node": state.current_node.value,
                    "awaiting_confirmation": state.awaiting_confirmation,
                    "total_turns": state.total_turns
                }
            
            return {
                "total_sessions": len(sessions),
                "sessions": sessions
            }
        
        @self.app.get("/webhook/responses/{session_id}")
        async def get_session_responses(session_id: str) -> Dict[str, Any]:
            """Get stored responses for a session (polling endpoint).
            
            Args:
                session_id: Session identifier
                
            Returns:
                List of stored responses
            """
            responses = response_handler.get_stored_responses(session_id)
            
            return {
                "session_id": session_id,
                "response_count": len(responses),
                "responses": [
                    {
                        "content": r.content,
                        "response_type": r.response_type.value,
                        "timestamp": r.timestamp.isoformat(),
                        "metadata": r.metadata
                    }
                    for r in responses
                ]
            }
        
        @self.app.delete("/webhook/responses/{session_id}")
        async def clear_session_responses(session_id: str) -> Dict[str, Any]:
            """Clear stored responses for a session.
            
            Args:
                session_id: Session identifier
                
            Returns:
                Confirmation of clearing
            """
            response_handler.clear_stored_responses(session_id)
            
            return {
                "status": "cleared",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.get("/webhook/channels/{channel_type}/test")
        async def test_channel_connectivity(channel_type: str) -> Dict[str, Any]:
            """Test connectivity to a specific channel.
            
            Args:
                channel_type: Type of channel to test (telegram, whatsapp, etc.)
                
            Returns:
                Test results
            """
            try:
                channel = ChannelType(channel_type.lower())
                result = await response_handler.test_channel_connectivity(channel)
                return result
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid channel type: {channel_type}"
                )
    
    async def _normalize_payload(
        self, 
        payload: Dict[str, Any], 
        headers: Dict[str, str]
    ) -> GovernorEvent:
        """Normalize incoming webhook payload to GovernorEvent.
        
        This method handles payload normalization from various platforms:
        - Telegram Bot API
        - WhatsApp Business API
        - Generic webhooks
        - Direct API calls
        
        Args:
            payload: Raw webhook payload
            headers: HTTP headers from the request
            
        Returns:
            Normalized GovernorEvent
            
        Raises:
            ValueError: If payload cannot be normalized
        """
        try:
            # Detect channel type from headers or payload structure
            channel = self._detect_channel_type(payload, headers)
            
            # Extract common fields based on channel
            if channel == ChannelType.TELEGRAM:
                return self._normalize_telegram_payload(payload)
            elif channel == ChannelType.WHATSAPP:
                return self._normalize_whatsapp_payload(payload)
            elif channel == ChannelType.DIRECT:
                return self._normalize_direct_payload(payload)
            else:
                # Generic normalization
                return self._normalize_generic_payload(payload)
                
        except Exception as e:
            logger.error(f"Payload normalization failed: {e}")
            raise ValueError(f"Cannot normalize payload: {e}")
    
    def _detect_channel_type(
        self, 
        payload: Dict[str, Any], 
        headers: Dict[str, str]
    ) -> ChannelType:
        """Detect the communication channel from payload structure.
        
        Args:
            payload: Webhook payload
            headers: HTTP headers
            
        Returns:
            Detected channel type
        """
        # Check User-Agent or specific headers
        user_agent = headers.get("user-agent", "").lower()
        
        # Telegram Bot API
        if "telegram" in user_agent or "update_id" in payload:
            return ChannelType.TELEGRAM
        
        # WhatsApp Business API
        if "whatsapp" in user_agent or "messaging_product" in payload:
            return ChannelType.WHATSAPP
        
        # Direct API calls (from tests or direct integrations)
        if headers.get("x-channel-type") == "direct":
            return ChannelType.DIRECT
        
        # Default to generic
        return ChannelType.WEBHOOK
    
    def _normalize_telegram_payload(self, payload: Dict[str, Any]) -> GovernorEvent:
        """Normalize Telegram Bot API payload."""
        message = payload.get("message", {})
        chat = message.get("chat", {})
        user = message.get("from", {})
        
        user_id = f"telegram_{user.get('id', 'unknown')}"
        session_id = f"tg_{chat.get('id', 'unknown')}_{message.get('message_id', uuid.uuid4())}"
        content = message.get("text", "")
        
        return GovernorEvent(
            user_id=user_id,
            session_id=session_id,
            message_type=MessageType.TEXT,
            content=content,
            metadata={
                "platform": "telegram",
                "chat_type": chat.get("type"),
                "username": user.get("username"),
                "first_name": user.get("first_name"),
                "message_id": message.get("message_id")
            },
            timestamp=datetime.utcnow(),
            channel=ChannelType.TELEGRAM
        )
    
    def _normalize_whatsapp_payload(self, payload: Dict[str, Any]) -> GovernorEvent:
        """Normalize WhatsApp Business API payload."""
        entry = payload.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})
        messages = value.get("messages", [{}])
        
        if not messages:
            raise ValueError("No messages in WhatsApp payload")
        
        message = messages[0]
        
        user_id = f"whatsapp_{message.get('from', 'unknown')}"
        session_id = f"wa_{message.get('id', uuid.uuid4())}"
        content = message.get("text", {}).get("body", "")
        
        return GovernorEvent(
            user_id=user_id,
            session_id=session_id,
            message_type=MessageType.TEXT,
            content=content,
            metadata={
                "platform": "whatsapp",
                "phone_number": message.get("from"),
                "message_id": message.get("id"),
                "timestamp": message.get("timestamp")
            },
            timestamp=datetime.utcnow(),
            channel=ChannelType.WHATSAPP
        )
    
    def _normalize_direct_payload(self, payload: Dict[str, Any]) -> GovernorEvent:
        """Normalize direct API call payload."""
        return GovernorEvent(
            user_id=payload.get("user_id", f"direct_{uuid.uuid4()}"),
            session_id=payload.get("session_id", f"direct_{uuid.uuid4()}"),
            message_type=MessageType(payload.get("message_type", "text")),
            content=payload.get("content", ""),
            metadata=payload.get("metadata", {}),
            timestamp=datetime.utcnow(),
            channel=ChannelType.DIRECT
        )
    
    def _normalize_generic_payload(self, payload: Dict[str, Any]) -> GovernorEvent:
        """Normalize generic webhook payload."""
        return GovernorEvent(
            user_id=payload.get("user_id", f"generic_{uuid.uuid4()}"),
            session_id=payload.get("session_id", f"generic_{uuid.uuid4()}"),
            message_type=MessageType.TEXT,
            content=payload.get("message", payload.get("content", "")),
            metadata=payload,
            timestamp=datetime.utcnow(),
            channel=ChannelType.WEBHOOK
        )
    
    async def _process_governor_event(self, event: GovernorEvent) -> None:
        """Process event through Governor state machine.
        
        This is the core processing function that runs the event through
        the complete Governor pipeline asynchronously.
        
        Args:
            event: Normalized GovernorEvent to process
        """
        try:
            # Get or create session state
            state = await self._get_or_create_state(event.user_id, event.session_id)
            
            # Get available tools for this user/context
            available_tools = tool_registry.list_tools()
            
            # Add event to conversation history
            if not hasattr(state, 'conversation_history'):
                state.conversation_history = []
            
            state.conversation_history.append({
                "role": "user",
                "content": event.content,
                "timestamp": event.timestamp.isoformat(),
                "channel": event.channel.value
            })
            
            # Assemble context for the LLM
            context_prompt = context_assembler.assemble_context(
                user_id=event.user_id,
                current_input=event.content,
                state=state,
                available_tools=list(available_tools)
            )
            
            # Create graph input
            graph_input = {
                "state": state,
                "event": event,
                "context_prompt": context_prompt,
                "available_tools": available_tools
            }
            
            # Process through LangGraph state machine
            result = await governor_graph.invoke(graph_input)
            
            # Extract response from result
            if isinstance(result, dict) and "response" in result:
                response = result["response"]
            else:
                # Fallback response
                response = GovernorResponse(
                    user_id=event.user_id,
                    session_id=event.session_id,
                    content="I received your message and am processing it.",
                    response_type=ResponseType.TEXT,
                    channel=event.channel,
                    metadata={"status": "processed"}
                )
            
            # Send response back to user
            await self._send_response(response, event.channel)
            
            # Store interaction in memory (triggers reflection job)
            try:
                if self.memory_manager:
                    await self.memory_manager.store_interaction(
                        user_id=event.user_id,
                        interaction={
                            "content": event.content,
                            "assistant_response": response.get("response", ""),
                            "timestamp": datetime.utcnow().isoformat(),
                            "session_id": event.session_id,
                            "metadata": event.metadata or {}
                        },
                        session_id=event.session_id
                    )
                    logger.debug(f"✅ Webhook interaction stored and reflection job queued for {event.user_id}")
            except Exception as e:
                logger.error(f"❌ Webhook memory storage failed for {event.user_id}: {e}")
            
            # Update session state
            state.total_turns += 1
            session_states[event.session_id] = state
            
            logger.info(f"Successfully processed event for {event.user_id}")
            
        except Exception as e:
            logger.error(f"Error processing governor event: {e}")
            # Send error response to user
            error_response = GovernorResponse(
                user_id=event.user_id,
                session_id=event.session_id,
                content="I encountered an error processing your request. Please try again.",
                response_type=ResponseType.ERROR,
                channel=event.channel,
                metadata={"error": str(e)}
            )
            await self._send_response(error_response, event.channel)
    
    async def _get_or_create_state(
        self, 
        user_id: str, 
        session_id: str
    ) -> GovernorState:
        """Get existing session state or create a new one.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Governor state for the session
        """
        if session_id in session_states:
            return session_states[session_id]
        
        # Create new state
        state = GovernorState(
            user_id=user_id,
            session_id=session_id,
            current_node=StateNode.IDLE
        )
        
        # Initialize state attributes
        state.total_turns = 0
        state.total_tools_executed = 0
        state.error_count = 0
        state.awaiting_confirmation = False
        state.pending_tools = []
        state.conversation_history = []
        state.created_at = datetime.utcnow()
        
        session_states[session_id] = state
        return state
    
    async def _send_response(
        self, 
        response: GovernorResponse, 
        channel: ChannelType
    ) -> None:
        """Send response back to the user through the appropriate channel.
        
        Args:
            response: Governor response to send
            channel: Communication channel to use
        """
        try:
            # Use the response handler to deliver the response
            success = await response_handler.deliver_response(response, channel)
            
            if success:
                logger.info(
                    f"Response delivered to {response.user_id} via {channel.value}: "
                    f"{response.content[:100]}..."
                )
            else:
                logger.warning(
                    f"Response delivery failed for {response.user_id} via {channel.value}, "
                    f"stored for polling"
                )
            
        except Exception as e:
            logger.error(f"Error sending response: {e}")


# Create global app instance
webhook_app = WebhookApp()
app = webhook_app.app


def run_server(host: str = "0.0.0.0", port: int = 8000, debug: bool = False) -> None:
    """Run the webhook server.
    
    Args:
        host: Host address to bind to
        port: Port number to listen on
        debug: Enable debug mode
    """
    uvicorn.run(
        "src.interfaces.webhooks.app:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )


if __name__ == "__main__":
    run_server(debug=True)