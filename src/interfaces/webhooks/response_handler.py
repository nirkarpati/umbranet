"""Response delivery system for webhook interface.

This module handles sending responses back to users through various channels
including Telegram, WhatsApp, direct API callbacks, and webhook deliveries.
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import aiohttp
import json

from ...core.domain.events import GovernorResponse, ChannelType, ResponseType

logger = logging.getLogger(__name__)


class ResponseDeliveryError(Exception):
    """Exception raised when response delivery fails."""
    pass


class ResponseHandler:
    """Handles delivery of responses through various communication channels."""
    
    def __init__(self):
        """Initialize the response handler with channel configurations."""
        self.telegram_config = {
            "bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
            "api_url": "https://api.telegram.org/bot"
        }
        
        self.whatsapp_config = {
            "access_token": os.getenv("WHATSAPP_ACCESS_TOKEN"),
            "phone_number_id": os.getenv("WHATSAPP_PHONE_NUMBER_ID"),
            "api_url": "https://graph.facebook.com/v18.0"
        }
        
        self.webhook_config = {
            "callback_urls": {},  # Will be populated from user preferences
            "default_timeout": 30,
            "max_retries": 3
        }
        
        # Response store for polling-based channels
        self.response_store: Dict[str, List[GovernorResponse]] = {}
    
    async def deliver_response(
        self, 
        response: GovernorResponse, 
        channel: ChannelType,
        channel_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Deliver response through the specified channel.
        
        Args:
            response: The response to deliver
            channel: Target communication channel
            channel_metadata: Additional channel-specific metadata
            
        Returns:
            True if delivery successful, False otherwise
            
        Raises:
            ResponseDeliveryError: If delivery fails critically
        """
        try:
            if channel == ChannelType.TELEGRAM:
                return await self._deliver_telegram_response(response, channel_metadata or {})
            elif channel == ChannelType.WHATSAPP:
                return await self._deliver_whatsapp_response(response, channel_metadata or {})
            elif channel == ChannelType.DIRECT:
                return await self._deliver_direct_response(response, channel_metadata or {})
            elif channel == ChannelType.WEBHOOK:
                return await self._deliver_webhook_response(response, channel_metadata or {})
            else:
                # Fallback to storing response for polling
                return await self._store_response_for_polling(response)
                
        except Exception as e:
            logger.error(f"Response delivery failed for {response.session_id}: {e}")
            raise ResponseDeliveryError(f"Failed to deliver response: {e}")
    
    async def _deliver_telegram_response(
        self, 
        response: GovernorResponse, 
        metadata: Dict[str, Any]
    ) -> bool:
        """Deliver response via Telegram Bot API.
        
        Args:
            response: Response to deliver
            metadata: Telegram-specific metadata (chat_id, etc.)
            
        Returns:
            True if successful
        """
        if not self.telegram_config["bot_token"]:
            logger.warning("Telegram bot token not configured, storing response")
            return await self._store_response_for_polling(response)
        
        # Extract chat ID from user_id or metadata
        chat_id = metadata.get("chat_id")
        if not chat_id and response.user_id.startswith("telegram_"):
            chat_id = response.user_id.replace("telegram_", "")
        
        if not chat_id:
            logger.error("No Telegram chat ID available for response delivery")
            return False
        
        url = f"{self.telegram_config['api_url']}{self.telegram_config['bot_token']}/sendMessage"
        
        payload = {
            "chat_id": chat_id,
            "text": response.content
        }
        
        # Add formatting based on response type
        if response.response_type == ResponseType.ERROR:
            payload["text"] = f"❌ Error: {response.content}"
        elif response.response_type == ResponseType.CONFIRMATION_REQUEST:
            payload["text"] = f"⚠️ Confirmation Required: {response.content}"
            # Add inline keyboard for yes/no
            payload["reply_markup"] = {
                "inline_keyboard": [[
                    {"text": "✅ Yes", "callback_data": "confirm_yes"},
                    {"text": "❌ No", "callback_data": "confirm_no"}
                ]]
            }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        logger.info(f"Telegram response delivered to {chat_id}")
                        return True
                    else:
                        error_text = await resp.text()
                        logger.error(f"Telegram API error: {resp.status} - {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Telegram delivery error: {e}")
            return False
    
    async def _deliver_whatsapp_response(
        self, 
        response: GovernorResponse, 
        metadata: Dict[str, Any]
    ) -> bool:
        """Deliver response via WhatsApp Business API.
        
        Args:
            response: Response to deliver
            metadata: WhatsApp-specific metadata
            
        Returns:
            True if successful
        """
        if not self.whatsapp_config["access_token"]:
            logger.warning("WhatsApp access token not configured, storing response")
            return await self._store_response_for_polling(response)
        
        # Extract phone number from user_id or metadata
        phone_number = metadata.get("phone_number")
        if not phone_number and response.user_id.startswith("whatsapp_"):
            phone_number = response.user_id.replace("whatsapp_", "")
        
        if not phone_number:
            logger.error("No WhatsApp phone number available for response delivery")
            return False
        
        url = f"{self.whatsapp_config['api_url']}/{self.whatsapp_config['phone_number_id']}/messages"
        
        headers = {
            "Authorization": f"Bearer {self.whatsapp_config['access_token']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messaging_product": "whatsapp",
            "to": phone_number,
            "type": "text",
            "text": {"body": response.content}
        }
        
        # Add formatting for special response types
        if response.response_type == ResponseType.ERROR:
            payload["text"]["body"] = f"❌ Error: {response.content}"
        elif response.response_type == ResponseType.CONFIRMATION_REQUEST:
            payload["text"]["body"] = f"⚠️ Confirmation Required: {response.content}\n\nReply with 'yes' or 'no'"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        logger.info(f"WhatsApp response delivered to {phone_number}")
                        return True
                    else:
                        error_text = await resp.text()
                        logger.error(f"WhatsApp API error: {resp.status} - {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"WhatsApp delivery error: {e}")
            return False
    
    async def _deliver_direct_response(
        self, 
        response: GovernorResponse, 
        metadata: Dict[str, Any]
    ) -> bool:
        """Deliver response via direct API callback or storage.
        
        Args:
            response: Response to deliver
            metadata: Direct channel metadata
            
        Returns:
            True if successful
        """
        callback_url = metadata.get("callback_url")
        
        if callback_url:
            # Deliver via webhook callback
            return await self._send_webhook_callback(response, callback_url)
        else:
            # Store for polling
            return await self._store_response_for_polling(response)
    
    async def _deliver_webhook_response(
        self, 
        response: GovernorResponse, 
        metadata: Dict[str, Any]
    ) -> bool:
        """Deliver response via generic webhook.
        
        Args:
            response: Response to deliver
            metadata: Webhook metadata
            
        Returns:
            True if successful
        """
        webhook_url = metadata.get("webhook_url")
        
        if not webhook_url:
            logger.warning("No webhook URL provided, storing response")
            return await self._store_response_for_polling(response)
        
        return await self._send_webhook_callback(response, webhook_url)
    
    async def _send_webhook_callback(
        self, 
        response: GovernorResponse, 
        callback_url: str
    ) -> bool:
        """Send response to a webhook callback URL.
        
        Args:
            response: Response to send
            callback_url: Target webhook URL
            
        Returns:
            True if successful
        """
        payload = {
            "user_id": response.user_id,
            "session_id": response.session_id,
            "content": response.content,
            "response_type": response.response_type.value,
            "metadata": response.metadata,
            "timestamp": response.timestamp.isoformat()
        }
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Headless-Governor-Webhook/1.0"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    callback_url, 
                    json=payload, 
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.webhook_config["default_timeout"])
                ) as resp:
                    if 200 <= resp.status < 300:
                        logger.info(f"Webhook callback successful: {callback_url}")
                        return True
                    else:
                        error_text = await resp.text()
                        logger.error(f"Webhook callback failed: {resp.status} - {error_text}")
                        return False
                        
        except asyncio.TimeoutError:
            logger.error(f"Webhook callback timeout: {callback_url}")
            return False
        except Exception as e:
            logger.error(f"Webhook callback error: {e}")
            return False
    
    async def _store_response_for_polling(self, response: GovernorResponse) -> bool:
        """Store response for polling-based retrieval.
        
        Args:
            response: Response to store
            
        Returns:
            True (always successful for storage)
        """
        session_id = response.session_id
        
        if session_id not in self.response_store:
            self.response_store[session_id] = []
        
        self.response_store[session_id].append(response)
        
        # Keep only last 50 responses per session
        if len(self.response_store[session_id]) > 50:
            self.response_store[session_id] = self.response_store[session_id][-50:]
        
        logger.info(f"Response stored for polling: {session_id}")
        return True
    
    def get_stored_responses(
        self, 
        session_id: str, 
        since: Optional[datetime] = None
    ) -> List[GovernorResponse]:
        """Retrieve stored responses for a session.
        
        Args:
            session_id: Session identifier
            since: Only return responses after this timestamp
            
        Returns:
            List of stored responses
        """
        responses = self.response_store.get(session_id, [])
        
        if since:
            responses = [
                r for r in responses 
                if r.timestamp > since
            ]
        
        return responses
    
    def clear_stored_responses(self, session_id: str) -> None:
        """Clear stored responses for a session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.response_store:
            del self.response_store[session_id]
            logger.info(f"Cleared stored responses for {session_id}")
    
    async def test_channel_connectivity(self, channel: ChannelType) -> Dict[str, Any]:
        """Test connectivity to a specific channel.
        
        Args:
            channel: Channel to test
            
        Returns:
            Test results dictionary
        """
        test_result = {
            "channel": channel.value,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "unknown",
            "details": {}
        }
        
        try:
            if channel == ChannelType.TELEGRAM:
                if not self.telegram_config["bot_token"]:
                    test_result["status"] = "not_configured"
                    test_result["details"]["error"] = "Bot token not configured"
                else:
                    # Test with getMe API call
                    url = f"{self.telegram_config['api_url']}{self.telegram_config['bot_token']}/getMe"
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url) as resp:
                            if resp.status == 200:
                                bot_info = await resp.json()
                                test_result["status"] = "connected"
                                test_result["details"]["bot_info"] = bot_info.get("result", {})
                            else:
                                test_result["status"] = "error"
                                test_result["details"]["error"] = await resp.text()
            
            elif channel == ChannelType.WHATSAPP:
                if not self.whatsapp_config["access_token"]:
                    test_result["status"] = "not_configured"
                    test_result["details"]["error"] = "Access token not configured"
                else:
                    test_result["status"] = "configured"
                    test_result["details"]["note"] = "WhatsApp connectivity requires phone number ID"
            
            else:
                test_result["status"] = "not_implemented"
                test_result["details"]["note"] = f"Connectivity test not implemented for {channel.value}"
                
        except Exception as e:
            test_result["status"] = "error"
            test_result["details"]["error"] = str(e)
        
        return test_result


# Global response handler instance
response_handler = ResponseHandler()