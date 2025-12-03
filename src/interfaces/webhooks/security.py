"""Security middleware and authentication for webhook endpoints.

This module provides security features including webhook signature verification,
API key authentication, rate limiting, and request validation.
"""

import hashlib
import hmac
import logging
import time
import os
from typing import Dict, Any, Optional, Set
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict

from fastapi import HTTPException, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    pass


class InvalidSignature(Exception):
    """Exception raised when webhook signature validation fails."""
    pass


class RateLimiter:
    """Token bucket rate limiter for API endpoints."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        """Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed in the time window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = defaultdict(list)
        self.blocked_ips: Set[str] = set()
        self.block_duration = 300  # 5 minutes
        self.block_times: Dict[str, datetime] = {}
    
    def is_allowed(self, client_ip: str) -> bool:
        """Check if request from client IP is allowed.
        
        Args:
            client_ip: Client IP address
            
        Returns:
            True if request is allowed, False if rate limited
            
        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        now = time.time()
        
        # Check if IP is currently blocked
        if client_ip in self.blocked_ips:
            block_time = self.block_times.get(client_ip)
            if block_time and (datetime.utcnow() - block_time).seconds < self.block_duration:
                raise RateLimitExceeded(f"IP {client_ip} is temporarily blocked")
            else:
                # Unblock IP after block duration
                self.blocked_ips.discard(client_ip)
                self.block_times.pop(client_ip, None)
        
        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if now - req_time < self.window_seconds
        ]
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.max_requests:
            # Block IP for repeated violations
            self.blocked_ips.add(client_ip)
            self.block_times[client_ip] = datetime.utcnow()
            
            logger.warning(f"Rate limit exceeded for IP {client_ip}, blocking for {self.block_duration} seconds")
            raise RateLimitExceeded(f"Rate limit exceeded for IP {client_ip}")
        
        # Add current request
        self.requests[client_ip].append(now)
        return True


class WebhookSecurity:
    """Security handler for webhook endpoints."""
    
    def __init__(self):
        """Initialize webhook security."""
        self.rate_limiter = RateLimiter(max_requests=100, window_seconds=3600)  # 100 req/hour
        self.api_keys = self._load_api_keys()
        self.webhook_secrets = self._load_webhook_secrets()
        self.bearer_security = HTTPBearer(auto_error=False)
        
    def _load_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """Load API keys from environment or configuration.
        
        Returns:
            Dictionary of API keys with metadata
        """
        api_keys = {}
        
        # Load from environment variables
        master_key = os.getenv("GOVERNOR_API_KEY")
        if master_key:
            api_keys[master_key] = {
                "name": "master",
                "permissions": ["read", "write", "admin"],
                "created_at": datetime.utcnow().isoformat()
            }
        
        # Load additional keys from environment
        for i in range(1, 11):  # Support up to 10 additional keys
            key = os.getenv(f"GOVERNOR_API_KEY_{i}")
            if key:
                api_keys[key] = {
                    "name": f"api_key_{i}",
                    "permissions": ["read", "write"],
                    "created_at": datetime.utcnow().isoformat()
                }
        
        if not api_keys:
            logger.warning("No API keys configured, authentication disabled")
        
        return api_keys
    
    def _load_webhook_secrets(self) -> Dict[str, str]:
        """Load webhook secrets for signature verification.
        
        Returns:
            Dictionary mapping channel types to their secrets
        """
        return {
            "telegram": os.getenv("TELEGRAM_WEBHOOK_SECRET", ""),
            "whatsapp": os.getenv("WHATSAPP_WEBHOOK_SECRET", ""),
            "github": os.getenv("GITHUB_WEBHOOK_SECRET", ""),
            "generic": os.getenv("WEBHOOK_SECRET", "")
        }
    
    async def verify_rate_limit(self, request: Request) -> bool:
        """Verify request against rate limits.
        
        Args:
            request: FastAPI request object
            
        Returns:
            True if request is allowed
            
        Raises:
            HTTPException: If rate limit exceeded
        """
        try:
            # Get client IP
            client_ip = self._get_client_ip(request)
            
            # Check rate limit
            self.rate_limiter.is_allowed(client_ip)
            return True
            
        except RateLimitExceeded as e:
            logger.warning(f"Rate limit exceeded: {e}")
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": "3600"}
            )
    
    async def verify_api_key(
        self, 
        authorization: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Verify API key authentication.
        
        Args:
            authorization: Authorization header value
            
        Returns:
            API key metadata if valid, None otherwise
            
        Raises:
            HTTPException: If authentication is required but invalid
        """
        if not self.api_keys:
            # No API keys configured, skip authentication
            return None
        
        if not authorization:
            raise HTTPException(
                status_code=401,
                detail="API key required",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Extract token from Authorization header
        if authorization.startswith("Bearer "):
            token = authorization[7:]
        else:
            token = authorization
        
        # Verify token
        if token in self.api_keys:
            logger.info(f"Valid API key used: {self.api_keys[token]['name']}")
            return self.api_keys[token]
        
        logger.warning(f"Invalid API key attempted: {token[:10]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    async def verify_webhook_signature(
        self,
        request: Request,
        payload_body: bytes,
        signature_header: Optional[str] = None,
        channel_type: str = "generic"
    ) -> bool:
        """Verify webhook signature.
        
        Args:
            request: FastAPI request object
            payload_body: Raw request body
            signature_header: Signature from webhook headers
            channel_type: Type of webhook channel
            
        Returns:
            True if signature is valid
            
        Raises:
            HTTPException: If signature verification fails
        """
        secret = self.webhook_secrets.get(channel_type, "")
        if not secret:
            # No secret configured, skip verification
            logger.info(f"No webhook secret configured for {channel_type}, skipping verification")
            return True
        
        if not signature_header:
            logger.warning(f"No signature header provided for {channel_type} webhook")
            raise HTTPException(
                status_code=401,
                detail="Webhook signature required"
            )
        
        try:
            if channel_type == "telegram":
                return self._verify_telegram_signature(payload_body, signature_header, secret)
            elif channel_type == "github":
                return self._verify_github_signature(payload_body, signature_header, secret)
            else:
                return self._verify_generic_signature(payload_body, signature_header, secret)
                
        except InvalidSignature as e:
            logger.warning(f"Webhook signature verification failed: {e}")
            raise HTTPException(
                status_code=401,
                detail="Invalid webhook signature"
            )
    
    def _verify_telegram_signature(
        self, 
        payload: bytes, 
        signature: str, 
        secret: str
    ) -> bool:
        """Verify Telegram webhook signature.
        
        Args:
            payload: Raw payload bytes
            signature: Signature from X-Telegram-Bot-Api-Secret-Token header
            secret: Webhook secret
            
        Returns:
            True if signature is valid
        """
        # Telegram uses simple token comparison
        if signature == secret:
            return True
        
        raise InvalidSignature("Telegram signature mismatch")
    
    def _verify_github_signature(
        self, 
        payload: bytes, 
        signature: str, 
        secret: str
    ) -> bool:
        """Verify GitHub webhook signature.
        
        Args:
            payload: Raw payload bytes
            signature: Signature from X-Hub-Signature-256 header
            secret: Webhook secret
            
        Returns:
            True if signature is valid
        """
        # GitHub uses HMAC-SHA256
        expected_signature = hmac.new(
            secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        # GitHub sends signature as "sha256=<hash>"
        if signature.startswith("sha256="):
            provided_signature = signature[7:]
        else:
            provided_signature = signature
        
        if hmac.compare_digest(expected_signature, provided_signature):
            return True
        
        raise InvalidSignature("GitHub signature mismatch")
    
    def _verify_generic_signature(
        self, 
        payload: bytes, 
        signature: str, 
        secret: str
    ) -> bool:
        """Verify generic webhook signature using HMAC-SHA256.
        
        Args:
            payload: Raw payload bytes
            signature: Signature from request header
            secret: Webhook secret
            
        Returns:
            True if signature is valid
        """
        expected_signature = hmac.new(
            secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        if hmac.compare_digest(expected_signature, signature):
            return True
        
        raise InvalidSignature("Generic signature mismatch")
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Client IP address
        """
        # Check for forwarded headers first (for reverse proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in case of multiple proxies
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection IP
        return request.client.host if request.client else "unknown"
    
    async def validate_request_size(self, request: Request, max_size: int = 1024 * 1024) -> bool:
        """Validate request size to prevent DoS attacks.
        
        Args:
            request: FastAPI request object
            max_size: Maximum allowed request size in bytes
            
        Returns:
            True if request size is valid
            
        Raises:
            HTTPException: If request is too large
        """
        content_length = request.headers.get("content-length")
        if content_length:
            size = int(content_length)
            if size > max_size:
                logger.warning(f"Request too large: {size} bytes (max: {max_size})")
                raise HTTPException(
                    status_code=413,
                    detail=f"Request too large (max: {max_size} bytes)"
                )
        
        return True
    
    def is_valid_user_agent(self, user_agent: Optional[str]) -> bool:
        """Check if User-Agent header is from a known/allowed source.
        
        Args:
            user_agent: User-Agent header value
            
        Returns:
            True if User-Agent is valid/allowed
        """
        if not user_agent:
            return False
        
        # Allow known webhook sources
        allowed_patterns = [
            "telegram",
            "whatsapp",
            "github-hookshot",
            "postman",
            "curl",
            "httpie",
            "headless-governor"
        ]
        
        user_agent_lower = user_agent.lower()
        return any(pattern in user_agent_lower for pattern in allowed_patterns)


# Global security instance
webhook_security = WebhookSecurity()