"""RabbitMQ producer for memory reflection jobs."""

import asyncio
import logging
from typing import Optional
import aio_pika
from aio_pika import Message, DeliveryMode
from datetime import datetime

from ...core.config import settings
from .schemas import MemoryReflectionJob

logger = logging.getLogger(__name__)

class QueueProducer:
    """RabbitMQ producer for memory reflection jobs."""
    
    def __init__(self):
        self.connection: Optional[aio_pika.Connection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self.exchange: Optional[aio_pika.Exchange] = None
        self._is_connected = False
    
    async def connect(self) -> None:
        """Initialize RabbitMQ connection and setup."""
        try:
            # Connect to RabbitMQ
            self.connection = await aio_pika.connect_robust(
                settings.rabbitmq_url,
                client_properties={"connection_name": "umbranet_producer"}
            )
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=100)  # Producer QoS
            
            # Declare exchange (topic type for wildcard routing)
            self.exchange = await self.channel.declare_exchange(
                settings.rabbitmq_exchange,
                aio_pika.ExchangeType.TOPIC,
                durable=True
            )
            
            # Declare main queue
            main_queue = await self.channel.declare_queue(
                settings.rabbitmq_queue,
                durable=True,
                arguments={
                    "x-dead-letter-exchange": f"{settings.rabbitmq_exchange}.dlx",
                    "x-dead-letter-routing-key": "failed",
                    "x-max-retries": settings.reflection_max_retries
                }
            )
            
            # Bind queue to exchange
            await main_queue.bind(self.exchange, "reflection.*")
            
            # Declare dead letter exchange and queue
            dlx_exchange = await self.channel.declare_exchange(
                f"{settings.rabbitmq_exchange}.dlx",
                aio_pika.ExchangeType.TOPIC,
                durable=True
            )
            
            dlx_queue = await self.channel.declare_queue(
                settings.rabbitmq_dead_letter_queue,
                durable=True
            )
            
            await dlx_queue.bind(dlx_exchange, "failed")
            
            self._is_connected = True
            logger.info("🐰 RabbitMQ producer connected and configured")
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close RabbitMQ connection."""
        if self.connection:
            await self.connection.close()
            self._is_connected = False
            logger.info("🐰 RabbitMQ producer disconnected")
    
    async def send_reflection_job(self, job: MemoryReflectionJob) -> bool:
        """Send memory reflection job to queue."""
        if not self._is_connected:
            await self.connect()
        
        try:
            # Create message
            message = Message(
                job.to_json().encode(),
                priority=job.priority.value,
                delivery_mode=DeliveryMode.PERSISTENT,  # Survive broker restart
                timestamp=datetime.utcnow(),
                message_id=job.job_id,
                headers={
                    "user_id": job.user_id,  # Put user_id in headers instead of message property
                    "retry_count": job.retry_count,
                    "max_retries": job.max_retries,
                    "created_at": job.created_at.isoformat()
                }
            )
            
            # Send to exchange with user-specific routing key
            routing_key = f"reflection.{job.user_id}"
            await self.exchange.publish(message, routing_key=routing_key)
            
            logger.debug(f"📤 Queued reflection job {job.job_id} for user {job.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send reflection job {job.job_id}: {e}")
            return False
    
    async def send_batch_jobs(self, jobs: list[MemoryReflectionJob]) -> int:
        """Send multiple reflection jobs efficiently."""
        if not self._is_connected:
            await self.connect()
        
        successful = 0
        try:
            async with self.channel.transaction():
                for job in jobs:
                    success = await self.send_reflection_job(job)
                    if success:
                        successful += 1
            
            logger.info(f"📤 Batch sent {successful}/{len(jobs)} reflection jobs")
            return successful
            
        except Exception as e:
            logger.error(f"Batch send failed: {e}")
            return 0
    
    async def health_check(self) -> dict:
        """Check producer health and queue status."""
        if not self._is_connected:
            return {"status": "disconnected", "healthy": False}
        
        try:
            # Simple connection check
            if self.connection and not self.connection.is_closed:
                return {
                    "status": "connected",
                    "healthy": True,
                    "connection_open": not self.connection.is_closed
                }
            else:
                return {"status": "connection_closed", "healthy": False}
        except Exception as e:
            return {"status": "error", "healthy": False, "error": str(e)}

# Global producer instance
_producer: Optional[QueueProducer] = None

async def get_queue_producer() -> QueueProducer:
    """Get or create global queue producer."""
    global _producer
    if _producer is None:
        _producer = QueueProducer()
        await _producer.connect()
    return _producer