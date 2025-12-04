"""RabbitMQ consumer for memory reflection jobs."""

import asyncio
import logging
from typing import Optional, Callable
import aio_pika
from aio_pika import IncomingMessage
from datetime import datetime, timedelta

from ...core.config import settings
from .schemas import MemoryReflectionJob

logger = logging.getLogger(__name__)

class QueueConsumer:
    """RabbitMQ consumer for memory reflection jobs."""
    
    def __init__(self, processor_callback: Callable[[MemoryReflectionJob], bool]):
        self.processor_callback = processor_callback
        self.connection: Optional[aio_pika.Connection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self.queue: Optional[aio_pika.Queue] = None
        self._is_consuming = False
        
        # Performance metrics
        self.processed_count = 0
        self.failed_count = 0
        self.start_time = datetime.utcnow()
    
    async def connect(self) -> None:
        """Connect to RabbitMQ and setup consumer."""
        try:
            self.connection = await aio_pika.connect_robust(
                settings.rabbitmq_url,
                client_properties={"connection_name": "umbranet_reflector"}
            )
            self.channel = await self.connection.channel()
            
            # Set QoS - process multiple messages concurrently but limit prefetch
            await self.channel.set_qos(
                prefetch_count=settings.reflection_workers * 2
            )
            
            # Get the queue (should already exist from producer with same arguments)
            self.queue = await self.channel.declare_queue(
                settings.rabbitmq_queue,
                durable=True,
                arguments={
                    "x-dead-letter-exchange": f"{settings.rabbitmq_exchange}.dlx",
                    "x-dead-letter-routing-key": "failed",
                    "x-max-retries": settings.reflection_max_retries
                }
            )
            
            logger.info(f"🐰 RabbitMQ consumer connected to queue: {settings.rabbitmq_queue}")
            
        except Exception as e:
            logger.error(f"Failed to connect consumer to RabbitMQ: {e}")
            raise
    
    async def start_consuming(self) -> None:
        """Start consuming messages from the queue."""
        if not self.connection:
            await self.connect()
        
        # Start consuming with callback
        await self.queue.consume(
            self._process_message,
            consumer_tag="memory_reflector"
        )
        
        self._is_consuming = True
        self.start_time = datetime.utcnow()
        logger.info(f"🔄 Started consuming reflection jobs with {settings.reflection_workers} workers")
    
    async def stop_consuming(self) -> None:
        """Stop consuming and close connection."""
        if self.connection:
            await self.connection.close()
        self._is_consuming = False
        
        # Log final stats
        runtime = datetime.utcnow() - self.start_time
        logger.info(
            f"🔄 Stopped consuming. Processed: {self.processed_count}, "
            f"Failed: {self.failed_count}, Runtime: {runtime}"
        )
    
    async def _process_message(self, message: IncomingMessage) -> None:
        """Process individual reflection job message."""
        start_time = datetime.utcnow()
        
        try:
            async with message.process(requeue=False):
                # Parse the job from message body
                job = MemoryReflectionJob.from_json(message.body.decode())
                
                logger.debug(
                    f"🔄 Processing reflection job {job.job_id} for user {job.user_id} "
                    f"(attempt {job.retry_count + 1}/{job.max_retries + 1})"
                )
                
                # Call the processor callback
                success = await self.processor_callback(job)
                
                if success:
                    self.processed_count += 1
                    processing_time = (datetime.utcnow() - start_time).total_seconds()
                    logger.info(
                        f"✅ Completed reflection job {job.job_id} in {processing_time:.2f}s"
                    )
                    # Message is acknowledged automatically by context manager
                else:
                    # Handle failure with retry logic
                    await self._handle_job_failure(message, job)
                    
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
            self.failed_count += 1
            # Message will be nacked and potentially requeued
            raise
    
    async def _handle_job_failure(self, message: IncomingMessage, job: MemoryReflectionJob) -> None:
        """Handle failed job processing with retry logic."""
        self.failed_count += 1
        
        if job.should_retry():
            # Increment retry count and requeue with delay
            job.increment_retry()
            
            # Calculate exponential backoff delay
            delay = settings.reflection_retry_delay * (2 ** job.retry_count)
            
            logger.warning(
                f"⚠️  Job {job.job_id} failed, retrying in {delay}s "
                f"(attempt {job.retry_count}/{job.max_retries})"
            )
            
            # Schedule retry by republishing with delay
            await asyncio.sleep(delay)
            
            # Republish to queue
            retry_message = aio_pika.Message(
                job.to_json().encode(),
                priority=job.priority.value,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                headers=message.headers or {}
            )
            
            exchange = await self.channel.get_exchange(settings.rabbitmq_exchange, ensure=False)
            await exchange.publish(
                retry_message, 
                routing_key=f"reflection.{job.user_id}"
            )
        else:
            # Max retries exceeded - send to dead letter queue
            logger.error(
                f"💀 Job {job.job_id} exceeded max retries, sending to dead letter queue"
            )
            # The dead letter queue setup will handle this automatically
    
    def get_stats(self) -> dict:
        """Get consumer performance statistics."""
        runtime = datetime.utcnow() - self.start_time
        rate = self.processed_count / max(runtime.total_seconds(), 1)
        
        return {
            "is_consuming": self._is_consuming,
            "processed_count": self.processed_count,
            "failed_count": self.failed_count,
            "success_rate": self.processed_count / max(self.processed_count + self.failed_count, 1),
            "processing_rate_per_second": rate,
            "runtime_seconds": runtime.total_seconds()
        }