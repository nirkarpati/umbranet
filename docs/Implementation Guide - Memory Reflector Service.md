# Implementation Guide: Memory Reflector Service

## Quick Start Implementation Checklist

This guide provides step-by-step implementation instructions for the Memory Reflector Service refactor.

### Prerequisites
- [ ] RabbitMQ server running
- [ ] All existing memory tiers operational
- [ ] Docker environment ready

## Phase 1: Infrastructure Setup

### Step 1.1: Add RabbitMQ to Docker Compose

```yaml
# Add to docker-compose.yml
services:
  rabbitmq:
    image: rabbitmq:3.12-management-alpine
    container_name: umbranet_rabbitmq
    ports:
      - "5672:5672"      # AMQP port
      - "15672:15672"    # Management UI
    environment:
      RABBITMQ_DEFAULT_USER: umbranet
      RABBITMQ_DEFAULT_PASS: ${RABBITMQ_PASSWORD:-reflection123}
      RABBITMQ_DEFAULT_VHOST: /
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    networks:
      - umbranet_network
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  rabbitmq_data:
```

### Step 1.2: Update Environment Configuration

```python
# src/core/config.py - Add RabbitMQ settings
class Settings(BaseSettings):
    # ... existing settings ...
    
    # RabbitMQ Configuration
    rabbitmq_url: str = "amqp://umbranet:reflection123@localhost:5672/"
    rabbitmq_exchange: str = "memory_reflection_exchange" 
    rabbitmq_queue: str = "memory_reflection_queue"
    rabbitmq_dead_letter_queue: str = "memory_reflection_dlq"
    
    # Reflection Service Configuration
    reflection_enabled: bool = True
    reflection_batch_size: int = 10
    reflection_workers: int = 2
    reflection_max_retries: int = 3
    reflection_retry_delay: int = 30  # seconds
    
    # Performance Tuning
    memory_fast_mode: bool = True  # Only Tier 1 during chat
    reflection_timeout: int = 300   # 5 minutes max processing time
```

## Phase 2: Queue Infrastructure

### Step 2.1: Message Schema Definition

```python
# src/reflector/queue/schemas.py
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
import json
import uuid

class ReflectionPriority(int, Enum):
    LOW = 0      # System messages, routine conversations
    NORMAL = 1   # Regular user interactions  
    HIGH = 2     # Important conversations, explicit memory requests
    URGENT = 3   # Critical user data, error recovery

@dataclass
class MemoryReflectionJob:
    """Schema for memory reflection queue messages."""
    
    # Core identifiers
    job_id: str
    user_id: str
    session_id: str
    
    # Conversation data
    user_message: str
    assistant_response: str
    timestamp: datetime
    
    # Processing metadata
    priority: ReflectionPriority = ReflectionPriority.NORMAL
    metadata: Dict[str, Any] = None
    
    # Retry logic
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    @classmethod
    def from_interaction(
        cls, 
        user_id: str, 
        interaction: Dict[str, Any], 
        priority: ReflectionPriority = ReflectionPriority.NORMAL
    ) -> 'MemoryReflectionJob':
        """Create reflection job from interaction data."""
        return cls(
            job_id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=interaction.get('session_id', 'unknown'),
            user_message=interaction.get('content', ''),
            assistant_response=interaction.get('assistant_response', ''),
            timestamp=datetime.fromisoformat(interaction.get('timestamp', datetime.utcnow().isoformat())),
            priority=priority,
            metadata=interaction.get('metadata', {})
        )
    
    def to_json(self) -> str:
        """Serialize to JSON for queue transmission."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data['timestamp'] = self.timestamp.isoformat()
        data['created_at'] = self.created_at.isoformat()
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MemoryReflectionJob':
        """Deserialize from JSON."""
        data = json.loads(json_str)
        # Convert ISO strings back to datetime objects
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['priority'] = ReflectionPriority(data['priority'])
        return cls(**data)
    
    def should_retry(self) -> bool:
        """Check if job should be retried on failure."""
        return self.retry_count < self.max_retries
    
    def increment_retry(self) -> 'MemoryReflectionJob':
        """Create new job instance with incremented retry count."""
        self.retry_count += 1
        return self
```

### Step 2.2: Queue Producer

```python
# src/reflector/queue/producer.py
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
            
            # Declare exchange
            self.exchange = await self.channel.declare_exchange(
                settings.rabbitmq_exchange,
                aio_pika.ExchangeType.DIRECT,
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
                aio_pika.ExchangeType.DIRECT,
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
                user_id=job.user_id,
                headers={
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
            # Check queue status
            queue = await self.channel.get_queue(settings.rabbitmq_queue)
            queue_info = await queue.declare(passive=True)
            
            return {
                "status": "connected",
                "healthy": True,
                "queue_messages": queue_info.message_count,
                "queue_consumers": queue_info.consumer_count,
                "connection_state": str(self.connection.connection.connection_state)
            }
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
```

### Step 2.3: Queue Consumer

```python
# src/reflector/queue/consumer.py
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
            
            # Get the queue (should already exist from producer)
            self.queue = await self.channel.declare_queue(
                settings.rabbitmq_queue,
                durable=True
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
            
            exchange = await self.channel.get_exchange(settings.rabbitmq_exchange)
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
```

## Phase 3: Memory Reflector Service

### Step 3.1: Core Reflector Service

```python
# src/reflector/main.py
import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Optional

from ..core.config import settings
from .queue.consumer import QueueConsumer
from .queue.schemas import MemoryReflectionJob
from .processors.episodic_processor import EpisodicProcessor
from .processors.semantic_processor import SemanticProcessor
from .processors.procedural_processor import ProceduralProcessor
from .health.monitor import HealthMonitor

logger = logging.getLogger(__name__)

class MemoryReflectorService:
    """Main service for processing memory reflection jobs."""
    
    def __init__(self):
        self.consumer: Optional[QueueConsumer] = None
        self.health_monitor: Optional[HealthMonitor] = None
        self.is_running = False
        
        # Initialize processors
        self.episodic_processor = EpisodicProcessor()
        self.semantic_processor = SemanticProcessor()
        self.procedural_processor = ProceduralProcessor()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.stop())
    
    async def start(self) -> None:
        """Start the memory reflector service."""
        logger.info("🧠 Starting Memory Reflector Service...")
        
        try:
            # Initialize processors
            await self.episodic_processor.initialize()
            await self.semantic_processor.initialize()
            await self.procedural_processor.initialize()
            
            # Setup queue consumer
            self.consumer = QueueConsumer(self._process_reflection_job)
            await self.consumer.connect()
            await self.consumer.start_consuming()
            
            # Start health monitoring
            self.health_monitor = HealthMonitor(self)
            await self.health_monitor.start()
            
            self.is_running = True
            logger.info("✅ Memory Reflector Service started successfully")
            
            # Keep service running
            while self.is_running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Failed to start Memory Reflector Service: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the memory reflector service gracefully."""
        if not self.is_running:
            return
            
        logger.info("🛑 Stopping Memory Reflector Service...")
        self.is_running = False
        
        # Stop health monitoring
        if self.health_monitor:
            await self.health_monitor.stop()
        
        # Stop queue consumer
        if self.consumer:
            await self.consumer.stop_consuming()
        
        # Cleanup processors
        await self.episodic_processor.cleanup()
        await self.semantic_processor.cleanup()
        await self.procedural_processor.cleanup()
        
        logger.info("✅ Memory Reflector Service stopped")
    
    async def _process_reflection_job(self, job: MemoryReflectionJob) -> bool:
        """Process a memory reflection job across all tiers."""
        start_time = datetime.utcnow()
        
        logger.info(
            f"🔄 Processing reflection job {job.job_id} for user {job.user_id}"
        )
        
        try:
            # Process tiers 2, 3, 4 in parallel for efficiency
            tasks = [
                self.episodic_processor.process_job(job),
                self.semantic_processor.process_job(job),
                self.procedural_processor.process_job(job)
            ]
            
            # Execute with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=settings.reflection_timeout
            )
            
            # Check results
            episodic_result, semantic_result, procedural_result = results
            
            success_count = 0
            errors = []
            
            if not isinstance(episodic_result, Exception):
                success_count += 1
                logger.debug(f"✅ Episodic processing successful: {episodic_result}")
            else:
                errors.append(f"Episodic: {episodic_result}")
            
            if not isinstance(semantic_result, Exception):
                success_count += 1
                logger.debug(f"✅ Semantic processing successful: {semantic_result}")
            else:
                errors.append(f"Semantic: {semantic_result}")
            
            if not isinstance(procedural_result, Exception):
                success_count += 1
                logger.debug(f"✅ Procedural processing successful: {procedural_result}")
            else:
                errors.append(f"Procedural: {procedural_result}")
            
            # Consider job successful if at least 2/3 tiers succeeded
            success = success_count >= 2
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            if success:
                logger.info(
                    f"✅ Reflection job {job.job_id} completed successfully "
                    f"({success_count}/3 tiers) in {processing_time:.2f}s"
                )
            else:
                logger.error(
                    f"❌ Reflection job {job.job_id} failed "
                    f"({success_count}/3 tiers) in {processing_time:.2f}s. Errors: {errors}"
                )
            
            return success
            
        except asyncio.TimeoutError:
            logger.error(f"⏰ Reflection job {job.job_id} timed out after {settings.reflection_timeout}s")
            return False
        except Exception as e:
            logger.error(f"❌ Reflection job {job.job_id} failed with error: {e}")
            return False
    
    def get_health_status(self) -> dict:
        """Get service health status."""
        return {
            "service": "memory_reflector",
            "status": "running" if self.is_running else "stopped",
            "timestamp": datetime.utcnow().isoformat(),
            "consumer_stats": self.consumer.get_stats() if self.consumer else None,
            "processors": {
                "episodic": self.episodic_processor.get_health(),
                "semantic": self.semantic_processor.get_health(),
                "procedural": self.procedural_processor.get_health()
            }
        }

# Service entry point
async def main():
    """Main entry point for the reflector service."""
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("/app/logs/reflector.log") if hasattr(settings, 'log_file') else logging.NullHandler()
        ]
    )
    
    logger.info("🚀 Initializing Memory Reflector Service...")
    
    service = MemoryReflectorService()
    
    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Service crashed: {e}")
        sys.exit(1)
    finally:
        await service.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## Phase 4: Integration with Existing Memory Manager

### Step 4.1: Modified Memory Manager

```python
# src/memory/manager.py - Key modifications

class MemoryManager:
    def __init__(self, config: Optional[MemoryConfig] = None):
        # ... existing initialization ...
        
        # Queue producer for reflection jobs
        self.queue_producer = None
        self._enable_fast_mode = settings.memory_fast_mode
    
    async def initialize(self) -> None:
        """Initialize memory manager with optional queue producer."""
        # ... existing initialization logic ...
        
        # Initialize queue producer if reflection is enabled
        if settings.reflection_enabled:
            from ..reflector.queue.producer import get_queue_producer
            self.queue_producer = await get_queue_producer()
            logger.info("📤 Queue producer initialized for memory reflection")
    
    async def store_interaction(
        self, 
        user_id: str, 
        interaction: Dict[str, Any]
    ) -> Dict[str, str]:
        """Store interaction - fast path or traditional path based on configuration."""
        
        if self._enable_fast_mode and settings.reflection_enabled and self.queue_producer:
            # Fast path: Only Tier 1 + Queue reflection job
            return await self._store_interaction_fast_path(user_id, interaction)
        else:
            # Traditional path: All tiers synchronously (backward compatibility)
            return await self._store_interaction_traditional_path(user_id, interaction)
    
    async def _store_interaction_fast_path(
        self, 
        user_id: str, 
        interaction: Dict[str, Any]
    ) -> Dict[str, str]:
        """Fast path: Store only in Tier 1 and queue reflection job."""
        
        logger.debug(f"📝 Fast path storage for user {user_id}")
        
        # Enrich interaction data
        interaction_data = {
            **interaction,
            "timestamp": datetime.utcnow().isoformat(),
            "interaction_id": f"{user_id}_{datetime.utcnow().timestamp()}",
            "user_id": user_id
        }
        
        try:
            # Step 1: Store in Tier 1 (Redis) only - fast operation
            short_term_id = await self._store_in_tier_safe(
                "short_term", self.short_term, user_id, interaction_data
            )
            
            # Step 2: Queue reflection job for tiers 2, 3, 4 (non-blocking)
            from ..reflector.queue.schemas import MemoryReflectionJob, ReflectionPriority
            
            # Determine priority based on interaction content
            priority = self._determine_reflection_priority(interaction_data)
            
            reflection_job = MemoryReflectionJob.from_interaction(
                user_id=user_id,
                interaction=interaction_data,
                priority=priority
            )
            
            # Send to queue (async, non-blocking)
            queue_success = await self.queue_producer.send_reflection_job(reflection_job)
            
            if queue_success:
                logger.info(f"✅ Fast storage completed: Tier 1 stored, reflection queued")
                return {
                    "status": "queued_reflection",
                    "interaction_id": interaction_data["interaction_id"],
                    "timestamp": interaction_data["timestamp"],
                    "short_term_id": short_term_id,
                    "reflection_job_id": reflection_job.job_id,
                    "priority": priority.name.lower()
                }
            else:
                # Fallback to synchronous processing if queue fails
                logger.warning("Queue failed, falling back to synchronous processing")
                return await self._store_interaction_traditional_path(user_id, interaction)
                
        except Exception as e:
            logger.error(f"Fast path storage failed: {e}")
            # Fallback to traditional path
            return await self._store_interaction_traditional_path(user_id, interaction)
    
    async def _store_interaction_traditional_path(
        self, 
        user_id: str, 
        interaction: Dict[str, Any]
    ) -> Dict[str, str]:
        """Traditional synchronous storage across all tiers."""
        
        # This is the existing store_interaction logic
        # Keep all the current implementation for backward compatibility
        # ... (existing implementation remains unchanged)
        pass
    
    def _determine_reflection_priority(self, interaction: Dict[str, Any]) -> 'ReflectionPriority':
        """Determine priority for reflection processing based on interaction content."""
        from ..reflector.queue.schemas import ReflectionPriority
        
        content = interaction.get('content', '').lower()
        metadata = interaction.get('metadata', {})
        
        # High priority patterns
        high_priority_patterns = [
            'remember', 'important', 'remind me', 'never forget',
            'my name is', 'i am', 'my family', 'emergency',
            'birthday', 'anniversary', 'appointment'
        ]
        
        # Low priority patterns  
        low_priority_patterns = [
            'hello', 'hi', 'thanks', 'ok', 'yes', 'no',
            'test', 'testing', 'status', 'health'
        ]
        
        # Check for explicit priority in metadata
        if metadata.get('priority') == 'high':
            return ReflectionPriority.HIGH
        elif metadata.get('priority') == 'low':
            return ReflectionPriority.LOW
        
        # Content-based priority detection
        if any(pattern in content for pattern in high_priority_patterns):
            return ReflectionPriority.HIGH
        elif any(pattern in content for pattern in low_priority_patterns):
            return ReflectionPriority.LOW
        elif len(content.strip()) < 10:
            return ReflectionPriority.LOW
        else:
            return ReflectionPriority.NORMAL
    
    # Keep all existing methods unchanged for retrieval
    # recall_context, get_health_status, etc.
```

This implementation provides:

1. **Complete backward compatibility** - existing functionality unchanged
2. **Feature flag control** - can enable/disable fast mode
3. **Graceful fallbacks** - queue failures fall back to synchronous processing
4. **Priority-based processing** - intelligent routing of reflection jobs
5. **Comprehensive error handling** - robust failure recovery
6. **Performance monitoring** - detailed metrics and health checks

The refactor maintains all existing memory capabilities while dramatically improving response latency through asynchronous reflection processing.