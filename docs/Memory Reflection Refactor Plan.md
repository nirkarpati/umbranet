# Memory Reflection Refactor Plan

## Overview

This document outlines a major architectural refactor to separate memory "reflection" (processing and storing to tiers 2, 3, 4) from the main chat API flow using an asynchronous queue-based system. This will significantly reduce response latency while maintaining the rich memory capabilities.

## Current Architecture Problems

### Synchronous Memory Processing
- **High Latency**: Every chat message triggers synchronous processing of all memory tiers
- **Blocking Operations**: LLM calls for entity extraction, curation, and routing block response
- **User Experience**: Users wait 2-4 seconds for responses due to memory processing
- **Resource Contention**: Memory operations compete with chat response generation

### Current Flow Analysis
```
User Message → FastAPI → Governor Workflow → Memory Manager
                ↓
            Store in All Tiers (synchronous):
            - Tier 1: Redis (fast)
            - Tier 2: Episodic + LLM Curation (slow)  
            - Tier 3: Semantic + LLM Extraction (slow)
            - Tier 4: Procedural + LLM Analysis (slow)
                ↓
            Generate Response → Return to User
```

**Total Latency**: ~2-4 seconds per message

## Proposed Architecture: Asynchronous Memory Reflection

### New Flow Design
```
User Message → FastAPI → Governor Workflow
                ↓
            Store Tier 1 Only (Redis - fast)
                ↓
            Send to Reflection Queue → Return Response to User
                ↓
            Memory Reflector Service (background):
            - Process Queue Messages
            - Handle Tiers 2, 3, 4 asynchronously
            - Rich LLM-powered analysis
```

**New Chat Latency**: ~200-500ms (95% reduction)

### Core Components

#### 1. Message Queue (RabbitMQ)
- **Queue**: `memory_reflection_queue`
- **Exchange**: `memory_reflection_exchange`
- **Routing Key**: `reflection.{user_id}`
- **Message Format**: JSON with conversation data + metadata

#### 2. Memory Reflector Service (New Container)
- **Purpose**: Background processing of memory tiers 2, 3, 4
- **Technology**: Python asyncio service
- **Scaling**: Horizontal scaling with multiple workers
- **Resilience**: Dead letter queue for failed processing

#### 3. Modified Memory Manager
- **Fast Path**: Only Tier 1 (Redis) operations during chat
- **Queue Path**: Send reflection job to RabbitMQ
- **Retrieval**: Unchanged (still queries all tiers)

## Technical Implementation Plan

### Phase 1: Queue Infrastructure Setup

#### 1.1 RabbitMQ Integration
```yaml
# docker-compose.yml addition
rabbitmq:
  image: rabbitmq:3.12-management-alpine
  container_name: umbranet_rabbitmq
  ports:
    - "5672:5672"
    - "15672:15672"
  environment:
    RABBITMQ_DEFAULT_USER: umbranet
    RABBITMQ_DEFAULT_PASS: ${RABBITMQ_PASSWORD}
  volumes:
    - rabbitmq_data:/var/lib/rabbitmq
  networks:
    - umbranet_network
```

#### 1.2 Queue Message Schema
```python
@dataclass
class MemoryReflectionJob:
    job_id: str
    user_id: str
    session_id: str
    user_message: str
    assistant_response: str
    timestamp: datetime
    metadata: Dict[str, Any]
    priority: int = 0  # 0=normal, 1=high
    retry_count: int = 0
    max_retries: int = 3
```

### Phase 2: Memory Reflector Service

#### 2.1 Service Architecture
```
src/reflector/
├── __init__.py
├── main.py                 # Service entry point
├── queue/
│   ├── __init__.py
│   ├── consumer.py         # RabbitMQ consumer
│   ├── producer.py         # Queue message producer
│   └── schemas.py          # Message schemas
├── processors/
│   ├── __init__.py
│   ├── episodic_processor.py    # Tier 2 processing
│   ├── semantic_processor.py    # Tier 3 processing
│   ├── procedural_processor.py  # Tier 4 processing
│   └── batch_processor.py       # Batch processing optimization
├── health/
│   ├── __init__.py
│   └── monitor.py          # Health checking and metrics
└── config/
    ├── __init__.py
    └── settings.py         # Reflector-specific config
```

#### 2.2 Core Processor Logic
```python
class MemoryReflector:
    async def process_reflection_job(self, job: MemoryReflectionJob):
        """Process memory reflection job asynchronously."""
        
        # Parallel processing of tiers 2, 3, 4
        processors = [
            self.episodic_processor.process(job),
            self.semantic_processor.process(job),
            self.procedural_processor.process(job)
        ]
        
        # Execute with timeout and error handling
        results = await asyncio.gather(*processors, return_exceptions=True)
        
        # Handle partial failures gracefully
        return self._consolidate_results(results, job)
```

### Phase 3: Modified Memory Manager

#### 3.1 Split Memory Operations
```python
class MemoryManager:
    async def store_interaction_fast(self, user_id: str, interaction: Dict[str, Any]) -> Dict[str, str]:
        """Fast path: Only Tier 1 + Queue job."""
        
        # 1. Store in Tier 1 (Redis) immediately
        short_term_id = await self._store_short_term_only(user_id, interaction)
        
        # 2. Queue reflection job (non-blocking)
        await self.queue_producer.send_reflection_job(
            MemoryReflectionJob.from_interaction(user_id, interaction)
        )
        
        return {"status": "queued", "short_term_id": short_term_id}
    
    # Keep existing recall_context unchanged
    async def recall_context(self, user_id: str, current_input: str) -> Dict[str, Any]:
        """Unchanged - still queries all tiers for retrieval."""
        # Same logic as before
```

#### 3.2 Queue Producer Integration
```python
class QueueProducer:
    async def send_reflection_job(self, job: MemoryReflectionJob, priority: int = 0):
        """Send reflection job to RabbitMQ."""
        
        # Serialize job
        message = job.to_json()
        
        # Send to queue with appropriate priority
        await self.channel.basic_publish(
            exchange='memory_reflection_exchange',
            routing_key=f'reflection.{job.user_id}',
            body=message,
            properties=aio_pika.MessageProperties(
                priority=priority,
                delivery_mode=2,  # Persistent
                timestamp=datetime.utcnow()
            )
        )
```

### Phase 4: Enhanced Features

#### 4.1 Batch Processing Optimization
```python
class BatchProcessor:
    async def process_user_batch(self, user_id: str, jobs: List[MemoryReflectionJob]):
        """Process multiple jobs for same user in batch for efficiency."""
        
        # Combine related conversations for better context
        # More efficient LLM calls
        # Better relationship extraction across conversations
```

#### 4.2 Priority Queue System
- **High Priority**: Important conversations, explicit user requests
- **Normal Priority**: Regular chat messages
- **Low Priority**: System messages, status updates

#### 4.3 Dead Letter Queue & Retry Logic
- Failed jobs moved to dead letter queue
- Exponential backoff for retries
- Admin interface for failed job analysis

## Configuration Changes

### Environment Variables
```bash
# RabbitMQ Configuration
RABBITMQ_URL=amqp://umbranet:password@localhost:5672/
RABBITMQ_EXCHANGE=memory_reflection_exchange
RABBITMQ_QUEUE=memory_reflection_queue

# Memory Reflector Service
REFLECTOR_WORKERS=3
REFLECTOR_BATCH_SIZE=10
REFLECTOR_MAX_RETRIES=3
REFLECTOR_RETRY_DELAY=30

# Performance Tuning
MEMORY_FAST_MODE=true
REFLECTION_QUEUE_ENABLED=true
```

### Docker Composition
```yaml
# New service addition
memory-reflector:
  build:
    context: .
    dockerfile: docker/Reflector.Dockerfile
  container_name: umbranet_reflector
  environment:
    - RABBITMQ_URL=${RABBITMQ_URL}
    - POSTGRES_URL=${POSTGRES_URL}
    - NEO4J_URL=${NEO4J_URL}
    - REDIS_URL=${REDIS_URL}
    - OPENAI_API_KEY=${OPENAI_API_KEY}
  depends_on:
    - rabbitmq
    - postgres
    - neo4j
    - redis
  volumes:
    - ./logs:/app/logs
  networks:
    - umbranet_network
  deploy:
    replicas: 2  # Multiple workers
```

## Migration Strategy

### Phase 1: Setup Infrastructure
1. Add RabbitMQ to docker-compose
2. Create queue producer in memory manager
3. Test queue connectivity

### Phase 2: Build Reflector Service  
1. Implement basic queue consumer
2. Port existing memory processing logic
3. Add error handling and monitoring

### Phase 3: Enable Fast Mode
1. Add feature flag for queue-based reflection
2. Modify chat API to use fast path
3. Monitor both old and new systems in parallel

### Phase 4: Full Migration
1. Switch all users to queue-based system
2. Remove synchronous memory processing
3. Optimize and tune performance

## Expected Benefits

### Performance Improvements
- **Response Latency**: 95% reduction (4s → 200ms)
- **Throughput**: 5x increase in concurrent users
- **Resource Utilization**: Better separation of concerns

### Scalability Benefits
- **Horizontal Scaling**: Independent scaling of chat API vs. memory processing
- **Resource Optimization**: Dedicated resources for different workloads
- **Queue Management**: Built-in backpressure and load balancing

### Reliability Improvements
- **Fault Tolerance**: Memory failures don't affect chat responses
- **Retry Logic**: Failed reflections are retried automatically
- **Monitoring**: Separate health metrics for each component

## Monitoring & Observability

### Metrics to Track
- Queue depth and processing rate
- Reflection job success/failure rates
- Memory tier processing latencies
- Chat API response times

### Alerting
- Queue backlog alerts
- Failed job rate thresholds
- Memory reflector service health
- Cross-service dependency monitoring

## Risks & Mitigations

### Potential Issues
1. **Eventual Consistency**: Memory might lag behind chat
2. **Queue Failures**: RabbitMQ downtime affects reflection
3. **Complex Debugging**: Distributed system complexity

### Mitigations
1. **Status API**: Endpoint to check reflection status
2. **Queue Persistence**: Durable queues with disk persistence  
3. **Comprehensive Logging**: Distributed tracing across services
4. **Graceful Degradation**: Fallback to synchronous mode if needed

## Success Metrics

### Primary Objectives
- [ ] Chat response latency < 500ms (95th percentile)
- [ ] Memory reflection processing within 30 seconds
- [ ] Zero message loss during queue transitions
- [ ] 99.9% uptime for both chat API and reflector service

### Secondary Objectives  
- [ ] 50% reduction in memory processing resource usage
- [ ] Support for 10x more concurrent users
- [ ] Implementation completed within 2 weeks
- [ ] Comprehensive monitoring dashboard deployed

---

*This refactor represents a significant architectural improvement that will transform Umbranet from a synchronous system into a modern, scalable, event-driven architecture while maintaining all existing memory capabilities.*