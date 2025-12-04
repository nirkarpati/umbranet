# Implementation Roadmap: Memory Reflector Service

## 🚀 Step-by-Step Implementation Guide

This roadmap provides the exact order to implement the Memory Reflector Service refactor safely and incrementally. Follow these steps in order to avoid breaking existing functionality.

## Phase 1: Infrastructure Setup (Day 1-2)

### Step 1.1: Add RabbitMQ to Environment

**Priority: CRITICAL - Do this first**

```bash
# 1. Update docker-compose.yml
# Add RabbitMQ service (copy from Implementation Guide)

# 2. Add environment variables
echo "RABBITMQ_PASSWORD=reflection123" >> .env

# 3. Test RabbitMQ setup
docker-compose up rabbitmq -d
curl http://localhost:15672  # Should show RabbitMQ management UI
# Login: umbranet / reflection123
```

**Validation**: ✅ RabbitMQ management UI accessible at http://localhost:15672

### Step 1.2: Update Configuration

```python
# Update src/core/config.py
# Add ALL RabbitMQ settings from Implementation Guide
```

**Files to modify:**
- `src/core/config.py` - Add RabbitMQ and reflection settings
- `docker-compose.yml` - Add RabbitMQ service
- `.env` - Add RABBITMQ_PASSWORD

**Validation**: ✅ Settings load without errors, RabbitMQ connects

---

## Phase 2: Queue Infrastructure (Day 2-3)

### Step 2.1: Create Queue Schemas

**Start here after infrastructure is working**

```bash
# Create directory structure
mkdir -p src/reflector/queue
mkdir -p src/reflector/processors
mkdir -p src/reflector/health

# Create files in this order:
touch src/reflector/__init__.py
touch src/reflector/queue/__init__.py
touch src/reflector/queue/schemas.py
```

**Implementation order:**
1. `src/reflector/queue/schemas.py` - Copy from Code Migration Guide
2. Test schema serialization/deserialization

```python
# Quick test script
from src.reflector.queue.schemas import MemoryReflectionJob
job = MemoryReflectionJob(...)
json_str = job.to_json()
restored = MemoryReflectionJob.from_json(json_str)
assert job.job_id == restored.job_id  # Should pass
```

**Validation**: ✅ Schemas serialize/deserialize correctly

### Step 2.2: Build Queue Producer

```bash
# Install required dependencies first
pip install aio-pika  # Add to requirements.txt
```

**Implementation:**
1. `src/reflector/queue/producer.py` - Copy from Implementation Guide  
2. Test connection to RabbitMQ
3. Test message sending

**Test script:**
```python
# test_producer.py
import asyncio
from src.reflector.queue.producer import QueueProducer
from src.reflector.queue.schemas import MemoryReflectionJob

async def test_producer():
    producer = QueueProducer()
    await producer.connect()
    
    job = MemoryReflectionJob(
        job_id="test-001",
        user_id="test-user", 
        session_id="test-session",
        user_message="Hello test",
        assistant_response="Hi there"
    )
    
    success = await producer.send_reflection_job(job)
    print(f"Message sent: {success}")
    
    await producer.disconnect()

asyncio.run(test_producer())
```

**Validation**: ✅ Messages appear in RabbitMQ management UI queue

### Step 2.3: Build Queue Consumer  

**Implementation:**
1. `src/reflector/queue/consumer.py` - Copy from Implementation Guide
2. Test message consumption with dummy processor

**Test script:**
```python
# test_consumer.py
import asyncio
from src.reflector.queue.consumer import QueueConsumer

async def dummy_processor(job):
    print(f"Processing job: {job.job_id}")
    await asyncio.sleep(0.1)  # Simulate work
    return True

async def test_consumer():
    consumer = QueueConsumer(dummy_processor)
    await consumer.connect()
    await consumer.start_consuming()
    
    # Let it run for 30 seconds to process test messages
    await asyncio.sleep(30)
    await consumer.stop_consuming()

asyncio.run(test_consumer())
```

**Validation**: ✅ Consumer processes messages from queue

---

## Phase 3: Memory Processors (Day 3-4)

### Step 3.1: Create Processor Base Structure

**Create files in this order:**

```bash
touch src/reflector/processors/__init__.py
touch src/reflector/processors/episodic_processor.py
touch src/reflector/processors/semantic_processor.py  
touch src/reflector/processors/procedural_processor.py
```

### Step 3.2: Implement Episodic Processor

**CRITICAL: Follow Code Migration Guide exactly**

1. Copy `EpisodicProcessor` class from Code Migration Guide
2. Test against existing episodic memory logic

**Test script:**
```python
# test_episodic.py
import asyncio
from src.reflector.processors.episodic_processor import EpisodicProcessor
from src.reflector.queue.schemas import MemoryReflectionJob

async def test_episodic():
    processor = EpisodicProcessor()
    await processor.initialize()
    
    job = MemoryReflectionJob(
        job_id="episodic-test",
        user_id="nir",
        session_id="test",
        user_message="I had lunch with my mom today",
        assistant_response="That sounds nice! How was it?"
    )
    
    result = await processor.process_job(job)
    print(f"Episodic result: {result}")
    
    await processor.cleanup()

asyncio.run(test_episodic())
```

**Validation**: ✅ Processor creates same episodic memories as current system

### Step 3.3: Implement Semantic Processor

1. Copy `SemanticProcessor` class from Code Migration Guide
2. Test entity extraction works identically

**Test script:**
```python
# test_semantic.py - Similar pattern as episodic test
```

**Validation**: ✅ Same entities/relationships extracted as current system

### Step 3.4: Implement Procedural Processor

1. Copy `ProceduralProcessor` class 
2. Test (currently placeholder, should match current behavior)

**Validation**: ✅ Consistent with current procedural logic

---

## Phase 4: Memory Reflector Service (Day 4-5)

### Step 4.1: Build Main Service

```bash
touch src/reflector/main.py
touch src/reflector/health/__init__.py
touch src/reflector/health/monitor.py
```

**Implementation order:**
1. `src/reflector/health/monitor.py` - Basic health monitoring
2. `src/reflector/main.py` - Copy from Implementation Guide
3. Test service startup

**Test script:**
```bash
# Test reflector service
cd src/reflector
python -m main  # Should start without errors
```

**Validation**: ✅ Service starts, connects to RabbitMQ, initializes all processors

### Step 4.2: Create Docker Configuration

```dockerfile
# docker/Reflector.Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
CMD ["python", "-m", "src.reflector.main"]
```

**Update docker-compose.yml:**
```yaml
# Add memory-reflector service (copy from Implementation Guide)
```

**Test:**
```bash
docker-compose build memory-reflector
docker-compose up memory-reflector  # Should start successfully
```

**Validation**: ✅ Reflector service runs in Docker, processes queue messages

---

## Phase 5: Integration with Memory Manager (Day 5-6)

### Step 5.1: Update Memory Manager (Non-Breaking)

**CRITICAL: Add feature flags first**

```python
# In src/memory/manager.py __init__
self._enable_fast_mode = settings.memory_fast_mode
self.queue_producer = None  # Will be initialized if needed
```

**Add methods from Code Migration Guide:**
1. `_store_interaction_fast_path()`
2. `_store_interaction_traditional_path()` 
3. `_determine_reflection_priority()`
4. Update `store_interaction()` with feature flag logic

**Validation**: ✅ Existing functionality unchanged when fast_mode=False

### Step 5.2: Add Queue Producer Integration

```python
# In initialize() method
if settings.reflection_enabled:
    from ..reflector.queue.producer import get_queue_producer
    self.queue_producer = await get_queue_producer()
```

**Test:**
```python
# Test both paths work
memory_manager = await get_memory_manager()

# Traditional path
result1 = await memory_manager.store_interaction(user_id, interaction)

# Fast path (if enabled)  
result2 = await memory_manager.store_interaction(user_id, interaction)
```

**Validation**: ✅ Both fast and traditional paths work correctly

---

## Phase 6: End-to-End Testing (Day 6-7)

### Step 6.1: Integration Testing

**Test complete pipeline:**

```python
# test_full_pipeline.py
async def test_complete_reflection_pipeline():
    # 1. Start all services
    # 2. Send message via chat API 
    # 3. Verify Tier 1 stored immediately
    # 4. Wait for reflection processing
    # 5. Verify Tiers 2,3,4 processed correctly
    # 6. Verify same memory state as traditional path
```

### Step 6.2: Performance Testing

```python
# test_performance.py
async def test_latency_improvement():
    # Measure response time with/without fast mode
    # Should see ~95% reduction in chat response latency
```

### Step 6.3: Stress Testing

```bash
# Send 100 concurrent messages
# Verify no message loss
# Verify queue handles backlog correctly
```

**Validation**: ✅ System handles load, no data loss, correct processing

---

## Phase 7: Production Rollout (Day 7+)

### Step 7.1: Feature Flag Rollout

```python
# Start with fast_mode=False (traditional)
MEMORY_FAST_MODE=false
REFLECTION_ENABLED=false

# Enable queue with traditional fallback  
REFLECTION_ENABLED=true
MEMORY_FAST_MODE=false

# Enable fast mode gradually
MEMORY_FAST_MODE=true  # Big latency improvement here!
```

### Step 7.2: Monitoring Setup

```python
# Add metrics endpoints
/api/reflection/health
/api/reflection/stats  
/api/reflection/queue-status
```

### Step 7.3: Full Migration

1. Monitor for 24 hours with fast_mode=true
2. If stable, remove traditional path code
3. Clean up old synchronous processing logic

---

## 🚨 Critical Success Factors

### DO's ✅
1. **Follow exact order** - Each step depends on previous
2. **Test each step** - Don't skip validation steps
3. **Keep feature flags** - Enable safe rollback
4. **Monitor closely** - Watch for errors at each step
5. **Validate data** - Ensure same memory states produced

### DON'Ts ❌
1. **Don't skip infrastructure setup** - RabbitMQ must work first
2. **Don't modify memory logic** - Copy exactly from current code
3. **Don't go straight to production** - Test each component
4. **Don't remove feature flags** - Keep fallback options
5. **Don't rush** - Better to be thorough than fast

---

## 📋 Daily Checklist

### Day 1-2: Infrastructure
- [ ] RabbitMQ running and accessible
- [ ] Configuration updated
- [ ] Environment variables set
- [ ] Docker services connecting

### Day 3-4: Queue + Processors  
- [ ] Messages sent/received successfully
- [ ] All processors initialize without errors
- [ ] Memory operations produce same results
- [ ] Health checks working

### Day 5-6: Integration
- [ ] Memory manager feature flags working
- [ ] Both fast/traditional paths functional
- [ ] No breaking changes to existing API
- [ ] End-to-end pipeline tested

### Day 7+: Production
- [ ] Performance improvements measured
- [ ] No data loss or corruption  
- [ ] Monitoring and alerting active
- [ ] Rollback plan ready

---

## 🎯 Expected Results

**After full implementation:**
- ⚡ **95% faster responses**: 4s → 200ms chat latency
- 📈 **5x throughput**: More concurrent users supported  
- 🔄 **Scalable architecture**: Independent service scaling
- 🛡️ **Fault tolerance**: Memory issues don't break chat
- 📊 **Rich monitoring**: Detailed metrics and health checks

**Success metrics to track:**
- Chat API response time < 500ms
- Memory reflection processing < 30s  
- Zero message loss during queue processing
- Same memory accuracy as before refactor

This roadmap ensures a safe, incremental rollout with validation at every step!