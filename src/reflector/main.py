"""Memory Reflector Service - Main entry point for asynchronous memory processing."""

import asyncio
import logging
import signal
import sys
from typing import Dict, Any, Optional

from .queue.consumer import QueueConsumer
from .processors.episodic_processor import EpisodicProcessor
from .processors.semantic_processor import SemanticProcessor
from .processors.procedural_processor import ProceduralProcessor
from .health.monitor import get_health_monitor, HealthMonitor
from .queue.schemas import MemoryReflectionJob

logger = logging.getLogger(__name__)

class MemoryReflectorService:
    """Main service orchestrating asynchronous memory reflection processing."""
    
    def __init__(self):
        self.consumer: Optional[QueueConsumer] = None
        self.processors: Dict[str, Any] = {}
        self.health_monitor: HealthMonitor = get_health_monitor()
        self._running = False
        self._shutdown_event = asyncio.Event()
        
    async def initialize(self) -> None:
        """Initialize all service components."""
        logger.info("🚀 Starting Memory Reflector Service initialization...")
        
        try:
            # Initialize processors
            logger.info("⚙️ Initializing memory processors...")
            
            # Episodic processor
            self.processors["episodic"] = EpisodicProcessor()
            await self.processors["episodic"].initialize()
            self.health_monitor.register_processor("episodic", self.processors["episodic"])
            
            # Semantic processor
            self.processors["semantic"] = SemanticProcessor()
            await self.processors["semantic"].initialize()
            self.health_monitor.register_processor("semantic", self.processors["semantic"])
            
            # Procedural processor
            self.processors["procedural"] = ProceduralProcessor()
            await self.processors["procedural"].initialize()
            self.health_monitor.register_processor("procedural", self.processors["procedural"])
            
            logger.info("✅ All processors initialized successfully")
            
            # Initialize queue consumer with callback
            logger.info("📥 Initializing queue consumer...")
            self.consumer = QueueConsumer(self._process_reflection_job)
            await self.consumer.connect()
            self.health_monitor.register_queue_consumer(self.consumer)
            
            # Start consuming messages
            await self.consumer.start_consuming()
            
            logger.info("✅ Memory Reflector Service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Memory Reflector Service: {e}")
            raise
    
    async def _process_reflection_job(self, job: MemoryReflectionJob) -> Dict[str, Any]:
        """Process a memory reflection job using appropriate processors."""
        self.health_monitor.record_job_start()
        
        try:
            logger.debug(f"🔄 Processing reflection job {job.job_id} for user {job.user_id}")
            
            results = {}
            
            # Process with episodic processor (Tier 2)
            try:
                episodic_result = await self.processors["episodic"].process_job(job)
                results["episodic"] = episodic_result
                logger.debug(f"✅ Episodic processing completed for job {job.job_id}")
            except Exception as e:
                logger.error(f"❌ Episodic processing failed for job {job.job_id}: {e}")
                results["episodic"] = {"status": "failed", "error": str(e)}
            
            # Process with semantic processor (Tier 3)
            try:
                semantic_result = await self.processors["semantic"].process_job(job)
                results["semantic"] = semantic_result
                logger.debug(f"✅ Semantic processing completed for job {job.job_id}")
            except Exception as e:
                logger.error(f"❌ Semantic processing failed for job {job.job_id}: {e}")
                results["semantic"] = {"status": "failed", "error": str(e)}
            
            # Process with procedural processor (Tier 4)
            try:
                procedural_result = await self.processors["procedural"].process_job(job)
                results["procedural"] = procedural_result
                logger.debug(f"✅ Procedural processing completed for job {job.job_id}")
            except Exception as e:
                logger.error(f"❌ Procedural processing failed for job {job.job_id}: {e}")
                results["procedural"] = {"status": "failed", "error": str(e)}
            
            self.health_monitor.record_job_success()
            
            logger.info(f"✅ Completed reflection job {job.job_id} for user {job.user_id}")
            return {
                "job_id": job.job_id,
                "user_id": job.user_id,
                "status": "completed",
                "results": results
            }
            
        except Exception as e:
            self.health_monitor.record_job_failure()
            logger.error(f"❌ Failed to process reflection job {job.job_id}: {e}")
            return {
                "job_id": job.job_id,
                "user_id": job.user_id,
                "status": "failed",
                "error": str(e)
            }
    
    async def run(self) -> None:
        """Run the reflector service."""
        self._running = True
        logger.info("🔄 Memory Reflector Service is running...")
        
        try:
            # Wait for shutdown signal
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            logger.info("🛑 Service run cancelled")
        finally:
            self._running = False
    
    async def health_check(self) -> Dict[str, Any]:
        """Get service health status."""
        return await self.health_monitor.check_health()
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the reflector service."""
        logger.info("🛑 Shutting down Memory Reflector Service...")
        
        self._running = False
        self._shutdown_event.set()
        
        # Cleanup queue consumer
        if self.consumer:
            await self.consumer.disconnect()
            logger.info("📥 Queue consumer disconnected")
        
        # Cleanup processors
        for name, processor in self.processors.items():
            try:
                await processor.cleanup()
                logger.info(f"⚙️ {name.capitalize()} processor cleaned up")
            except Exception as e:
                logger.error(f"❌ Error cleaning up {name} processor: {e}")
        
        logger.info("✅ Memory Reflector Service shutdown complete")

# Global service instance
_service: Optional[MemoryReflectorService] = None

async def get_reflector_service() -> MemoryReflectorService:
    """Get or create global reflector service."""
    global _service
    if _service is None:
        _service = MemoryReflectorService()
        await _service.initialize()
    return _service

def setup_signal_handlers(service: MemoryReflectorService) -> None:
    """Setup signal handlers for graceful shutdown."""
    def handle_shutdown_signal(signum, frame):
        logger.info(f"📡 Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(service.shutdown())
    
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    signal.signal(signal.SIGINT, handle_shutdown_signal)

async def main():
    """Main entry point for Memory Reflector Service."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger.info("🚀 Starting Memory Reflector Service...")
    
    service = None
    try:
        # Create and initialize service
        service = MemoryReflectorService()
        await service.initialize()
        
        # Setup signal handlers
        setup_signal_handlers(service)
        
        # Run service
        await service.run()
        
    except KeyboardInterrupt:
        logger.info("📡 Received keyboard interrupt")
    except Exception as e:
        logger.error(f"❌ Service failed: {e}")
        return 1
    finally:
        if service:
            await service.shutdown()
    
    logger.info("👋 Memory Reflector Service stopped")
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))