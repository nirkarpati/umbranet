"""Health monitoring for memory reflector service."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class HealthMetrics:
    """Health metrics for reflector service."""
    service_start_time: datetime
    total_jobs_processed: int = 0
    successful_jobs: int = 0
    failed_jobs: int = 0
    queue_connection_healthy: bool = False
    processors_healthy: Dict[str, bool] = None
    last_job_processed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.processors_healthy is None:
            self.processors_healthy = {}

class HealthMonitor:
    """Monitors health of memory reflector service components."""
    
    def __init__(self):
        self.metrics = HealthMetrics(service_start_time=datetime.utcnow())
        self._processors: Dict[str, Any] = {}
        self._queue_consumer: Optional[Any] = None
        
    def register_processor(self, name: str, processor: Any) -> None:
        """Register a processor for health monitoring."""
        self._processors[name] = processor
        self.metrics.processors_healthy[name] = False
        logger.debug(f"📊 Registered processor for health monitoring: {name}")
    
    def register_queue_consumer(self, consumer: Any) -> None:
        """Register queue consumer for health monitoring."""
        self._queue_consumer = consumer
        logger.debug("📊 Registered queue consumer for health monitoring")
    
    async def check_health(self) -> Dict[str, Any]:
        """Check overall health of reflector service."""
        health_status = {
            "service": "memory_reflector",
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": (datetime.utcnow() - self.metrics.service_start_time).total_seconds(),
            "metrics": {
                "total_jobs_processed": self.metrics.total_jobs_processed,
                "successful_jobs": self.metrics.successful_jobs,
                "failed_jobs": self.metrics.failed_jobs,
                "success_rate": self._calculate_success_rate(),
                "last_job_processed_at": self.metrics.last_job_processed_at.isoformat() if self.metrics.last_job_processed_at else None
            },
            "components": {}
        }
        
        # Check queue consumer health
        if self._queue_consumer:
            try:
                queue_health = await self._queue_consumer.health_check()
                self.metrics.queue_connection_healthy = queue_health.get("healthy", False)
                health_status["components"]["queue_consumer"] = queue_health
            except Exception as e:
                self.metrics.queue_connection_healthy = False
                health_status["components"]["queue_consumer"] = {
                    "status": "error",
                    "healthy": False,
                    "error": str(e)
                }
        else:
            health_status["components"]["queue_consumer"] = {
                "status": "not_registered",
                "healthy": False
            }
        
        # Check processor health
        processors_health = {}
        all_processors_healthy = True
        
        for name, processor in self._processors.items():
            try:
                processor_health = processor.get_health()
                processors_health[name] = processor_health
                self.metrics.processors_healthy[name] = processor_health.get("healthy", False)
                if not processor_health.get("healthy", False):
                    all_processors_healthy = False
            except Exception as e:
                processors_health[name] = {
                    "status": "error",
                    "healthy": False,
                    "error": str(e)
                }
                self.metrics.processors_healthy[name] = False
                all_processors_healthy = False
        
        health_status["components"]["processors"] = processors_health
        
        # Overall health determination
        overall_healthy = (
            self.metrics.queue_connection_healthy and
            all_processors_healthy and
            len(self._processors) > 0
        )
        
        health_status["status"] = "healthy" if overall_healthy else "unhealthy"
        health_status["healthy"] = overall_healthy
        
        return health_status
    
    def record_job_start(self) -> None:
        """Record that a job processing has started."""
        self.metrics.total_jobs_processed += 1
        
    def record_job_success(self) -> None:
        """Record successful job completion."""
        self.metrics.successful_jobs += 1
        self.metrics.last_job_processed_at = datetime.utcnow()
        
    def record_job_failure(self) -> None:
        """Record failed job completion."""
        self.metrics.failed_jobs += 1
        self.metrics.last_job_processed_at = datetime.utcnow()
    
    def _calculate_success_rate(self) -> float:
        """Calculate job success rate."""
        if self.metrics.total_jobs_processed == 0:
            return 1.0
        return self.metrics.successful_jobs / self.metrics.total_jobs_processed

# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None

def get_health_monitor() -> HealthMonitor:
    """Get or create global health monitor."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor