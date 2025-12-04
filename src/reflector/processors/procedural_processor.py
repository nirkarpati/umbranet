"""Processor for Tier 4: Procedural Memory operations."""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from ...memory.tiers.procedural import ProceduralMemoryStore
from ..queue.schemas import MemoryReflectionJob

logger = logging.getLogger(__name__)

class ProceduralProcessor:
    """Processor for Tier 4: Procedural Memory operations."""
    
    def __init__(self):
        self.procedural_store: Optional[ProceduralMemoryStore] = None
        self.processing_count = 0
        self.success_count = 0
        self.error_count = 0
    
    async def initialize(self) -> None:
        """Initialize procedural memory store."""
        try:
            self.procedural_store = ProceduralMemoryStore()
            # Test connection
            async with self.procedural_store:
                pass  # Connection test
            logger.info("✅ Procedural processor initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize procedural processor: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Cleanup procedural processor resources."""
        if self.procedural_store:
            # ProceduralMemoryStore cleanup handled by context manager
            pass
        logger.info("🧹 Procedural processor cleaned up")
    
    async def process_job(self, job: MemoryReflectionJob) -> Dict[str, Any]:
        """Process procedural memory for reflection job.
        
        This method contains the EXACT logic from the current MemoryManager
        but adapted for the processor architecture.
        """
        start_time = datetime.utcnow()
        self.processing_count += 1
        
        try:
            logger.debug(f"⚙️ Processing procedural memory for job {job.job_id}")
            
            # MIGRATED CODE: Procedural memory stores preferences and rules
            # Note: Current implementation is a placeholder, but we keep the exact logic
            async with self.procedural_store:
                # TODO: The current MemoryManager implementation is a placeholder
                # This would involve preference extraction from the interaction
                # For now, we return a consistent result like the original
                
                result_id = f"procedural_{job.user_id}_{job.timestamp.isoformat()}"
                
                # Future enhancement: Add actual preference/rule extraction logic here
                # This could include:
                # - Pattern matching for user preferences
                # - Rule extraction from behavioral patterns  
                # - Integration with tier routing logic for procedural content
                
                self.success_count += 1
                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                logger.info(f"✅ Procedural processing completed for job {job.job_id} in {processing_time:.1f}ms (placeholder)")
                
                return {
                    "status": "processed",
                    "result_id": result_id,
                    "note": "Placeholder implementation - matches current MemoryManager logic",
                    "processing_time_ms": processing_time
                }
                
        except Exception as e:
            self.error_count += 1
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.error(f"❌ Procedural processing failed for job {job.job_id} in {processing_time:.1f}ms: {e}")
            raise
    
    def get_health(self) -> Dict[str, Any]:
        """Get processor health metrics."""
        success_rate = self.success_count / max(self.processing_count, 1)
        return {
            "processor": "procedural",
            "healthy": True,
            "total_processed": self.processing_count,
            "successful": self.success_count,
            "errors": self.error_count,
            "success_rate": success_rate,
            "store_initialized": self.procedural_store is not None
        }