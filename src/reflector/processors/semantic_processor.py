"""Processor for Tier 3: Semantic Memory operations."""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from ...memory.tiers.semantic import SemanticMemoryStore
from ..queue.schemas import MemoryReflectionJob

logger = logging.getLogger(__name__)

class SemanticProcessor:
    """Processor for Tier 3: Semantic Memory operations."""
    
    def __init__(self):
        self.semantic_store: Optional[SemanticMemoryStore] = None
        self.processing_count = 0
        self.success_count = 0
        self.error_count = 0
    
    async def initialize(self) -> None:
        """Initialize semantic memory store."""
        try:
            self.semantic_store = SemanticMemoryStore()
            # Test connection
            async with self.semantic_store:
                pass  # Connection test
            logger.info("✅ Semantic processor initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize semantic processor: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Cleanup semantic processor resources."""
        if self.semantic_store:
            # SemanticMemoryStore cleanup handled by context manager
            pass
        logger.info("🧹 Semantic processor cleaned up")
    
    async def process_job(self, job: MemoryReflectionJob) -> Dict[str, Any]:
        """Process semantic memory for reflection job.
        
        This method contains the EXACT logic from the current MemoryManager
        but adapted for the processor architecture.
        """
        start_time = datetime.utcnow()
        self.processing_count += 1
        
        try:
            logger.debug(f"🕸️ Processing semantic memory for job {job.job_id}")
            
            # MIGRATED CODE: Semantic memory extracts entities and relationships
            async with self.semantic_store:
                extraction_result = await self.semantic_store.extract_and_store_entities(
                    user_id=job.user_id,
                    user_message=job.user_message,
                    assistant_response=job.assistant_response
                )
                
                entity_count = len(extraction_result.entities)
                relationship_count = len(extraction_result.relationships)
                
                self.success_count += 1
                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                logger.info(f"✅ Semantic processing completed for job {job.job_id}: {entity_count} entities, {relationship_count} relationships in {processing_time:.1f}ms")
                
                return {
                    "status": "stored",
                    "result_id": f"semantic_{entity_count}_{relationship_count}",
                    "entities_extracted": entity_count,
                    "relationships_extracted": relationship_count,
                    "entities": [{"name": e.name, "type": e.entity_type} for e in extraction_result.entities],
                    "relationships": [{"from": r.from_entity_id, "to": r.to_entity_id, "type": r.relationship_type} for r in extraction_result.relationships],
                    "processing_time_ms": processing_time
                }
                
        except Exception as e:
            self.error_count += 1
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.error(f"❌ Semantic processing failed for job {job.job_id} in {processing_time:.1f}ms: {e}")
            raise
    
    def get_health(self) -> Dict[str, Any]:
        """Get processor health metrics."""
        success_rate = self.success_count / max(self.processing_count, 1)
        return {
            "processor": "semantic",
            "healthy": True,
            "total_processed": self.processing_count,
            "successful": self.success_count,
            "errors": self.error_count,
            "success_rate": success_rate,
            "store_initialized": self.semantic_store is not None
        }