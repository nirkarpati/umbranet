"""Processor for Tier 2: Episodic Memory operations."""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from ...core.config import settings
from ...memory.tiers.episodic import EpisodicMemoryStore
from ...memory.services.episodic_curator import create_episodic_curator
from ..queue.schemas import MemoryReflectionJob

logger = logging.getLogger(__name__)

class EpisodicProcessor:
    """Processor for Tier 2: Episodic Memory operations."""
    
    def __init__(self):
        self.episodic_store: Optional[EpisodicMemoryStore] = None
        self.processing_count = 0
        self.success_count = 0
        self.error_count = 0
    
    async def initialize(self) -> None:
        """Initialize episodic memory store."""
        try:
            self.episodic_store = EpisodicMemoryStore()
            # Test connection
            async with self.episodic_store:
                pass  # Connection test
            logger.info("✅ Episodic processor initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize episodic processor: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Cleanup episodic processor resources."""
        if self.episodic_store:
            # EpisodicMemoryStore cleanup handled by context manager
            pass
        logger.info("🧹 Episodic processor cleaned up")
    
    async def process_job(self, job: MemoryReflectionJob) -> Dict[str, Any]:
        """Process episodic memory for reflection job.
        
        This method contains the EXACT logic from the current MemoryManager
        but adapted for the processor architecture.
        """
        start_time = datetime.utcnow()
        self.processing_count += 1
        
        try:
            logger.debug(f"📖 Processing episodic memory for job {job.job_id}")
            
            # MIGRATED CODE: Use curator to decide if/how to store in episodic memory
            async with await create_episodic_curator() as curator:
                curation_result = await curator.curate_interaction(
                    user_message=job.user_message,
                    assistant_response=job.assistant_response,
                    user_id=job.user_id
                )
            
            if not curation_result.should_store:
                logger.info(f"📝 Episodic curator decided not to store job {job.job_id}: {curation_result.reasoning}")
                self.success_count += 1
                return {
                    "status": "skipped",
                    "reason": curation_result.reasoning,
                    "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
                }
            
            logger.info(f"📝 Episodic curator storing job {job.job_id} with importance {curation_result.importance_score:.1f}")
            
            # MIGRATED CODE: Store the curated content  
            async with self.episodic_store:
                # Include session_id and curation metadata
                enhanced_metadata = job.metadata.copy()
                if job.session_id:
                    enhanced_metadata["session_id"] = job.session_id
                
                # Add curation metadata
                enhanced_metadata.update({
                    "importance_score": curation_result.importance_score,
                    "tags": curation_result.tags,
                    "summary": curation_result.summary,
                    "curation_method": "llm" if settings.openai_api_key else "rule_based",
                    "reflection_job_id": job.job_id,
                    "processed_at": datetime.utcnow().isoformat()
                })
                
                # Parse curated content - EXACT LOGIC from MemoryManager
                user_message = curation_result.content_to_store.split("\\nAssistant:")[0].replace("User: ", "")
                assistant_response = (
                    curation_result.content_to_store.split("\\nAssistant: ")[-1] 
                    if "\\nAssistant: " in curation_result.content_to_store 
                    else job.assistant_response
                )
                
                episode_id = await self.episodic_store.log_interaction(
                    user_id=job.user_id,
                    user_message=user_message,
                    assistant_response=assistant_response,
                    metadata=enhanced_metadata
                )
                
                self.success_count += 1
                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                logger.info(f"✅ Episodic processing completed for job {job.job_id} in {processing_time:.1f}ms")
                
                return {
                    "status": "stored",
                    "episode_id": episode_id,
                    "importance_score": curation_result.importance_score,
                    "summary": curation_result.summary,
                    "tags": curation_result.tags,
                    "processing_time_ms": processing_time
                }
                
        except Exception as e:
            self.error_count += 1
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.error(f"❌ Episodic processing failed for job {job.job_id} in {processing_time:.1f}ms: {e}")
            raise
    
    def get_health(self) -> Dict[str, Any]:
        """Get processor health metrics."""
        success_rate = self.success_count / max(self.processing_count, 1)
        return {
            "processor": "episodic",
            "healthy": True,
            "total_processed": self.processing_count,
            "successful": self.success_count,
            "errors": self.error_count,
            "success_rate": success_rate,
            "store_initialized": self.episodic_store is not None
        }