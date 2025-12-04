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
            
            # Extract procedural rules and preferences from the interaction
            async with self.procedural_store:
                # Analyze conversation for procedural rules using LLM
                rules = await self._extract_procedural_rules(job)
                
                stored_rules = []
                if rules:
                    for rule in rules:
                        try:
                            # Import here to avoid circular imports
                            from ...core.domain.procedural import InstructionCategory
                            
                            # Use LLM-generated category directly - no mapping, just use GENERAL for storage
                            # The actual category is stored in metadata as the LLM generated it
                            
                            rule_id = await self.procedural_store.add_instruction(
                                user_id=job.user_id,
                                category=InstructionCategory.GENERAL,  # Use generic category for storage
                                title=rule["title"],
                                instruction=rule["description"],  # Use correct parameter name
                                confidence=rule.get("confidence", 0.8),
                                priority=int(rule.get("confidence", 0.8) * 10),  # Convert confidence to priority 1-10
                                metadata={
                                    "source": "conversation",
                                    "job_id": job.job_id,
                                    "timestamp": job.timestamp.isoformat(),
                                    "context": rule.get("context", ""),
                                    "category": rule["type"],  # Store LLM-generated category as-is
                                    "confidence": rule.get("confidence", 0.8)
                                }
                            )
                            stored_rules.append({
                                "rule_id": rule_id,
                                "type": rule["type"],
                                "title": rule["title"]
                            })
                            logger.info(f"📋 Stored procedural rule: {rule['title']} ({rule['type']})")
                        except Exception as e:
                            logger.error(f"Failed to store procedural rule '{rule['title']}': {e}")
                
                self.success_count += 1
                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                logger.info(f"✅ Procedural processing completed for job {job.job_id} in {processing_time:.1f}ms: {len(stored_rules)} rules stored")
                
                return {
                    "status": "processed",
                    "rules_stored": len(stored_rules),
                    "rules": stored_rules,
                    "processing_time_ms": processing_time
                }
                
        except Exception as e:
            self.error_count += 1
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.error(f"❌ Procedural processing failed for job {job.job_id} in {processing_time:.1f}ms: {e}")
            raise
    
    async def _extract_procedural_rules(self, job: MemoryReflectionJob) -> list:
        """Extract procedural rules from conversation using LLM analysis."""
        try:
            # Import here to avoid circular imports
            from ...memory.services.summarizer import ConversationSummarizer
            
            # Build prompt for procedural rule extraction
            extraction_prompt = f"""
Analyze this conversation to extract procedural rules, preferences, and behavioral patterns that should be stored as actionable guidelines for future interactions.

USER MESSAGE: {job.user_message}
ASSISTANT RESPONSE: {job.assistant_response}

PROCEDURAL RULE TYPES:
Generate appropriate category names based on the type of rule/preference being established.

EXAMPLES OF RULE TYPES:
- "pronoun_preference" - for pronoun usage
- "response_style" - for communication preferences  
- "scheduling_rules" - for time/meeting preferences
- "formality_level" - for tone and formality
- "notification_settings" - for alert preferences
- "permission_rules" - for access control
- "workflow_preferences" - for task management
- "content_filters" - for topic preferences

Create specific, descriptive category names that accurately capture the nature of each rule.

EXTRACTION RULES:
✅ EXTRACT if user explicitly states preferences or rules
✅ EXTRACT if user corrects assistant behavior 
✅ EXTRACT if pattern is clearly established and actionable
❌ SKIP one-time events or temporary states
❌ SKIP vague or unclear preferences
❌ SKIP factual information (that belongs in semantic memory)

Return JSON array of rules:
[
  {{
    "type": "specific_category_name",
    "title": "Brief descriptive title (max 50 chars)",
    "description": "Clear actionable rule description",
    "confidence": 0.1-1.0,
    "context": "Why this rule was extracted"
  }}
]

If no procedural rules found, return empty array [].

EXAMPLES:
User: "Please use she/her pronouns for me" → type: "pronoun_preference"
User: "I prefer brief responses" → type: "response_style"  
User: "Don't schedule anything before 9 AM" → type: "scheduling_rules"
User: "Always confirm before deleting files" → type: "permission_rules"

Extract procedural rules:"""

            # Use existing summarizer for LLM calls
            summarizer = ConversationSummarizer()
            response = await summarizer._call_openai(extraction_prompt, max_tokens=800)
            
            # Parse JSON response
            import json
            import re
            try:
                # Try direct JSON parsing first
                rules = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON array from response
                json_match = re.search(r'\[.*?\]', response, re.DOTALL)
                if json_match:
                    rules = json.loads(json_match.group())
                else:
                    logger.warning(f"No valid JSON found in procedural extraction response: {response}")
                    return []
            
            # Validate and filter rules (no category restrictions - LLM generates them)
            valid_rules = []
            
            for rule in rules:
                if (isinstance(rule, dict) and 
                    rule.get("type") and  # Any type the LLM generates is valid
                    rule.get("title") and 
                    rule.get("description") and
                    0.1 <= rule.get("confidence", 0) <= 1.0):
                    valid_rules.append(rule)
                else:
                    logger.warning(f"Invalid procedural rule format: {rule}")
            
            if valid_rules:
                logger.info(f"🔍 Extracted {len(valid_rules)} procedural rules from conversation")
                for rule in valid_rules:
                    logger.debug(f"  - {rule['title']} ({rule['type']})")
            
            return valid_rules
            
        except Exception as e:
            logger.error(f"Failed to extract procedural rules: {e}")
            return []
    
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