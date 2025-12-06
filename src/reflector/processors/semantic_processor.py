"""Processor for Tier 3: Semantic Memory operations."""

import json
import logging
from datetime import datetime
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ...core.config import settings
from ...core.embeddings.provider_factory import get_embedding_provider
from ...memory.database.neo4j_client import get_neo4j_connection
from ...memory.tools.graph_ops import GraphMaintenanceTools
from ..queue.schemas import MemoryReflectionJob

logger = logging.getLogger(__name__)


class SemanticProcessor:
    """Processor for Tier 3: Semantic Memory operations using OpenAI Tool-Use Agent."""

    def __init__(self):
        self.graph_tools: GraphMaintenanceTools | None = None
        self.http_client = httpx.AsyncClient()
        self.processing_count = 0
        self.success_count = 0
        self.error_count = 0
        self.model = "gpt-4o-mini"

    async def initialize(self) -> None:
        """Initialize graph tools and connections."""
        try:
            # Initialize Neo4j connection and embedding provider
            neo4j_client = await get_neo4j_connection()
            embedding_provider = get_embedding_provider()
            
            # Initialize graph maintenance tools
            self.graph_tools = GraphMaintenanceTools(
                neo4j_client=neo4j_client,
                embedding_provider=embedding_provider
            )
            
            logger.info("✅ Semantic processor initialized with GraphMaintenanceTools")
        except Exception as e:
            logger.error(f"❌ Failed to initialize semantic processor: {e}")
            raise
    
    async def _execute_agent_loop(
        self, 
        user_id: str, 
        user_message: str, 
        assistant_response: str
    ) -> dict[str, Any]:
        """Execute OpenAI Tool-Use Agent loop for knowledge graph curation."""
        
        # System prompt defining the Knowledge Graph Curator persona
        system_prompt = """
You are the Knowledge Graph Curator. Your job is to accurately reflect the user's latest conversation into the Graph.

ALWAYS search for existing entities before creating new ones to avoid duplicates.

Your workflow:
1. First, search for similar entities using search_similar_nodes to avoid duplicates
2. Create or update entities using upsert_node (with merge_id if updating existing)
3. Create relationships between entities using create_relationship
4. Focus on permanent facts, not temporary events
5. When done, provide a final status message

Rules:
- The user is ALWAYS the entity 'User' - if they reveal their name, add it as a property
- Only store semantic knowledge (facts, preferences, relationships), not episodic events
- Use specific relationship types (HAS_MOTHER, WORKS_AT, LIKES, etc.)
- Search before creating to prevent duplicates
"""
        
        # Initial conversation context
        initial_message = f"""
Analyze this conversation and update the knowledge graph accordingly:

User: {user_message}
Assistant: {assistant_response}

Start by searching for any entities mentioned to check for existing knowledge.
"""
        
        # Initialize message history
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_message}
        ]
        
        # Define available tools for the agent
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_similar_nodes",
                    "description": "Search for similar entities in the knowledge graph",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query string"
                            },
                            "threshold": {
                                "type": "number",
                                "description": "Similarity threshold (0.0-1.0)",
                                "default": 0.8
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "upsert_node",
                    "description": "Create or update a node in the knowledge graph",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Node name"
                            },
                            "label": {
                                "type": "string",
                                "description": "Node label/type"
                            },
                            "properties": {
                                "type": "object",
                                "description": "Node properties as key-value pairs",
                                "default": {}
                            },
                            "merge_id": {
                                "type": "string",
                                "description": "Optional existing node ID to update"
                            }
                        },
                        "required": ["name", "label"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_relationship",
                    "description": "Create a relationship between two nodes",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "from_name": {
                                "type": "string",
                                "description": "Source node name"
                            },
                            "to_name": {
                                "type": "string",
                                "description": "Target node name"
                            },
                            "relation_type": {
                                "type": "string",
                                "description": "Relationship type"
                            },
                            "properties": {
                                "type": "object",
                                "description": "Relationship properties",
                                "default": {}
                            }
                        },
                        "required": ["from_name", "to_name", "relation_type"]
                    }
                }
            }
        ]
        
        # Execute the tool-calling loop with safety limit
        max_steps = 10
        step = 0
        
        while step < max_steps:
            step += 1
            logger.info(f"🤖 Agent loop step {step}/{max_steps}")
            
            # Make API call to OpenAI
            response = await self._call_openai_with_tools(messages, tools)
            
            # Add assistant response to message history
            messages.append(response["choices"][0]["message"])
            
            # Check if assistant wants to call tools
            assistant_message = response["choices"][0]["message"]
            
            if "tool_calls" in assistant_message and assistant_message["tool_calls"]:
                logger.info(f"🔧 Agent requesting {len(assistant_message['tool_calls'])} tool calls")
                
                # Execute each tool call
                for tool_call in assistant_message["tool_calls"]:
                    function_name = tool_call["function"]["name"]
                    arguments = json.loads(tool_call["function"]["arguments"])
                    
                    logger.info(f"🛠️ Executing {function_name} with args: {arguments}")
                    
                    # Execute the tool
                    if function_name == "search_similar_nodes":
                        tool_result = await self.graph_tools.search_similar_nodes(**arguments, user_id=user_id)
                    elif function_name == "upsert_node":
                        tool_result = await self.graph_tools.upsert_node(**arguments, user_id=user_id)
                    elif function_name == "create_relationship":
                        tool_result = await self.graph_tools.create_relationship(**arguments)
                    else:
                        tool_result = f"Error: Unknown function {function_name}"
                    
                    # Add tool result to message history
                    messages.append({
                        "role": "tool",
                        "content": tool_result,
                        "tool_call_id": tool_call["id"]
                    })
                    
                    logger.info(f"✓ Tool result: {tool_result[:100]}...")
            else:
                # No more tool calls, agent is done
                final_content = assistant_message.get("content", "Processing completed")
                logger.info(f"✅ Agent loop completed after {step} steps: {final_content}")
                
                return {
                    "reasoning": final_content,
                    "steps": step
                }
        
        # Max steps reached
        logger.warning(f"⚠️ Agent loop reached max steps ({max_steps})")
        return {
            "reasoning": "Agent loop reached maximum steps limit",
            "steps": max_steps
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.RequestError),
        reraise=True
    )
    async def _call_openai_with_tools(
        self, 
        messages: list[dict], 
        tools: list[dict]
    ) -> dict[str, Any]:
        """Make API call to OpenAI with tools support."""
        if not settings.openai_api_key:
            raise Exception("OpenAI API key not configured")
        
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        try:
            response = await self.http_client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30.0
            )
            response.raise_for_status()
            
            return response.json()
        
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("OpenAI API rate limit hit, retrying...")
                raise  # Will be retried by tenacity
            else:
                logger.error(
                    f"OpenAI API error {e.response.status_code}: {e.response.text}"
                )
                raise Exception(
                    f"API error: {e.response.status_code}"
                ) from e
        
        except (httpx.RequestError, json.JSONDecodeError) as e:
            logger.error(f"Request failed: {str(e)}")
            raise Exception(f"Request failed: {str(e)}") from e

    async def cleanup(self) -> None:
        """Cleanup semantic processor resources."""
        if self.http_client:
            await self.http_client.aclose()
        logger.info("🧹 Semantic processor cleaned up")

    async def process_job(self, job: MemoryReflectionJob) -> dict[str, Any]:
        """Process semantic memory using OpenAI Tool-Use Agent loop."""
        start_time = datetime.utcnow()
        self.processing_count += 1

        try:
            logger.debug(f"🕸️ Processing semantic memory for job {job.job_id}")
            
            if not self.graph_tools:
                raise Exception("Graph tools not initialized")

            # Execute the agent loop
            result = await self._execute_agent_loop(
                user_id=job.user_id,
                user_message=job.user_message,
                assistant_response=job.assistant_response,
            )

            self.success_count += 1
            processing_time = (
                datetime.utcnow() - start_time
            ).total_seconds() * 1000

            logger.info(
                f"✅ Semantic processing completed for job {job.job_id} in {processing_time:.1f}ms"
            )

            return {
                "status": "completed",
                "result_id": f"semantic_{job.job_id}",
                "reasoning": result.get("reasoning", "Agent processing completed"),
                "processing_time_ms": processing_time,
                "agent_steps": result.get("steps", 0),
            }

        except Exception as e:
            self.error_count += 1
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.error(
                f"❌ Semantic processing failed for job {job.job_id} in {processing_time:.1f}ms: {e}"
            )
            raise



    def get_health(self) -> dict[str, Any]:
        """Get processor health metrics."""
        success_rate = self.success_count / max(self.processing_count, 1)
        return {
            "processor": "semantic",
            "healthy": True,
            "total_processed": self.processing_count,
            "successful": self.success_count,
            "errors": self.error_count,
            "success_rate": success_rate,
            "graph_tools_initialized": self.graph_tools is not None,
        }
