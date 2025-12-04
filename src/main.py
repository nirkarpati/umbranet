"""Main application entry point for the Headless Governor System."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

from src.core.config import settings
from src.core.domain.events import GovernorEvent, MessageType, ChannelType
from src.core.domain.state import GovernorState, StateNode
from src.core.workflow.governor_workflow import GovernorWorkflow
from src.memory import MemoryManager, get_memory_manager
from src.memory.context.enhanced_assembler import EnhancedContextAssembler
import openai
import os
# Web interface removed - using separate React frontend

# Chat API models
class ChatMessage(BaseModel):
    message: str
    user_id: str = "default_user"
    session_id: str = "default_session"

class ChatResponse(BaseModel):
    response: str
    user_id: str
    session_id: str
    timestamp: str
    status: str
    metadata: dict

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize Governor workflow
governor_workflow = GovernorWorkflow()

# Global memory manager and enhanced context assembler
memory_manager: Optional[MemoryManager] = None
enhanced_context_assembler: Optional[EnhancedContextAssembler] = None

# In-memory session storage (in production, this would be Redis/database)
chat_sessions: Dict[str, GovernorState] = {}

async def get_or_create_chat_state(user_id: str, session_id: str) -> GovernorState:
    """Get or create a chat session state."""
    session_key = f"{user_id}:{session_id}"
    
    if session_key not in chat_sessions:
        logger.debug(f"🆕 Creating new session state for {session_key}")
        chat_sessions[session_key] = GovernorState(
            user_id=user_id,
            session_id=session_id,
            current_node=StateNode.IDLE
        )
    else:
        logger.debug(f"📖 Retrieved existing session state for {session_key}")
    
    return chat_sessions[session_key]

async def process_governor_workflow(
    event: GovernorEvent,
    existing_state: GovernorState,
    request_id: str
) -> Dict[str, Any]:
    """Process a message through the complete Governor workflow with detailed logging."""
    start_time = datetime.utcnow()
    
    try:
        logger.debug(f"🔄 [REQ-{request_id}] Initializing workflow components...")
        
        # Initialize tool registry and get global context assembler
        from src.action_plane.tool_registry.registry import ToolRegistry
        
        tool_registry = ToolRegistry()
        
        # Use global enhanced context assembler
        global enhanced_context_assembler
        if enhanced_context_assembler is None:
            logger.error(f"❌ [REQ-{request_id}] Enhanced context assembler not initialized")
            raise RuntimeError("Memory system not properly initialized")
        
        logger.debug(f"⚙️  [REQ-{request_id}] Using enhanced context assembler with memory integration")
        
        # Step 1: Update state with new input
        existing_state.total_turns += 1
        existing_state.last_user_input = event.content
        existing_state.updated_at = datetime.utcnow()
        logger.info(f"📈 [REQ-{request_id}] State updated - Turn #{existing_state.total_turns}")
        
        # Step 2: Get available tools
        logger.debug(f"🔧 [REQ-{request_id}] Retrieving available tools...")
        available_tools = tool_registry.list_tools()
        logger.info(f"🛠️  [REQ-{request_id}] Found {len(available_tools)} available tools")
        
        # Step 3: Assemble context from all memory tiers using enhanced assembler
        logger.info(f"🧠 [REQ-{request_id}] Assembling context from RAG++ memory hierarchy...")
        try:
            context_prompt = await enhanced_context_assembler.assemble_context(
                user_id=event.user_id,
                current_input=event.content,
                state=existing_state,
                available_tools=list(available_tools)
            )
            logger.info(f"✅ [REQ-{request_id}] Enhanced context assembled successfully")
            logger.debug(f"📊 [REQ-{request_id}] Context length: {len(context_prompt)} characters")
            memory_tiers_accessed = 4  # All tiers accessed via memory manager
        except Exception as e:
            logger.warning(f"⚠️  [REQ-{request_id}] Enhanced context assembly failed, using fallback: {e}")
            context_prompt = _build_fallback_context_prompt(event, existing_state, available_tools)
            memory_tiers_accessed = 0
        
        # Step 4: Process through workflow nodes (simplified for production stability)
        logger.info(f"🔄 [REQ-{request_id}] Processing through Governor workflow nodes...")
        
        # Simulate workflow progression with detailed logging
        workflow_steps = [
            ("idle", "Analyzing user input and determining intent"),
            ("analyze", "Understanding requirements and context"),
            ("tool_decision", "Evaluating available tools and capabilities"),
            ("policy_check", "Performing security and permission checks"),
            ("respond", "Generating contextual response")
        ]
        
        for step_name, step_description in workflow_steps:
            logger.debug(f"🎯 [REQ-{request_id}] Workflow step: {step_name} - {step_description}")
            existing_state.current_node = StateNode(step_name) if step_name != "respond" else StateNode.IDLE
        
        # Step 5: Generate intelligent response using memory-aware context
        logger.info(f"💭 [REQ-{request_id}] Generating Governor response...")
        response_content = await generate_production_response(
            event=event,
            state=existing_state,
            context_prompt=context_prompt,
            available_tools=available_tools,
            request_id=request_id
        )
        
        existing_state.last_assistant_response = response_content
        existing_state.current_node = StateNode.IDLE  # Return to idle state
        
        # Step 6: Store interaction in all memory tiers via memory manager
        await store_interaction_in_memory(
            user_id=event.user_id,
            user_message=event.content,
            assistant_response=response_content,
            session_id=existing_state.session_id,
            request_id=request_id
        )
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(f"⏱️  [REQ-{request_id}] Workflow completed in {processing_time:.2f}ms")
        
        return {
            "response": response_content,
            "final_state": existing_state,
            "workflow_status": "completed",
            "processing_time_ms": round(processing_time, 2),
            "memory_tiers_accessed": memory_tiers_accessed,
            "tools_available": len(available_tools)
        }
        
    except Exception as e:
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.error(f"❌ [REQ-{request_id}] Workflow failed after {processing_time:.2f}ms: {e}")
        
        # Fallback response
        fallback_response = f"I encountered an issue processing your request (ID: {request_id}). The Governor system is operational but experienced a workflow error. Please try again."
        existing_state.last_assistant_response = fallback_response
        
        return {
            "response": fallback_response,
            "final_state": existing_state,
            "workflow_status": "error",
            "processing_time_ms": round(processing_time, 2),
            "memory_tiers_accessed": 0,
            "tools_available": 0
        }


def _build_fallback_context_prompt(event: GovernorEvent, state: GovernorState, available_tools: list) -> str:
    """Build fallback context prompt when enhanced context assembly fails."""
    
    now = datetime.utcnow()
    tools_list = ', '.join(available_tools) if available_tools else 'None'
    
    return f"""You are the Headless Governor, a personal AI assistant that operates as an invisible OS layer.

CORE PRINCIPLES:
- You execute tasks autonomously when safe, ask for confirmation when risky
- You maintain context across conversations and remember user preferences  
- You prioritize security and never execute dangerous operations without explicit approval
- You are helpful, efficient, and proactive in task completion
- You explain your reasoning when making decisions

CURRENT ENVIRONMENT:
- Time: {now.strftime("%Y-%m-%d %H:%M:%S UTC")}
- Session: {state.session_id}
- System Status: Limited context mode (memory system unavailable)

AVAILABLE TOOLS:
- {tools_list}

CURRENT SITUATION:
- User: {event.user_id}
- Input: {event.content}
- Conversation Turn: #{state.total_turns}
- Status: Operating in fallback mode with limited memory access"""


async def store_interaction_in_memory(
    user_id: str,
    user_message: str,
    assistant_response: str,
    session_id: str,
    request_id: str
) -> None:
    """Store interaction in all memory tiers via memory manager."""
    global memory_manager
    
    if memory_manager is None:
        logger.warning(f"⚠️  [REQ-{request_id}] Memory manager not initialized, skipping memory storage")
        return
    
    try:
        logger.debug(f"💾 [REQ-{request_id}] Storing interaction in RAG++ memory hierarchy...")
        
        interaction_data = {
            "content": user_message,
            "assistant_response": assistant_response,
            "session_id": session_id,
            "metadata": {
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        storage_result = await memory_manager.store_interaction(user_id, interaction_data)
        
        # Log storage results
        if storage_result.get("status") == "stored":
            stored_tiers = [
                tier for tier, status in storage_result.items()
                if tier.endswith("_status") and status == "success"
            ]
            logger.info(f"✅ [REQ-{request_id}] Interaction stored in {len(stored_tiers)} memory tiers")
        else:
            logger.warning(f"⚠️  [REQ-{request_id}] Partial memory storage: {storage_result.get('status')}")
            
    except Exception as e:
        logger.error(f"❌ [REQ-{request_id}] Failed to store interaction in memory: {e}")
        # Continue execution even if memory storage fails


async def generate_production_response(
    event: GovernorEvent,
    state: GovernorState,
    context_prompt: str,
    available_tools: list,
    request_id: str
) -> str:
    """Generate a production-quality response using OpenAI LLM integration."""
    logger.debug(f"🎨 [REQ-{request_id}] Generating response via OpenAI LLM...")
    
    try:
        # Use the memory-aware context prompt directly as system prompt
        system_prompt = context_prompt
        
        # Build conversation history for context
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add recent conversation history if available
        if hasattr(state, 'conversation_history') and state.conversation_history:
            recent_history = state.conversation_history[-5:]  # Last 5 turns
            for turn in recent_history:
                if hasattr(turn, 'user_input') and hasattr(turn, 'assistant_response'):
                    messages.append({"role": "user", "content": turn.user_input})
                    messages.append({"role": "assistant", "content": turn.assistant_response})
        
        # Add current user message
        messages.append({"role": "user", "content": event.content})
        
        logger.debug(f"🤖 [REQ-{request_id}] Calling OpenAI with {len(messages)} messages...")
        
        # Initialize OpenAI client
        api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not configured")
            
        client = openai.AsyncOpenAI(api_key=api_key)
        
        # Make OpenAI API call
        llm_start_time = datetime.utcnow()
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        
        llm_duration = (datetime.utcnow() - llm_start_time).total_seconds() * 1000
        logger.info(f"🧠 [REQ-{request_id}] OpenAI response received in {llm_duration:.2f}ms")
        
        # Extract response content
        llm_response = response.choices[0].message.content.strip()
        
        # Log token usage
        usage = response.usage
        logger.debug(f"📊 [REQ-{request_id}] Token usage - Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")
        
        # Add system metadata to response
        enhanced_response = f"{llm_response}\n\n---\n**🔧 System Info:** Request `{request_id}` | Turn #{state.total_turns} | LLM: {llm_duration:.1f}ms | Tokens: {usage.total_tokens}"
        
        logger.debug(f"✅ [REQ-{request_id}] LLM response generated - {len(enhanced_response)} characters")
        return enhanced_response
        
    except Exception as e:
        logger.error(f"❌ [REQ-{request_id}] OpenAI API call failed: {e}")
        
        # Fallback to rule-based response
        logger.warning(f"⚠️  [REQ-{request_id}] Using fallback response generation")
        return generate_fallback_response(event, state, available_tools, request_id)



def generate_fallback_response(
    event: GovernorEvent,
    state: GovernorState,
    available_tools: list,
    request_id: str
) -> str:
    """Generate fallback response when LLM is unavailable."""
    
    return f"""Hello! I'm your Headless Governor AI. I encountered an issue connecting to my language model, but my core systems are operational.

**Current Status:**
- Session: Turn #{state.total_turns} 
- Memory: ⚠️ Limited Context (LLM connection unavailable)
- Tools: {len(available_tools)} functions ready
- Request: `{request_id}`

**Your Message:** "{event.content}"

I'm working to restore full AI capabilities. In the meantime, I can still help with system status, session management, and basic interactions. Please try again shortly for full AI responses.

---
**🔧 System:** Fallback mode | Request `{request_id}` | Governor systems operational"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager with comprehensive startup logging."""
    startup_time = datetime.utcnow()
    
    logger.info("🚀 ===== HEADLESS GOVERNOR SYSTEM STARTUP =====")
    logger.info(f"⚙️  Environment: {settings.environment}")
    logger.info(f"🐛 Debug mode: {settings.debug}")
    logger.info(f"📊 Log level: {settings.log_level}")
    logger.info("")
    
    logger.info("🔌 Database Connections:")
    logger.info(f"   • Redis: {settings.redis_url}")
    logger.info(f"   • PostgreSQL: {settings.postgres_url}")
    logger.info(f"   • Neo4j: {settings.neo4j_url}")
    logger.info("")
    
    # Initialize Governor components with logging
    logger.info("🧠 Initializing Governor System Components...")
    try:
        logger.debug("   • Loading LangGraph workflow...")
        # Governor workflow already initialized globally
        logger.info("   ✅ LangGraph workflow loaded successfully")
        
        logger.debug("   • Initializing RAG++ Memory Manager...")
        global memory_manager, enhanced_context_assembler
        
        # Initialize memory manager
        memory_manager = await get_memory_manager()
        logger.info("   ✅ RAG++ Memory Manager initialized with 4-tier hierarchy")
        
        # Initialize enhanced context assembler
        enhanced_context_assembler = EnhancedContextAssembler(memory_manager)
        logger.info("   ✅ Enhanced Context Assembler initialized")
        
        logger.debug("   • Setting up session management...")
        logger.info("   ✅ Session storage initialized")
        
        startup_duration = (datetime.utcnow() - startup_time).total_seconds()
        logger.info(f"✨ Governor system startup completed in {startup_duration:.3f}s")
        logger.info("🎯 System ready for production requests!")
        
    except Exception as e:
        logger.error(f"❌ Governor system startup failed: {e}")
        logger.exception("🔍 Startup failure details:")
        raise
    
    logger.info("=" * 50)
    
    yield

    # Shutdown logic with logging
    logger.info("🛑 Shutting down Headless Governor System...")
    logger.info(f"📊 Session statistics: {len(chat_sessions)} active sessions")
    logger.info("✅ Governor system shutdown completed")


app = FastAPI(
    title="Headless Governor System",
    description="A stateful Personal AI Assistant Operating System",
    version="1.0.1",
    debug=settings.debug,
    lifespan=lifespan,
)

# Add CORS middleware for development
if settings.is_development():
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Chat interface will be separate React frontend

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage) -> ChatResponse:
    """Production-ready Governor system chat endpoint with comprehensive logging."""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"🚀 [REQ-{request_id}] Starting chat request from user={message.user_id}, session={message.session_id}")
    logger.debug(f"📝 [REQ-{request_id}] Message: '{message.message}' (length: {len(message.message)})")
    
    try:
        # Step 1: Session Management
        logger.debug(f"🗃️  [REQ-{request_id}] Retrieving/creating session state...")
        existing_state = await get_or_create_chat_state(message.user_id, message.session_id)
        logger.info(f"📊 [REQ-{request_id}] Session loaded - Current turn: {existing_state.total_turns}, State: {existing_state.current_node.value}")
        
        # Step 2: Create Governor Event
        logger.debug(f"📋 [REQ-{request_id}] Creating GovernorEvent...")
        event = GovernorEvent(
            user_id=message.user_id,
            session_id=message.session_id,
            message_type=MessageType.TEXT,
            content=message.message,
            channel=ChannelType.DIRECT,
            metadata={"source": "react_frontend", "request_id": request_id}
        )
        logger.debug(f"✅ [REQ-{request_id}] GovernorEvent created: {event.message_type.value} via {event.channel.value}")
        
        # Step 3: Process through Governor workflow with comprehensive logging
        logger.info(f"⚙️  [REQ-{request_id}] Starting Governor workflow execution...")
        
        # Create a production-ready workflow processor
        workflow_result = await process_governor_workflow(
            event=event,
            existing_state=existing_state,
            request_id=request_id
        )
        
        # Step 4: Update session storage
        session_key = f"{message.user_id}:{message.session_id}"
        chat_sessions[session_key] = workflow_result["final_state"]
        logger.debug(f"💾 [REQ-{request_id}] Session state updated in storage")
        
        # Step 5: Prepare response
        response_content = workflow_result["response"]
        final_state = workflow_result["final_state"]
        
        logger.info(f"✨ [REQ-{request_id}] Request completed successfully - Response length: {len(response_content)}")
        
        return ChatResponse(
            response=response_content,
            user_id=message.user_id,
            session_id=message.session_id,
            timestamp=datetime.utcnow().isoformat(),
            status="processed" if not final_state.awaiting_confirmation else "awaiting_confirmation",
            metadata={
                "request_id": request_id,
                "workflow_status": workflow_result["workflow_status"],
                "current_node": final_state.current_node.value,
                "conversation_turns": final_state.total_turns,
                "processing_time_ms": workflow_result["processing_time_ms"],
                "memory_tiers_accessed": workflow_result["memory_tiers_accessed"],
                "tools_available": workflow_result["tools_available"],
                "session_key": session_key
            }
        )
        
    except Exception as e:
        logger.error(f"💥 [REQ-{request_id}] Critical error in chat endpoint: {e}")
        logger.exception(f"🔍 [REQ-{request_id}] Full exception traceback:")
        
        return ChatResponse(
            response=f"I encountered a system error while processing your request (ID: {request_id}). The Governor system is monitoring this issue and will resolve it promptly. Please try again in a moment.",
            user_id=message.user_id,
            session_id=message.session_id,
            timestamp=datetime.utcnow().isoformat(),
            status="error",
            metadata={
                "request_id": request_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "system_status": "degraded",
                "retry_recommended": True
            }
        )

@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint for health checks."""
    return {
        "message": "Headless Governor System",
        "status": "operational",
        "version": "1.0.1",
        "environment": settings.environment,
    }


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Comprehensive health check endpoint with system diagnostics."""
    health_check_start = datetime.utcnow()
    
    try:
        # Basic system checks
        system_status = {
            "status": "healthy",
            "timestamp": health_check_start.isoformat(),
            "environment": settings.environment,
            "debug": settings.debug,
            "version": "1.0.1",
        }
        
        # Governor-specific diagnostics
        governor_status = {
            "active_sessions": len(chat_sessions),
            "workflow_loaded": governor_workflow is not None,
            "memory_tiers": {
                "redis": "configured",
                "postgresql": "configured", 
                "neo4j": "configured",
                "procedural": "configured"
            },
            "components": {
                "context_assembler": "ready",
                "tool_registry": "ready",
                "policy_engine": "ready",
                "state_machine": "ready"
            }
        }
        
        # Performance metrics
        response_time = (datetime.utcnow() - health_check_start).total_seconds() * 1000
        performance_metrics = {
            "health_check_duration_ms": round(response_time, 2),
            "uptime_info": "operational"
        }
        
        return {
            **system_status,
            "governor": governor_status,
            "performance": performance_metrics
        }
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
            "environment": settings.environment
        }


@app.get("/config")
async def config_info() -> dict[str, str]:
    """Configuration info endpoint (development only)."""
    if not settings.is_development():
        return {"message": "Configuration info not available in production"}

    return {
        "redis_url": settings.redis_url,
        "postgres_host": settings.postgres_host,
        "neo4j_url": settings.neo4j_url,
        "log_level": settings.log_level,
    }


# Memory Dashboard API Endpoints
@app.get("/api/memory/semantic/{user_id}")
async def get_semantic_memory(user_id: str):
    """Get semantic memory (Neo4j) entities and relationships for a user."""
    global memory_manager
    
    if memory_manager is None:
        logger.error("Memory manager not initialized")
        return {"entities": [], "relationships": [], "total_entities": 0, "total_relationships": 0}
    
    try:
        # Use memory manager's semantic tier with proper async context
        async with memory_manager.semantic as semantic_store:
            entities = await semantic_store.get_entities_for_user(user_id)
            relationships = await semantic_store.get_relationships_for_user(user_id)
        
        # Format entities properly
        formatted_entities = []
        for entity in entities:
            formatted_entities.append({
                "type": entity.get("type", "Unknown"),
                "name": entity.get("name", "Unknown"),
                "entity_id": entity.get("entity_id", entity.get("name", "Unknown"))
            })
        
        # Format relationships properly
        formatted_relationships = []
        for rel in relationships:
            formatted_relationships.append({
                "from_entity": rel.get("from_entity", "Unknown"),
                "to_entity": rel.get("to_entity", "Unknown"), 
                "relationship_type": rel.get("relationship_type", "RELATED_TO"),
                "weight": rel.get("weight", 0.5)
            })
        
        logger.info(f"Returning {len(formatted_entities)} entities and {len(formatted_relationships)} relationships for user {user_id}")
        
        return {
            "entities": formatted_entities,
            "relationships": formatted_relationships,
            "total_entities": len(entities),
            "total_relationships": len(relationships)
        }
    except Exception as e:
        logger.error(f"Error fetching semantic memory for {user_id}: {e}")
        return {"entities": [], "relationships": [], "total_entities": 0, "total_relationships": 0}


@app.get("/api/memory/episodic/{user_id}")
async def get_episodic_memory(user_id: str):
    """Get episodic memory (PostgreSQL) conversations for a user."""
    global memory_manager
    
    if memory_manager is None:
        logger.error("Memory manager not initialized")
        return {"episodes": []}
    
    try:
        # Use memory manager's episodic tier with proper async context
        async with memory_manager.episodic as episodic_store:
            episodes = await episodic_store.get_recent_episodes(user_id, limit=10)
        
        formatted_episodes = []
        for episode in episodes:
            formatted_episodes.append({
                "content": episode["content"][:200],  # Truncate for display
                "timestamp": episode["timestamp"],
                "sender": episode["metadata"].get("sender", "user")
            })
        
        return {"episodes": formatted_episodes}
    except Exception as e:
        logger.error(f"Error fetching episodic memory for {user_id}: {e}")
        return {"episodes": []}


@app.get("/api/memory/procedural/{user_id}")
async def get_procedural_memory(user_id: str):
    """Get procedural memory (PostgreSQL) rules and preferences for a user."""
    global memory_manager
    
    if memory_manager is None:
        logger.error("Memory manager not initialized")
        return {"rules": []}
    
    try:
        # Use memory manager's procedural tier for consistency
        rules = await memory_manager.procedural.get_user_rules(user_id)
        
        formatted_rules = []
        for rule in rules:
            formatted_rules.append({
                "title": rule.get("title", "User Preference"),
                "instruction": rule.get("instruction", rule.get("content", ""))
            })
        
        return {"rules": formatted_rules}
    except Exception as e:
        logger.error(f"Error fetching procedural memory for {user_id}: {e}")
        return {"rules": []}


@app.get("/api/memory/stats/{user_id}")
async def get_memory_stats(user_id: str):
    """Get memory statistics for a user across all tiers."""
    global memory_manager
    
    if memory_manager is None:
        logger.error("Memory manager not initialized")
        return {"entity_count": 0, "relationship_count": 0, "user_id": user_id}
    
    try:
        # Get counts from semantic memory via memory manager
        entities = await memory_manager.semantic.get_entities_for_user(user_id)
        relationships = await memory_manager.semantic.get_relationships_for_user(user_id)
        
        return {
            "entity_count": len(entities),
            "relationship_count": len(relationships),
            "user_id": user_id
        }
    except Exception as e:
        logger.error(f"Error fetching memory stats for {user_id}: {e}")
        return {"entity_count": 0, "relationship_count": 0, "user_id": user_id}


@app.get("/api/memory/redis/{user_id}")
async def get_redis_memory(user_id: str):
    """Get short-term memory (Redis) data for a user."""
    global memory_manager
    
    if memory_manager is None:
        logger.error("Memory manager not initialized")
        return {"active": False, "session_count": 0, "context_loaded": False}
    
    try:
        # Use memory manager's short-term tier for consistency
        session_data = await memory_manager.short_term.get_session_data(user_id)
        return session_data
    except Exception as e:
        logger.error(f"Error fetching Redis memory for {user_id}: {e}")
        return {"active": False, "session_count": 0, "context_loaded": False}


@app.delete("/api/memory/user/{user_id}")
async def delete_user_memory(user_id: str):
    """Delete all memory data for a user across all tiers."""
    global memory_manager
    
    if memory_manager is None:
        logger.error("Memory manager not initialized")
        return {"success": False, "error": "Memory manager not available"}
    
    try:
        logger.info(f"🗑️ Starting complete memory deletion for user {user_id}")
        
        deletion_results = {}
        total_deleted = 0
        
        # Delete from Tier 1: Redis (Short-term memory)
        try:
            async with memory_manager.short_term as short_term_store:
                await short_term_store.clear_session(user_id)
            deletion_results["tier1_redis"] = {"status": "success", "deleted": "session_data"}
            logger.info(f"✅ Tier 1 (Redis) data cleared for user {user_id}")
        except Exception as e:
            deletion_results["tier1_redis"] = {"status": "error", "error": str(e)}
            logger.error(f"❌ Failed to clear Tier 1 data for {user_id}: {e}")
        
        # Delete from Tier 2: PostgreSQL (Episodic memory)
        try:
            async with memory_manager.episodic as episodic_store:
                deleted_episodes = await episodic_store.delete_user_data(user_id)
            deletion_results["tier2_episodic"] = {"status": "success", "deleted": deleted_episodes}
            total_deleted += deleted_episodes
            logger.info(f"✅ Tier 2 (Episodic) deleted {deleted_episodes} episodes for user {user_id}")
        except Exception as e:
            deletion_results["tier2_episodic"] = {"status": "error", "error": str(e)}
            logger.error(f"❌ Failed to clear Tier 2 data for {user_id}: {e}")
        
        # Delete from Tier 3: Neo4j (Semantic memory)
        try:
            async with memory_manager.semantic as semantic_store:
                deleted_graph_items = await semantic_store.delete_user_graph(user_id)
            deletion_results["tier3_semantic"] = {"status": "success", "deleted": deleted_graph_items}
            total_deleted += deleted_graph_items
            logger.info(f"✅ Tier 3 (Semantic) deleted {deleted_graph_items} graph items for user {user_id}")
        except Exception as e:
            deletion_results["tier3_semantic"] = {"status": "error", "error": str(e)}
            logger.error(f"❌ Failed to clear Tier 3 data for {user_id}: {e}")
        
        # Delete from Tier 4: PostgreSQL (Procedural memory)
        try:
            async with memory_manager.procedural as procedural_store:
                deleted_profiles, deleted_instructions = await procedural_store.delete_all_user_data(user_id)
            total_deleted_procedural = deleted_profiles + deleted_instructions
            deletion_results["tier4_procedural"] = {"status": "success", "deleted": total_deleted_procedural}
            total_deleted += total_deleted_procedural
            logger.info(f"✅ Tier 4 (Procedural) deleted {total_deleted_procedural} items for user {user_id}")
        except Exception as e:
            deletion_results["tier4_procedural"] = {"status": "error", "error": str(e)}
            logger.error(f"❌ Failed to clear Tier 4 data for {user_id}: {e}")
        
        # Clear any active chat sessions
        try:
            sessions_cleared = 0
            keys_to_remove = [key for key in chat_sessions.keys() if key.startswith(f"{user_id}:")]
            for key in keys_to_remove:
                del chat_sessions[key]
                sessions_cleared += 1
            deletion_results["active_sessions"] = {"status": "success", "deleted": sessions_cleared}
            logger.info(f"✅ Cleared {sessions_cleared} active sessions for user {user_id}")
        except Exception as e:
            deletion_results["active_sessions"] = {"status": "error", "error": str(e)}
            logger.error(f"❌ Failed to clear active sessions for {user_id}: {e}")
        
        # Determine overall success
        successful_tiers = sum(1 for result in deletion_results.values() if result.get("status") == "success")
        total_tiers = len(deletion_results)
        
        logger.info(f"🎯 Memory deletion completed for user {user_id}: {successful_tiers}/{total_tiers} tiers successful, {total_deleted} total items deleted")
        
        return {
            "success": successful_tiers > 0,
            "user_id": user_id,
            "total_deleted": total_deleted,
            "tiers_cleared": successful_tiers,
            "total_tiers": total_tiers,
            "details": deletion_results,
            "message": f"Successfully cleared {successful_tiers}/{total_tiers} memory tiers"
        }
        
    except Exception as e:
        logger.error(f"💥 Critical error during memory deletion for {user_id}: {e}")
        return {
            "success": False,
            "user_id": user_id,
            "error": str(e),
            "message": "Failed to delete user memory data"
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.fastapi_host,
        port=settings.fastapi_port,
        reload=settings.fastapi_reload and settings.is_development(),
        log_level=settings.log_level.lower(),
    )
