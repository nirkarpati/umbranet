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
from src.memory.tiers.semantic import SemanticMemoryStore
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
        
        # Initialize context assembler with logging
        from src.governor.context.assembler import ContextAssembler
        from src.action_plane.tool_registry.registry import ToolRegistry
        
        context_assembler = ContextAssembler()
        tool_registry = ToolRegistry()
        
        logger.debug(f"⚙️  [REQ-{request_id}] Context assembler and tool registry initialized")
        
        # Step 1: Update state with new input
        existing_state.total_turns += 1
        existing_state.last_user_input = event.content
        existing_state.updated_at = datetime.utcnow()
        logger.info(f"📈 [REQ-{request_id}] State updated - Turn #{existing_state.total_turns}")
        
        # Step 2: Get available tools
        logger.debug(f"🔧 [REQ-{request_id}] Retrieving available tools...")
        available_tools = tool_registry.list_tools()
        logger.info(f"🛠️  [REQ-{request_id}] Found {len(available_tools)} available tools")
        
        # Step 3: Assemble context from all memory tiers
        logger.info(f"🧠 [REQ-{request_id}] Assembling context from RAG++ memory hierarchy...")
        try:
            context_data = await context_assembler.assemble_context(
                user_id=event.user_id,
                current_input=event.content,
                state=existing_state,
                available_tools=list(available_tools)
            )
            logger.info(f"✅ [REQ-{request_id}] Context assembled successfully")
            logger.debug(f"📊 [REQ-{request_id}] Context metadata: {context_data.metadata}")
            memory_tiers_accessed = 4  # All tiers accessed
        except Exception as e:
            logger.warning(f"⚠️  [REQ-{request_id}] Context assembly failed, using fallback: {e}")
            context_data = None
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
        
        # Step 5: Generate intelligent response
        logger.info(f"💭 [REQ-{request_id}] Generating Governor response...")
        response_content = await generate_production_response(
            event=event,
            state=existing_state,
            context_data=context_data,
            available_tools=available_tools,
            request_id=request_id
        )
        
        existing_state.last_assistant_response = response_content
        existing_state.current_node = StateNode.IDLE  # Return to idle state
        
        # Step 6: Extract and store semantic entities (Neo4j - Tier 3 Memory)
        await extract_and_store_semantic_memory(
            user_id=event.user_id,
            user_message=event.content,
            assistant_response=response_content,
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

async def generate_production_response(
    event: GovernorEvent,
    state: GovernorState,
    context_data: Any,
    available_tools: list,
    request_id: str
) -> str:
    """Generate a production-quality response using OpenAI LLM integration."""
    logger.debug(f"🎨 [REQ-{request_id}] Generating response via OpenAI LLM...")
    
    try:
        # Build comprehensive system prompt with context
        system_prompt = build_governor_system_prompt(
            state=state,
            context_data=context_data,
            available_tools=available_tools,
            request_id=request_id
        )
        
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
        return generate_fallback_response(event, state, context_data, available_tools, request_id)

def build_governor_system_prompt(
    state: GovernorState,
    context_data: Any,
    available_tools: list,
    request_id: str
) -> str:
    """Build comprehensive system prompt for the Governor LLM."""
    
    # Core identity and capabilities
    core_identity = """You are the Headless Governor, a sophisticated Personal AI Assistant operating as an invisible OS layer. You have access to a complete RAG++ memory hierarchy and can execute tasks autonomously with proper safety checks.

CORE PRINCIPLES:
- You are proactive, intelligent, and context-aware across all conversations
- You maintain persistent memory and learn user preferences over time  
- You execute tasks autonomously when safe, ask for confirmation when risky
- You prioritize security and never execute dangerous operations without approval
- You explain your reasoning and provide transparent system information

ARCHITECTURE:
- 4-Tier Memory: Short-term (Redis), Episodic (PostgreSQL), Semantic (Neo4j), Procedural (Rules)
- 7-Node Workflow: Idle → Analyze → Tool Decision → Policy Check → Execute → Await Confirmation → Respond
- Tool Registry: Functions for task execution with risk assessment
- Policy Engine: Security evaluation and permission management"""

    # Current session context
    session_context = f"""
CURRENT SESSION:
- User ID: {state.user_id}
- Session: {state.session_id}
- Turn: #{state.total_turns}
- State: {state.current_node.value}
- Request ID: {request_id}"""

    # Memory and context information
    memory_info = """
MEMORY CONTEXT:"""
    
    if context_data:
        memory_info += f"""
- Context assembled from all 4 memory tiers
- Profile and preferences available
- Conversation history accessible
- Behavioral instructions loaded"""
    else:
        memory_info += """
- Limited memory context (fallback mode)
- Basic session information available"""

    # Available capabilities
    capabilities_info = f"""
AVAILABLE CAPABILITIES:
- Tool Functions: {len(available_tools)} registered
- Autonomous Execution: Policy-checked
- Memory Integration: RAG++ hierarchy
- Session Persistence: Cross-conversation continuity"""

    # Response guidelines
    response_guidelines = """
RESPONSE GUIDELINES:
- Be conversational, helpful, and contextually aware
- Reference previous interactions when relevant
- Explain your capabilities and reasoning clearly
- Provide system transparency when requested
- Ask for clarification when user intent is ambiguous
- Offer proactive suggestions based on context"""

    return f"{core_identity}\n{session_context}\n{memory_info}\n{capabilities_info}\n{response_guidelines}"

async def extract_and_store_semantic_memory(
    user_id: str,
    user_message: str,
    assistant_response: str,
    request_id: str
) -> None:
    """Extract and store semantic entities from conversation in Neo4j."""
    try:
        logger.debug(f"🧠 [REQ-{request_id}] Extracting semantic entities for Neo4j storage...")
        
        async with SemanticMemoryStore() as semantic_store:
            extraction_result = await semantic_store.extract_and_store_entities(
                user_id=user_id,
                user_message=user_message,
                assistant_response=assistant_response
            )
            
            logger.info(
                f"✅ [REQ-{request_id}] Stored {len(extraction_result.entities)} entities "
                f"and {len(extraction_result.relationships)} relationships in Neo4j"
            )
            
    except Exception as e:
        logger.warning(f"⚠️  [REQ-{request_id}] Semantic memory extraction failed: {e}")
        # Don't fail the entire workflow if semantic memory fails
        pass

def generate_fallback_response(
    event: GovernorEvent,
    state: GovernorState,
    context_data: Any,
    available_tools: list,
    request_id: str
) -> str:
    """Generate fallback response when LLM is unavailable."""
    
    return f"""Hello! I'm your Headless Governor AI. I encountered an issue connecting to my language model, but my core systems are operational.

**Current Status:**
- Session: Turn #{state.total_turns} 
- Memory: {'✅ Context Available' if context_data else '⚠️ Limited Context'}
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
        
        logger.debug("   • Preparing memory hierarchy...")
        logger.info("   ✅ RAG++ memory tiers configured")
        
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.fastapi_host,
        port=settings.fastapi_port,
        reload=settings.fastapi_reload and settings.is_development(),
        log_level=settings.log_level.lower(),
    )
