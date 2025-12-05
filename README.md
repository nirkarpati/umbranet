# Umbranet - The Headless Governor System

A production-ready Personal AI Assistant with persistent memory, autonomous tool execution, and multi-tier knowledge management. Built with FastAPI, React, and a sophisticated RAG++ memory architecture.

## 🎯 What is Umbranet?

Umbranet is a "Headless Governor" - an AI assistant that operates as an invisible OS layer rather than a traditional chatbot. It features infinite memory persistence across years, autonomous task execution with safety controls, and a React-based dashboard for memory exploration.

## 🏗️ Architecture

### Core Components

- **Governor Engine** (FastAPI) - LangGraph state machine with 7-node workflow
- **RAG++ Memory System** - Four-tier persistent memory hierarchy  
- **Action Plane** - Tool registry with risk-based policy engine
- **Memory Reflector** - Background processing service with RabbitMQ
- **React Frontend** - Chat interface with interactive memory dashboard

### RAG++ Memory Tiers

| Tier | Technology | Purpose | Data |
|------|------------|---------|------|
| **1** | Redis | Short-term working memory | Active conversation context |
| **2** | PostgreSQL + pgvector | Episodic memory | Searchable conversation history |  
| **3** | Neo4j | Semantic memory | Entity knowledge graph |
| **4** | PostgreSQL | Procedural memory | User preferences & rules |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Poetry
- Docker & Docker Compose  
- Node.js 16+ (for frontend)

### Development Setup

```bash
# Clone and setup
git clone <repo-url>
cd umbranet

# Install Python dependencies
poetry install
poetry shell

# Add OpenAI API key (required)
export OPENAI_API_KEY="your-key-here"

# Start all services
docker-compose up --build

# The system will be available at:
# - Frontend: http://localhost:3000 
# - Backend API: http://localhost:8000
# - Neo4j Browser: http://localhost:7474
# - RabbitMQ Management: http://localhost:15672
```

## 💬 Usage

1. **Chat Interface**: Open http://localhost:3000
2. **Enter User ID**: Choose any identifier (e.g., "alice", "bob") 
3. **Start Chatting**: Messages are processed through the Governor workflow
4. **Explore Memory**: Click the brain icon to view the memory dashboard

## 🧠 Memory Dashboard Features

- **Knowledge Graph Visualization** - Interactive Neo4j entity relationships
- **Conversation History** - Searchable episodic memory
- **User Rules** - Learned preferences and procedural memory
- **Session Data** - Real-time short-term memory
- **Memory Deletion** - Complete user data removal across all tiers

## 🛠️ Development

### Backend (Python)

```bash
# Code quality
poetry run ruff check .       # Lint
poetry run ruff format .      # Format  
poetry run mypy src/          # Type check

# Testing
poetry run pytest tests/unit
poetry run pytest tests/integration
poetry run pytest tests/e2e

# Run standalone (without Docker)
poetry run uvicorn src.main:app --reload
```

### Frontend (React/TypeScript)

```bash
cd frontend

# Development
npm start                     # Dev server
npm run build                 # Production build
npm test                      # Tests
```

### Memory Reflector (Background Service)

```bash
# Run reflector standalone
poetry run python src/reflector/main.py

# Monitor reflection queue
docker-compose logs memory-reflector
```

## 📡 API Endpoints

### Chat & Health
- `POST /api/chat` - Main conversation endpoint
- `GET /health` - System health with Governor diagnostics
- `GET /` - Basic system info

### Memory Management
- `GET /api/memory/semantic/{user_id}` - Knowledge graph entities & relationships
- `GET /api/memory/episodic/{user_id}` - Conversation episodes  
- `GET /api/memory/procedural/{user_id}` - User rules & preferences
- `GET /api/memory/redis/{user_id}` - Short-term session data
- `DELETE /api/memory/user/{user_id}` - Complete memory deletion

## 🔧 Implemented Features

### ✅ Governor State Machine
- **7-node LangGraph workflow**: idle → analyze → tool_decision → policy_check → execute → await_confirmation → respond
- **State persistence** across conversation turns
- **Error handling** with fallback responses

### ✅ RAG++ Memory Integration
- **Multi-tier storage** with automatic routing
- **Entity-aware extraction** with User-Property model
- **Background reflection** processing via RabbitMQ
- **Vector similarity search** in episodic memory

### ✅ Tool Registry & Policy Engine
- **Risk-based categorization**: Safe, Sensitive, Dangerous
- **Auto-execution** for safe tools, confirmation for risky ones
- **Built-in tools**: Weather, file operations, communication, data lookup
- **Custom tool decorator** with schema validation

### ✅ Enhanced Context Assembly
- **Privacy-safe prompts** (no internal user_id leaks)
- **Dynamic context construction** from all memory tiers
- **Token-aware truncation** and prioritization
- **Fallback context** when memory systems unavailable

### ✅ Identity-Aware Entity Extraction
- **User-Property model** prevents graph fragmentation
- **Consistent user entity handling** across semantic processor
- **Deterministic entity ID generation**

### ✅ Production Frontend
- **Real-time chat** with WebSocket-like experience
- **Interactive knowledge graph** with hash-based positioning (no randomness)
- **Memory tier visualization** with real API data only
- **Multi-user support** with isolated memory spaces

## 🔒 Security & Privacy

- **Tenant isolation** - Complete data separation between users
- **Risk assessment** - Three-level tool permission system
- **Privacy protection** - Internal IDs never exposed to LLM
- **GDPR compliance** - Complete user data deletion capability
- **Secure defaults** - All tools require explicit registration

## 📊 System Requirements

### Development
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB for Docker images and data
- **Ports**: 3000, 5432, 6379, 7474, 7687, 8000, 15672

### Production (Estimated)
- **Response time**: < 500ms for cached context
- **Concurrent users**: 100+ with current setup
- **Memory growth**: ~10MB per user per month of active usage
- **Database scaling**: Horizontal scaling supported for all tiers

## 🧪 Testing

```bash
# Run all test suites
./scripts/ci-test.sh

# Individual test categories  
poetry run pytest tests/unit          # Fast unit tests
poetry run pytest tests/integration   # Database integration tests
poetry run pytest tests/e2e           # End-to-end workflow tests
```

## 📁 Project Structure

```
umbranet/
├── src/
│   ├── main.py                    # FastAPI application entry
│   ├── core/                      # Domain models & configuration
│   │   ├── config.py              # Pydantic settings
│   │   ├── domain/                # Event & state models
│   │   └── workflow/              # LangGraph state machine
│   ├── memory/                    # RAG++ memory system
│   │   ├── manager.py             # Multi-tier coordinator
│   │   ├── tiers/                 # Individual memory implementations
│   │   ├── context/               # Enhanced context assembly
│   │   └── services/              # Entity extraction & curation
│   ├── action_plane/              # Tool execution system
│   │   ├── tool_registry/         # Tool registration & discovery
│   │   ├── policy_engine/         # Risk assessment & permissions
│   │   └── tools/                 # Built-in tool implementations
│   ├── reflector/                 # Background memory processing
│   │   ├── main.py                # RabbitMQ consumer service
│   │   └── processors/            # Memory reflection processors
│   └── interfaces/                # External communication
├── frontend/                      # React TypeScript UI
├── tests/                         # Comprehensive test suite
└── docker-compose.yml             # Full development environment
```

## 🚧 Current Limitations

- **Single-node deployment** (no clustering yet)
- **Memory reflector** runs asynchronously (some features may have delays)  
- **Tool execution** limited to built-in tools (plugin system planned)
- **Channel integration** currently supports HTTP only (webhook integrations planned)

## 🛣️ Roadmap

- **Multi-channel support** - WhatsApp, Telegram, Phone integrations
- **Advanced reasoning** - Chain-of-thought and multi-step planning
- **Plugin system** - Third-party tool integration
- **Clustering support** - Multi-node deployment with shared state
- **Advanced analytics** - Memory growth patterns and usage insights

---

**Note**: This is a real production system with no mock data. All integrations require actual API keys and database connections.