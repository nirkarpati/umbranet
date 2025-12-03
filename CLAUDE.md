# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

- **No Mock Data**: Never implement mock, simulated, fake, or dummy data - this is a real production system requiring real integrations

**Umbranet** is the "Headless Governor System" - a revolutionary Personal AI Assistant that operates as an invisible operating system layer rather than a traditional chatbot application. The project is currently in the planning/design phase with comprehensive architectural documentation.

## Current Status

- **Phase**: Planning/Documentation (No implementation yet)
- **Ready for**: Phase 0 implementation (infrastructure setup)
- **Next Step**: Begin Python environment setup and basic FastAPI application structure

## Technology Stack (Planned)

- **Language**: Python 3.11+
- **Package Manager**: Poetry
- **Web Framework**: FastAPI
- **State Management**: LangGraph (for state machine)
- **Databases**: Redis (short-term), PostgreSQL with pgvector (episodic), Neo4j (semantic)
- **Linting**: ruff
- **Type Checking**: mypy
- **Testing**: pytest
- **Containerization**: Docker & Docker Compose

## Development Commands (When Implemented)

```bash
# Environment setup
poetry install                    # Install dependencies
poetry shell                     # Activate virtual environment

# Development
poetry run uvicorn src.main:app --reload  # Start development server
docker-compose up                 # Start full development environment with databases

# Code Quality
poetry run ruff check .           # Lint code
poetry run ruff format .          # Format code
poetry run mypy src/              # Type checking
poetry run pytest tests/unit      # Run unit tests
poetry run pytest tests/integration  # Run integration tests

# Docker
docker-compose up --build         # Build and start all services
docker-compose down               # Stop all services
```

## Architecture Overview

The Governor System follows a "Headless OS" model with three core pillars:

### 1. Interface Independence

- Channel-agnostic AI (WhatsApp, Telegram, Phone calls)
- Webhook-based input normalization
- No traditional UI - uses existing communication channels

### 2. State Permanence

- Infinite session with persistent memory across years
- Four-tier memory hierarchy (RAG++):
  - **Tier 1**: Short-term memory (Redis)
  - **Tier 2**: Episodic memory (Vector DB)
  - **Tier 3**: Semantic memory (Knowledge Graph)
  - **Tier 4**: Procedural memory (Rules/Preferences)

### 3. Regulated Autonomy

- Permission-based autonomous task execution
- Policy engine with risk assessment (Level 0-2)
- Background processes with user confirmation flows

## Project Structure (Planned)

```
src/
├── main.py                    # Application entry point
├── core/                      # Shared utilities and domain models
│   ├── config.py             # Pydantic settings
│   └── domain/               # Event object definitions
├── governor/                  # Core "Kernel" logic
│   ├── state_machine/        # LangGraph state definitions
│   └── context/              # Context manager
├── action_plane/             # Tool execution layer
│   ├── tool_registry/        # Atomic tool definitions
│   └── policy_engine/        # Security/permission logic
├── memory/                   # Database adapters
│   ├── redis_client.py
│   ├── vector_store.py
│   └── graph_store.py
└── interfaces/               # I/O routers
    └── webhooks/             # Twilio/Telegram handlers
```

NOTE **No Mock Data**: Never implement mock, simulated, fake, or dummy data - this is a real production system requiring real integrations
\*\*never mention claude in the commit msg

## Key Architectural Concepts

### The Governor (Control Plane)

- Stateful personalization layer between user and LLM
- Deterministic state machine using LangGraph
- Context manager for dynamic prompt construction
- Policy engine for permission/safety checks

### The Action Plane

- Tool registry with atomic function definitions
- Risk-based permission system (Safe/Sensitive/Dangerous)
- Deterministic execution with structured output validation

### O-E-R Learning Loop

- **Observe**: Real-time fact extraction during conversations
- **Extract**: Episodic logging to vector database
- **Reflect**: Nightly pattern analysis to update procedural memory

## Configuration Requirements

- **pyproject.toml**: Poetry dependencies including LangGraph, FastAPI, Redis
- **docker-compose.yml**: Multi-service setup with Redis, PostgreSQL+pgvector, Neo4j
- **Strict typing**: mypy in strict mode with Pydantic integration
- **Linting**: ruff with pycodestyle, pyflakes, isort, bugbear rules

## Key Documentation

- `docs/Architectural Blueprint_ The _Headless_ Governor System.md`: Complete technical specification
- `docs/Development plan.md`: Detailed implementation phases and requirements

## Development Principles

- **Security First**: Never expose or log secrets, implement permission-based execution
- **State Management**: Use deterministic state machines, avoid LLM-controlled flow
- **Memory Hierarchy**: Implement proper data separation between temporary and persistent storage
- **Modular Design**: Atomic tools, clean interfaces, dependency injection
- **Type Safety**: Strict typing with Pydantic models for all data structures
- **No Mock Data**: Never implement mock, simulated, fake, or dummy data - this is a real production system requiring real integrations
- **Never mention claude in the commit msg**
