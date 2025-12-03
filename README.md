# Headless Governor System

A revolutionary Personal AI Assistant that operates as an invisible operating system layer.

## Overview

The Headless Governor System is designed as a "Headless OS" that eliminates the traditional chatbot interface model, instead operating through existing communication channels (WhatsApp, Telegram, Phone calls) with persistent memory and autonomous task execution capabilities.

## Architecture

- **Control Plane**: Governor engine with state machine and context management
- **Action Plane**: Tool registry with permission-based execution
- **Memory Hierarchy (RAG++)**: Four-tier memory system
  - Short-term memory (Redis)
  - Episodic memory (Vector DB)
  - Semantic memory (Knowledge Graph)
  - Procedural memory (Rules/Preferences)

## Development Setup

### Prerequisites

- Python 3.11+
- Poetry
- Docker & Docker Compose

### Installation

```bash
# Install dependencies
poetry install

# Start development environment
docker-compose up

# Run the application
poetry run uvicorn src.main:app --reload
```

### Development Commands

```bash
# Code quality
poetry run ruff check .       # Lint
poetry run ruff format .      # Format
poetry run mypy src/          # Type check

# Testing
poetry run pytest tests/unit
poetry run pytest tests/integration
```

## Project Status

Currently in Phase 0: Infrastructure & Environment Setup

See `docs/Development plan.md` for detailed implementation roadmap.