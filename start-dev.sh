#!/bin/bash

# Headless Governor Development Environment Start Script

set -e

echo "🚀 Starting Headless Governor Development Environment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from .env.example..."
    cp .env.example .env
fi

# Build and start services
echo "🔧 Building and starting services..."
docker compose up --build -d

echo "⏳ Waiting for services to be ready..."
sleep 30

# Check health of services
echo "🩺 Checking service health..."

# Check Redis
echo -n "  Redis: "
if docker compose exec -T redis redis-cli ping | grep -q PONG; then
    echo "✅ Ready"
else
    echo "❌ Not ready"
fi

# Check PostgreSQL
echo -n "  PostgreSQL: "
if docker compose exec -T vector_db pg_isready -U governor -d governor_memory > /dev/null 2>&1; then
    echo "✅ Ready"
else
    echo "❌ Not ready"
fi

# Check Neo4j (may take longer to start)
echo -n "  Neo4j: "
if docker compose exec -T graph_db cypher-shell -u neo4j -p dev_password "RETURN 1" > /dev/null 2>&1; then
    echo "✅ Ready"
else
    echo "⚠️  Still starting (this is normal)"
fi

# Check FastAPI
echo -n "  Governor Engine: "
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Ready"
else
    echo "⚠️  Still starting (this is normal)"
fi

echo ""
echo "🎉 Development environment is starting up!"
echo ""
echo "Available services:"
echo "  🌐 FastAPI App:     http://localhost:8000"
echo "  🌐 FastAPI Docs:    http://localhost:8000/docs"
echo "  📊 Neo4j Browser:   http://localhost:7474"
echo "  🔴 Redis:           localhost:6379"
echo "  🐘 PostgreSQL:      localhost:5432"
echo ""
echo "To view logs: docker compose logs -f"
echo "To stop:      docker compose down"
echo "To restart:   docker compose restart"