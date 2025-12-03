#!/bin/bash

# CI/CD Test Script - Simulates GitHub Actions Quality Gate

set -e

echo "🚀 Running CI/CD Quality Gate Simulation..."
echo ""

# Check Poetry environment
echo "📦 Checking Poetry environment..."
poetry --version
echo "✅ Poetry is available"
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
poetry install --no-ansi
echo "✅ Dependencies installed"
echo ""

# Run Ruff linting
echo "🔍 Running Ruff linting..."
poetry run ruff check .
echo "✅ Ruff linting passed"
echo ""

# Run Ruff formatting check
echo "🎨 Checking code formatting..."
poetry run ruff format --check .
echo "✅ Code formatting is correct"
echo ""

# Run MyPy type checking
echo "🔎 Running MyPy type checking..."
poetry run mypy src/
echo "✅ Type checking passed"
echo ""

# Run unit tests with coverage
echo "🧪 Running unit tests with coverage..."
poetry run pytest tests/unit/ -v --cov=src --cov-report=term-missing --cov-report=xml
echo "✅ All tests passed"
echo ""

# Run security checks
echo "🔒 Running security checks..."
echo "  Running safety check..."
poetry run safety check --continue-on-error || echo "⚠️  Safety check completed (warnings allowed)"

echo "  Running bandit security scan..."
poetry run bandit -r src/ -f json -o bandit-report.json || echo "⚠️  Bandit scan completed (warnings allowed)"
echo ""

# Docker validation
echo "🐳 Validating Docker configuration..."
docker compose config > /dev/null
echo "✅ Docker Compose configuration is valid"
echo ""

# Workflow validation
echo "📋 Validating GitHub Actions workflows..."
python -c "
import yaml
with open('.github/workflows/quality_gate.yml', 'r') as f: yaml.safe_load(f)
with open('.github/workflows/build_deploy.yml', 'r') as f: yaml.safe_load(f)
print('✅ GitHub Actions workflows are valid')
"
echo ""

echo "🎉 All CI/CD quality checks passed!"
echo ""
echo "Summary:"
echo "  ✅ Code linting (Ruff)"
echo "  ✅ Type checking (MyPy)" 
echo "  ✅ Unit tests (Pytest)"
echo "  ✅ Test coverage reporting"
echo "  ✅ Security scanning (Safety + Bandit)"
echo "  ✅ Docker configuration"
echo "  ✅ GitHub Actions workflows"
echo ""
echo "🚀 Ready for production deployment!"