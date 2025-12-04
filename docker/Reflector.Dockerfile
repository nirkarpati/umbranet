FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY docker-minimal-requirements.txt .
RUN pip install --no-cache-dir -r docker-minimal-requirements.txt

# Copy source code
COPY src/ src/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import asyncio; from src.reflector.main import get_reflector_service; asyncio.run(get_reflector_service().health_check())" || exit 1

# Run the reflector service
CMD ["python", "-m", "src.reflector.main"]