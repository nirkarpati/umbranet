# Lightweight Development Dockerfile for Umbranet Governor
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy minimal requirements first (for layer caching)
COPY docker-minimal-requirements.txt ./requirements.txt

# Install minimal Python dependencies
RUN pip install -r requirements.txt

# Copy source code
COPY src ./src

# Create non-root user
RUN groupadd --gid 1000 governor \
    && useradd --uid 1000 --gid governor --shell /bin/bash --create-home governor \
    && chown -R governor:governor /app

USER governor

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]