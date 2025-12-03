-- Database initialization script for Umbranet Governor
-- This sets up basic schemas and pgvector extension

-- Install pgvector extension (if available)
CREATE EXTENSION IF NOT EXISTS vector;

-- Set some optimal PostgreSQL settings
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';

-- Create database schema placeholder (tables will be created by the application)
CREATE SCHEMA IF NOT EXISTS memory;

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA memory TO governor;
GRANT ALL PRIVILEGES ON SCHEMA public TO governor;