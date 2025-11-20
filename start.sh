#!/bin/sh

# Railway automatically sets PORT env variable
PORT=${PORT:-8000}  # fallback for local testing

echo "Starting FastAPI server on port $PORT..."
exec uvicorn main:app --host 0.0.0.0 --port "$PORT"
