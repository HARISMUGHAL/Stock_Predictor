#!/bin/bash

# If Railway sends literal "$PORT", fix it
if [[ "$PORT" == "\$PORT" ]] || [[ "$PORT" == '$PORT' ]]; then
  export PORT=8000
fi

echo "Starting server on port: $PORT"

exec uvicorn main:app --host 0.0.0.0 --port $PORT
