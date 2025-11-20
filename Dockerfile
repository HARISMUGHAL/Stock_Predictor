# Use Python 3.12 slim image
FROM python:3.12-slim

# Install system dependencies (libgomp1 for LightGBM)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose the port (Railway sets it automatically)
EXPOSE 8000

# Use start.sh to handle dynamic PORT
CMD ["./start.sh"]
