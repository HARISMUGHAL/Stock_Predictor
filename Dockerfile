# Use Python 3.12 slim
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y libgomp1 bash && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Make start.sh executable
RUN chmod +x start.sh

# Expose port
EXPOSE 8000

# Use start.sh as entrypoint
CMD ["./start.sh"]
