# SETI ML Pipeline Docker Image
FROM tensorflow/tensorflow:2.13.0-gpu

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p /data/seti /models/seti /output/seti

# Set environment variables
ENV PYTHONPATH=/app
ENV SETI_DATA_PATH=/data/seti
ENV SETI_MODEL_PATH=/models/seti
ENV SETI_OUTPUT_PATH=/output/seti

# Default command
CMD ["python", "main.py", "--help"]
