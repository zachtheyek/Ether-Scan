# SETI ML Pipeline Docker Image
# For GCP VM deployment with GPU support

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    cmake \
    libhdf5-dev \
    libopenblas-dev \
    gfortran \
    pkg-config \
    software-properties-common \
    curl \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# Copy environment file
COPY environment.yml /tmp/environment.yml

# Create conda environment
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Activate environment by default
SHELL ["conda", "run", "-n", "seti-ml", "/bin/bash", "-c"]

# Install additional Python packages
RUN pip install --upgrade pip && \
    pip install \
    tensorflow==2.13.* \
    tensorflow-gpu==2.13.* \
    setigen \
    blimpy \
    tqdm \
    psutil \
    tensorboard

# Create working directories
RUN mkdir -p /app /data/seti /models/seti /output/seti /data/seti/training /data/seti/testing

# Set working directory
WORKDIR /app

# Copy application code
COPY . /app/

# Create models directory structure
RUN mkdir -p /app/models

# Set permissions
RUN chmod -R 755 /app && \
    chmod -R 777 /data/seti /models/seti /output/seti

# Environment variables for paths
ENV SETI_DATA_PATH=/data/seti
ENV SETI_MODEL_PATH=/models/seti
ENV SETI_OUTPUT_PATH=/output/seti
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose ports for TensorBoard and Jupyter (optional)
EXPOSE 6006 8888

# Create entrypoint script
RUN echo '#!/bin/bash\n\
    source /opt/conda/etc/profile.d/conda.sh\n\
    conda activate seti-ml\n\
    exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "main.py", "--help"]
