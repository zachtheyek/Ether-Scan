#!/usr/bin/env bash

OUTPUT_FILE="results.txt"
: > "$OUTPUT_FILE"  # Truncate the file

run_and_log() {
    echo "### Command: $*" | tee -a "$OUTPUT_FILE"
    eval "$*" 2>&1 | tee -a "$OUTPUT_FILE"
    echo -e "\n" >> "$OUTPUT_FILE"
}

# -------------------------------
# Check GPU hardware
run_and_log "nvidia-smi"

# Get detailed GPU info
run_and_log "nvidia-smi -q -d MEMORY"
run_and_log "nvidia-smi -q -d UTILIZATION"
run_and_log "nvidia-smi -q -d ECC"
run_and_log "nvidia-smi -q -d TEMPERATURE"
run_and_log "nvidia-smi -q -d PERFORMANCE"
run_and_log "nvidia-smi -q -d CLOCKS"

# CUDA/cuDNN Version Information
run_and_log "nvcc --version"
run_and_log "cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2"
run_and_log "cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2 2>/dev/null || echo 'cuDNN headers not found in /usr/include'"
run_and_log "find /usr -name '*cudnn*' 2>/dev/null | head -10"

# TensorFlow/Python Environment
run_and_log "python -c \"import tensorflow as tf; print('TF version:', tf.__version__); print('GPUs available:', len(tf.config.list_physical_devices('GPU'))); print('Built with CUDA:', tf.test.is_built_with_cuda())\""
run_and_log "conda info"
run_and_log "conda list | grep -E '(tensorflow|cuda|cudnn)'"

# System Resources
run_and_log "free -h"
run_and_log "nproc"
run_and_log "cat /proc/cpuinfo | grep 'model name' | head -1"

# GCP Instance Information
run_and_log "curl -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/machine-type"
run_and_log "curl -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/zone"

echo "Results saved to: $OUTPUT_FILE"
