#!/bin/bash

# Fast Training Script for Stuttering Detection
# This script ensures optimal GPU performance

echo "=================================================="
echo "STUTTERING DETECTION - FAST GPU TRAINING"
echo "=================================================="

# Check if running on Vast.ai or similar GPU instance
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu --format=csv,noheader,nounits
    echo ""
fi

# Set environment variables for optimal GPU performance
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=0  # Use first GPU

echo "Starting optimized training pipeline..."
echo ""

# Run the training with all optimizations
python3 main.py \
    --mode all \
    --fast \
    --use-wandb \
    --config config/config.yaml \
    2>&1 | tee training_log_$(date +%Y%m%d_%H%M%S).txt

echo ""
echo "Training completed!"
echo "Check the checkpoints/ directory for results."