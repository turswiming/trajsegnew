#!/bin/bash
# Training script with proper environment setup

eval "$(conda shell.bash hook)"
conda activate traj

# Set library paths for CUDA
export LD_LIBRARY_PATH=/venv/traj/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH

# Run training
cd /workspace/trajectories_for_seg
python3 train.py --config src/configs/config.yaml "$@"

