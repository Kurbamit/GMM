#!/bin/sh
#SBATCH -p gpu
#SBATCH -n1
#SBATCH --gres gpu
#SBATCH --time=12:00:00

echo "Starting training..."
. gpu_env/bin/activate
python3 test.py
