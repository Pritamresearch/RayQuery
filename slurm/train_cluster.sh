#!/bin/bash
#SBATCH --job-name=rayquery_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm_%j_rayquery.out
#SBATCH --error=logs/slurm_%j_rayquery.err

PROJECT_DIR=/mnt/scratch2/users/2641309/projects/rayquery
cd $PROJECT_DIR
mkdir -p logs results/checkpoints

# Activate Kelvin2 HPC Environment
flight env activate conda@opencood
conda activate coop_v2v
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

echo "====== Starting RayQuery Training at $(date) ======"
python src/train.py --epochs 60 --lr 2e-4 2>&1 | tee logs/train_${SLURM_JOB_ID}.log

echo "====== Completed at $(date) ======"
