#!/bin/bash
#SBATCH --job-name=hyppo-validation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=8G
#SBATCH --partition=short
#SBATCH --time=00:05:00
#SBATCH --output=validation_%j.out
#SBATCH --error=validation_%j.err

echo "=========================================="
echo "HYPPO Validation - Quick smoke test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Started: $(date)"
echo "=========================================="

# Activate conda/mamba environment (adjust path as needed)
source ~/micromamba/etc/profile.d/micromamba.sh
micromamba activate hyppo3.13

# Change to benchmark directory
cd ~/work/hyppo/benchmark

# Run the validation
python scripts/run_validation.py

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
