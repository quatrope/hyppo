#!/bin/bash
#SBATCH --job-name=hyppo-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=120G
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err

echo "=========================================="
echo "HYPPO Benchmark - SLURM Job"
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

# Run the benchmark
python scripts/run_local_benchmark.py

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
