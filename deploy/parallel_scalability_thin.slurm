#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --time=120:00:00
#SBATCH -o "logs/scalability-array-%A_%a.out"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load Armadillo/12.6.2-foss-2023a
module load FlexiBLAS/3.3.1-GCC-12.3.0

run_scripts/run_parallel_scalability_bisection.sh ${SLURM_ARRAY_TASK_ID}

# Important: pass --array=1-1920%20
