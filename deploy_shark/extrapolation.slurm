#!/bin/bash
#
#SBATCH --partition=all,RTAIgpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32GB
#SBATCH --time=10-00:00:00
#SBATCH -o "extrapolation-logs/extrapolation-array-%A_%a.out"

module purge
module load library/blas/0.3.13/gcc-8.3.1
module load library/lapack/3.9.0/gcc-8.3.1

module load system/python/3.8.1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gandreadis/arma/lib64

run_scripts/run_extrapolated_scalability.sh ${SLURM_ARRAY_TASK_ID}
#--array=0-69
