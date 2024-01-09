#!/bin/bash
sbatch <<EOT
#!/bin/bash
#
#SBATCH --partition=all,RTAIgpu
#SBATCH --array=1-100%40
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48GB
#SBATCH --time=21-00:00:00
#SBATCH -o "logs/scalability-array-%A_${1}_"%a".out"

module purge
module load library/blas/0.3.13/gcc-8.3.1
module load library/lapack/3.9.0/gcc-8.3.1

module load system/python/3.8.1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gandreadis/arma/lib64

OFFSET=$1
TASK_INDEX=$((SLURM_ARRAY_TASK_ID+OFFSET))

run_scripts/run_parallel_scalability_bisection.sh ${TASK_INDEX}

EOT
