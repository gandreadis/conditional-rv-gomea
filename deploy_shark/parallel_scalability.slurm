#!/bin/bash
sbatch <<EOT
#!/bin/bash
#
#SBATCH --partition=all
#SBATCH --array=${2}%40
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=80GB
#SBATCH --time=21-00:00:00
#SBATCH -o "logs/scalability-array-%A_${1}_"%a".out"

module purge
module load library/blas/0.3.13/gcc-8.3.1
module load library/lapack/3.9.0/gcc-8.3.1

module load system/python/3.8.1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gandreadis/arma/lib64

run_scripts/run_parallel_scalability_bisection_with_offset.sh "\${SLURM_ARRAY_TASK_ID}" "${1}"

EOT
