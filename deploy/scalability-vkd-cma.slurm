#!/bin/bash
sbatch <<EOT
#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --time=120:00:00
#SBATCH -o "logs/scalability-vkd-cma-${1}-%j.out"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

run_scripts/run_scalability_bisection_vkd_cma.sh "${1}" "${2}"

EOT

# bash deploy/scalability-vkd-cma.slurm 1 2
