#!/bin/bash
sbatch <<EOT
#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --time=120:00:00
#SBATCH -o "logs/scalability-${1}-%j.out"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load Armadillo/12.6.2-foss-2023a
module load FlexiBLAS/3.3.1-GCC-12.3.0

make
run_scripts/run_scalability_bisection.sh "${1}" "${2}"

EOT

# bash deploy/scalability.slurm 1 2
