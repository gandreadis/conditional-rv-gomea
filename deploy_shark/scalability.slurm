#!/bin/bash
sbatch <<EOT
#!/bin/bash
#
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=64GB
#SBATCH --time=21-00:00:00
#SBATCH -o "logs/scalability-${1}-%j.out"

module purge
module load library/blas/0.3.13/gcc-8.3.1
module load library/lapack/3.9.0/gcc-8.3.1

module load system/python/3.8.1

pip install --user numpy pandas

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gandreadis/arma/lib64

make
run_scripts/run_scalability_bisection.sh "${1}" "${2}"

EOT

# bash deploy/scalability.slurm 1 2
