#!/bin/bash
sbatch <<EOT
#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00
#SBATCH -o "logs/conditional-%j.out"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load Armadillo/12.6.2-foss-2023a
module load FlexiBLAS/3.3.1-GCC-12.3.0

#$1 = ...
#uni-hg-gbo-without_clique_seeding-conditional,mp-hg-gbo-with_clique_seeding-conditional,uni-hg-fb_no_order-without_clique_seeding-conditional,mp-hg-fb_no_order-with_clique_seeding-conditional

make
run_scripts/run_reb_chain_bisection.sh "${1}" conditional

EOT
