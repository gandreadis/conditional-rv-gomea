#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=120:00:00
#SBATCH -o "logs/lt-vs-mcond-%j.out"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load Armadillo/12.6.2-foss-2023a
module load FlexiBLAS/3.3.1-GCC-12.3.0

make
run_scripts/run_reb_chain_bisection.sh lt-fb-online-pruned,mp-hg-fb_no_order-without_clique_seeding-conditional,mp-hg-fb_no_order-with_clique_seeding-conditional,full lt-vs-mcond
