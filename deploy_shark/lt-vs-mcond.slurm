#!/bin/bash
#
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=64GB
#SBATCH --time=21-00:00:00
#SBATCH -o "logs/lt-vs-mcond-%j.out"

module purge
module load library/blas/0.3.13/gcc-8.3.1
module load library/lapack/3.9.0/gcc-8.3.1

module load system/python/3.8.1

pip install --user numpy pandas

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gandreadis/arma/lib64

make

run_scripts/run_reb_chain_bisection.sh lt-fb-online-pruned,mp-hg-fb_no_order-without_clique_seeding-conditional,mp-hg-fb_no_order-with_clique_seeding-conditional,full lt-vs-mcond
