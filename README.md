# Conditional RV-GOMEA

## Building

Run `make` in this directory to build the project. The Armadillo library needs to be installed on the system.

## Running

To start a single run of the algorithm, run the command below. This will give you an overview of options with which you can control the algorithm.

```shell
python rvgomea/cmd/run.py --help
```

## Testing

To run a quick test of different linkage models, try the following command:

```shell
make && run_scripts/test_all_models
```

CWI m3
```shell
screen -S runner
(run_scripts/run_scalability_bisection.sh sphere 10,20,40,80 && run_scripts/run_scalability_bisection.sh rosenbrock 10,20,40,80 && run_scripts/run_scalability_bisection.sh osoreb  10,20,40,80 && run_scripts/run_set_cover_scalability_bisection.sh) 2>&1 | tee log.txt


(run_scripts/run_reb_chain_bisection.sh lt-fb-online-pruned,mp-hg-fb_no_order-without_clique_seeding-conditional,mp-hg-fb_no_order-with_clique_seeding-conditional,full lt-vs-mcond && run_scripts/run_set_cover_scalability_bisection.sh && run_scripts/run_scalability_bisection.sh sphere 10,20,40,80 && run_scripts/run_scalability_bisection.sh rosenbrock 10,20,40,80 && run_scripts/run_scalability_bisection.sh osoreb  10,20,40,80 && run_scripts/run_scalability_bisection.sh reb-grid 16,36,64,81 && run_scripts/run_scalability_bisection.sh reb2-chain-weak 10,20,40,80 && run_scripts/run_scalability_bisection.sh reb2-chain-strong 10,20,40,80 && run_scripts/run_scalability_bisection.sh reb2-chain-alternating 10,20,40,80 && run_scripts/run_scalability_bisection.sh reb5-no-overlap 10,20,40,80 && run_scripts/run_scalability_bisection.sh reb5-small-overlap 9,21,41,81 && run_scripts/run_scalability_bisection.sh reb5-small-overlap-alternating 9,21,41,81 && run_scripts/run_scalability_bisection.sh reb5-large-overlap 10,20,40,80 && run_scripts/run_scalability_bisection.sh reb5-disjoint-pairs 9,18,36,72) 2>&1 | tee log.txt

Ctrl+A + d  --> to detach
```

Snellius
```shell
bash deploy/scalability.slurm reb-grid 16,36,64,81 && bash deploy/scalability.slurm reb2-chain-weak 10,20,40,80 && bash deploy/scalability.slurm reb2-chain-strong 10,20,40,80 && bash deploy/scalability.slurm reb2-chain-alternating 10,20,40,80 && bash deploy/scalability.slurm reb5-no-overlap 10,20,40,80 && bash deploy/scalability.slurm reb5-small-overlap 9,21,41,81 && bash deploy/scalability.slurm reb5-small-overlap-alternating 9,21,41,81 && bash deploy/scalability.slurm reb5-large-overlap 10,20,40,80 && bash deploy/scalability.slurm reb5-disjoint-pairs 9,18,36,72

bash deploy_shark/scalability.slurm reb-grid 16,36,64,81 && bash deploy_shark/scalability.slurm reb2-chain-weak 10,20,40,80 && bash deploy_shark/scalability.slurm reb2-chain-strong 10,20,40,80 && bash deploy_shark/scalability.slurm reb2-chain-alternating 10,20,40,80 && bash deploy_shark/scalability.slurm reb5-no-overlap 10,20,40,80 && bash deploy_shark/scalability.slurm reb5-small-overlap 9,21,41,81 && bash deploy_shark/scalability.slurm reb5-small-overlap-alternating 9,21,41,81 && bash deploy_shark/scalability.slurm reb5-large-overlap 10,20,40,80 && bash deploy_shark/scalability.slurm reb5-disjoint-pairs 9,18,36,72 && bash deploy_shark/scalability.slurm sphere 10,20,40,80 && bash deploy_shark/scalability.slurm rosenbrock 10,20,40,80 && bash deploy_shark/scalability.slurm osoreb  10,20,40,80


bash deploy/scalability.slurm reb5-disjoint-pairs 9 && bash deploy/scalability.slurm reb5-disjoint-pairs 18 && bash deploy/scalability.slurm reb5-disjoint-pairs 36 && bash deploy/scalability.slurm reb5-disjoint-pairs 72 && bash deploy/scalability.slurm sphere 10 && bash deploy/scalability.slurm sphere 20 && bash deploy/scalability.slurm sphere 40 && bash deploy/scalability.slurm sphere 80 && bash deploy/scalability.slurm rosenbrock 10 && bash deploy/scalability.slurm rosenbrock 20 && bash deploy/scalability.slurm rosenbrock 40 && bash deploy/scalability.slurm rosenbrock 80 && bash deploy/scalability.slurm osoreb  10 && bash deploy/scalability.slurm osoreb  20 && bash deploy/scalability.slurm osoreb  40 && bash deploy/scalability.slurm osoreb  80 && bash deploy/scalability.slurm reb-grid 16 && bash deploy/scalability.slurm reb-grid 36 && bash deploy/scalability.slurm reb-grid 64 && bash deploy/scalability.slurm reb-grid 81 && bash deploy/scalability.slurm reb2-chain-weak 10 && bash deploy/scalability.slurm reb2-chain-weak 20 && bash deploy/scalability.slurm reb2-chain-weak 40 && bash deploy/scalability.slurm reb2-chain-weak 80 && bash deploy/scalability.slurm reb2-chain-strong 10 && bash deploy/scalability.slurm reb2-chain-strong 20 && bash deploy/scalability.slurm reb2-chain-strong 40 && bash deploy/scalability.slurm reb2-chain-strong 80 && bash deploy/scalability.slurm reb2-chain-alternating 10 && bash deploy/scalability.slurm reb2-chain-alternating 20 && bash deploy/scalability.slurm reb2-chain-alternating 40 && bash deploy/scalability.slurm reb2-chain-alternating 80 && bash deploy/scalability.slurm reb5-no-overlap 10 && bash deploy/scalability.slurm reb5-no-overlap 20 && bash deploy/scalability.slurm reb5-no-overlap 40 && bash deploy/scalability.slurm reb5-no-overlap 80 && bash deploy/scalability.slurm reb5-small-overlap 9 && bash deploy/scalability.slurm reb5-small-overlap 21 && bash deploy/scalability.slurm reb5-small-overlap 41 && bash deploy/scalability.slurm reb5-small-overlap 81 && bash deploy/scalability.slurm reb5-small-overlap-alternating 9 && bash deploy/scalability.slurm reb5-small-overlap-alternating 21 && bash deploy/scalability.slurm reb5-small-overlap-alternating 41 && bash deploy/scalability.slurm reb5-small-overlap-alternating 81 && bash deploy/scalability.slurm reb5-large-overlap 10 &&bash deploy/scalability.slurm reb5-large-overlap 20 &&bash deploy/scalability.slurm reb5-large-overlap 40 &&bash deploy/scalability.slurm reb5-large-overlap 80
And:
sbatch deploy/lt-vs-mcond.slurm

And:
sbatch deploy/set-cover-scalability.slurm

```

```
o = nothing
- = started
x = finished

# On Shark
--> 1-500
x   sbatch --array=1-100%40 deploy_shark/parallel_scalability.slurm
x   sbatch --array=101-120%40 deploy_shark/parallel_scalability.slurm
x   bash deploy_shark/parallel_scalability.slurm 120 1-80
x   bash deploy_shark/parallel_scalability.slurm 200 1-100
-   bash deploy_shark/parallel_scalability.slurm 300 1-100
17301264
x   bash deploy_shark/parallel_scalability.slurm 400 1-100
17301265

-   bash deploy_shark/parallel_scalability.slurm 920 1-80
17326340

--> 1241-1600:
-   bash deploy_shark/parallel_scalability.slurm 1240 1-60
17315645
-   bash deploy_shark/parallel_scalability.slurm 1300 1-100
17315646
-   bash deploy_shark/parallel_scalability.slurm 1400 1-100
17315661
-   bash deploy_shark/parallel_scalability.slurm 1500 1-100
17315663

# Other Snellius
--> 501-920
-   sbatch --array=501-920%40 deploy/parallel_scalability_thin.slurm
4808651

--> 1601-1920
-   sbatch --array=1601-1920%50 deploy/parallel_scalability_thin.slurm
4828843

# My Snellius:
--> 1001-1240
-   sbatch --array=1001-1240%20 deploy/parallel_scalability.slurm



# 
find gecco-data/scalability-all-results/ -name 'statistics.dat' -delete

# Send aggregates to cluster
cd gecco-data && find . -type f -name 'aggregated_results.csv' | tar -cf - -T - | tar -xf - -C scalability-aggregated

```
