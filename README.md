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

CWI
```shell
screen -S runner
(run_scripts/run_scalability_bisection.sh sphere  10,20,40,80 && run_scripts/run_scalability_bisection.sh rosenbrock  10,20,40,80 && run_scripts/run_scalability_bisection.sh summation-cancellation  10,20,40,80) 2>&1 | tee log.txt

Ctrl+A + d    --> to detach
```

Snellius
```shell
bash deploy/scalability.slurm reb2-chain-weak  10,20,40,80 && bash deploy/scalability.slurm reb2-chain-strong  10,20,40,80 && bash deploy/scalability.slurm reb2-chain-alternating  10,20,40,80 && bash deploy/scalability.slurm reb5-no-overlap  10,20,40,80 && bash deploy/scalability.slurm reb5-small-overlap  9,21,41,81 && bash deploy/scalability.slurm reb5-small-overlap-alternating  9,21,41,81 && bash deploy/scalability.slurm reb5-large-overlap  10,20,40,80 && bash deploy/scalability.slurm reb5-disjoint-pairs  9,18,36,72 && bash deploy/scalability.slurm reb-grid  16,36,64,81
```
