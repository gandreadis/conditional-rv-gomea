# Fitness-based Conditional RV-GOMEA

Code associated with the pre-print titled: "Fitness-based Linkage Learning and Maximum-Clique Conditional Linkage Modelling for Gray-box Optimization with RV-GOMEA", by [Georgios Andreadis](https://github.com/gandreadis/), Tanja Alderliesten, and Peter A.N. Bosman. This codebase includes contributions by [Anton Bouter](https://github.com/abouter/) and [Chantal Olieman](https://github.com/chantal-olieman/).

## Building

Run `make` in this directory to build the project. The Armadillo library needs to be installed on the system. Depending on your installation, please modify the `Makefile` to link to it, correctly.

## Running

To start a single run of the algorithm, run the command below. This will give you an overview of options with which you can control the algorithm.

```shell
python rvgomea/cmd/run.py --help
```

To reproduce the experiments in the pre-print, you can use the scripts in the `run_scripts` folder. SLURM configurations for each are available in the `deploy` and `deploy_shark` folders. Due to the volume of runs needed for the bisection scalability experiments, there are also job array scripts available. These divide the bulk of runs needed into one job per run.

An example run, launching the first 500 runs with at most 40 in parallel, would look as follows:

```shell
sbatch --array=1-500%40 deploy/parallel_scalability_thin.slurm
```

At times, when runs are stopped pre-maturely, it can help to clean-up left-over convergence files:

```shell
find data/scalability-all-results/ -name 'statistics.dat' -delete
```

To send only aggregate results (excluding individual results) to a different location for the extrapolation runs, use the following command:

```shell
cd gecco-data && find . -type f -name 'aggregated_results.csv' | tar -cf - -T - | tar -xf - -C scalability-aggregated
```

## Testing

To run a quick test of different linkage models, try the following command:

```shell
make && run_scripts/test_all_models
```

To test the VkD-CMA implementation on the different problems, run:

```shell
run_scripts/test_all_models_with_vkd_cma
```

## License

Fitness-based Conditional Real-Valued Gene-pool Optimal Mixing Evolutionary Algorithm © 2024 by Georgios Andreadis, Tanja Alderliesten, Peter A.N. Bosman, Anton Bouter, and Chantal Olieman is licensed under CC BY-NC-ND 4.0. A copy of the license is included in the `LICENSE` file.
