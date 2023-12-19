import argparse

from rvgomea.defaults import DEFAULT_LINKAGE_MODEL, DEFAULT_PROBLEM, DEFAULT_DIMENSIONALITY
from rvgomea.experiments.pop_sweep_runner import run_pop_sweep
from rvgomea.run_config import RunConfig


def main():
    parser = argparse.ArgumentParser(
        prog='Single pop sweep')

    parser.add_argument('-l', '--linkage-model', type=str, default=DEFAULT_LINKAGE_MODEL)
    parser.add_argument('-p', '--problem', type=str, default=DEFAULT_PROBLEM)
    parser.add_argument('-d', '--dimensionality', type=int, default=DEFAULT_DIMENSIONALITY)

    args = parser.parse_args()

    base_run_config = RunConfig(
        linkage_model=args.linkage_model,
        population_size=-1,
        random_seed=-1,
        problem=args.problem,
        dimensionality=args.dimensionality,
        lower_init_bound=-115,
        upper_init_bound=-110,
    )

    run_pop_sweep(
        "test_pop_sweep", base_run_config, 30, num_cpus=5,
        log_progress=True
    )


if __name__ == '__main__':
    main()
