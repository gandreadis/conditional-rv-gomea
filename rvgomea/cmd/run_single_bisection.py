import argparse

from rvgomea.defaults import DEFAULT_LINKAGE_MODEL, DEFAULT_PROBLEM, DEFAULT_DIMENSIONALITY
from rvgomea.experiments.bisection_runner import run_bisection
from rvgomea.run_config import RunConfig


def main():
    parser = argparse.ArgumentParser(
        prog='Single bisection',
        description='Run a single bisection')

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

    result_population_size, result_median_num_evaluations, result_corrected_num_evaluations = run_bisection(
        "test_bisection", base_run_config, 4, num_cpus=4,
        log_progress=True
    )

    print(f"Pop. size:                  {result_population_size}")
    print(f"Median num. evaluations:    {result_median_num_evaluations}")
    print(f"Corrected num. evaluations: {result_corrected_num_evaluations}")


if __name__ == '__main__':
    main()
