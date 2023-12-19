import argparse

from rvgomea.defaults import *
from rvgomea.run_config import RunConfig
from rvgomea.run_algorithm import run_algorithm, LINKAGE_MODEL_CODES, PROBLEM_CODES


def main():
    parser = argparse.ArgumentParser(
        prog='Conditional RV-GOMEA',
        description='Conditional version of the RV-GOMEA algorithm')

    parser.add_argument('-i', '--in-directory', type=str)
    parser.add_argument('-o', '--show-output', action="store_true")
    parser.add_argument('-z', '--save-fitness-dependencies', action="store_true")

    parser.add_argument('-l', '--linkage-model', type=str.lower,
                        default=DEFAULT_LINKAGE_MODEL, choices=LINKAGE_MODEL_CODES.keys())
    parser.add_argument('-s', '--population-size', type=int,
                        default=DEFAULT_POPULATION_SIZE)
    parser.add_argument('--num-populations', type=int,
                        default=DEFAULT_NUM_POPULATIONS)
    parser.add_argument('--tau', type=float,
                        default=DEFAULT_TAU)
    parser.add_argument('--distribution-multiplier-decrease', type=float,
                        default=DEFAULT_DISTRIBUTION_MULTIPLIER_DECREASE)
    parser.add_argument('--st-dev-threshold', type=float,
                        default=DEFAULT_ST_DEV_THRESHOLD)
    parser.add_argument('-r', '--random-seed', type=int,
                        default=DEFAULT_RANDOM_SEED)

    parser.add_argument('-p', '--problem', type=str.lower,
                        default=DEFAULT_PROBLEM, choices=PROBLEM_CODES.keys())
    parser.add_argument('-d', '--dimensionality', type=int,
                        default=DEFAULT_DIMENSIONALITY)
    parser.add_argument('-b', '--black-box', action="store_true",
                        default=DEFAULT_BLACK_BOX)
    parser.add_argument('-x', '--lower-init-bound', type=float,
                        default=DEFAULT_LOWER_INIT_BOUND)
    parser.add_argument('-y', '--upper-init-bound', type=float,
                        default=DEFAULT_UPPER_INIT_BOUND)
    parser.add_argument('-v', '--vtr', type=float,
                        default=DEFAULT_VTR)
    parser.add_argument('-a', '--rotation-angle', type=float,
                        default=DEFAULT_ROTATION_ANGLE)

    parser.add_argument('-e', '--max-num-evaluations', type=float,
                        default=DEFAULT_MAX_NUM_EVALUATIONS)
    parser.add_argument('-t', '--max-num-seconds', type=float,
                        default=DEFAULT_MAX_NUM_SECONDS)
    parser.add_argument('--max-no-improvement-stretch', type=int,
                        default=DEFAULT_MAX_NO_IMPROVEMENT_STRETCH)
    parser.add_argument('-f', '--fitness-variance-tolerance', type=float,
                        default=DEFAULT_FITNESS_VARIANCE_TOLERANCE)

    args = parser.parse_args()
    config = RunConfig(
        linkage_model=args.linkage_model,
        population_size=args.population_size,
        num_populations=args.num_populations,
        tau=args.tau,
        distribution_multiplier_decrease=args.distribution_multiplier_decrease,
        st_dev_threshold=args.st_dev_threshold,
        random_seed=args.random_seed,
        problem=args.problem,
        dimensionality=args.dimensionality,
        black_box=args.black_box,
        lower_init_bound=args.lower_init_bound,
        upper_init_bound=args.upper_init_bound,
        vtr=args.vtr,
        rotation_angle=args.rotation_angle,
        max_num_evaluations=args.max_num_evaluations,
        max_num_seconds=args.max_num_seconds,
        max_no_improvement_stretch=args.max_no_improvement_stretch,
        fitness_variance_tolerance=args.fitness_variance_tolerance,
    )

    if args.show_output:
        print(config)
        print()

    succeeded = run_algorithm(config, in_dir=args.in_directory, show_output=args.show_output,
                              save_fitness_dependencies=args.save_fitness_dependencies).succeeded

    exit(0 if succeeded else 1)


if __name__ == '__main__':
    main()
