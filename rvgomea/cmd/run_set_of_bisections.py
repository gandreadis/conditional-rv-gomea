import argparse
import os

import pandas as pd

from rvgomea.defaults import DEFAULT_LINKAGE_MODEL, DEFAULT_PROBLEM, DEFAULT_DIMENSIONALITY, \
    DEFAULT_NUM_REPEATS_PER_BISECTION_TEST, DEFAULT_MAX_NUM_EVALUATIONS
from rvgomea.experiments.bisection_runner import run_bisection
from rvgomea.run_config import RunConfig


def main():
    parser = argparse.ArgumentParser(
        prog='Set of bisections',
        description='Run a set of bisections')

    parser.add_argument('-o', '--output-dir', type=str, required=True)
    parser.add_argument('-p', '--problems', type=str,
                        default=DEFAULT_PROBLEM)
    parser.add_argument('-l', '--linkage-models', type=str,
                        default=DEFAULT_LINKAGE_MODEL)
    parser.add_argument('-d', '--dimensionalities', type=str,
                        default=DEFAULT_DIMENSIONALITY)
    parser.add_argument('-r', '--num-repeats', type=int,
                        default=DEFAULT_DIMENSIONALITY)

    args = parser.parse_args()

    output_dir = args.output_dir
    linkage_models = [t.strip() for t in args.linkage_models.split(",") if len(t.strip()) > 0]
    problems = [t.strip() for t in args.problems.split(",") if len(t.strip()) > 0]
    dimensionalities = [int(t) for t in args.dimensionalities.split(",") if len(t.strip()) > 0]
    num_repeats = args.num_repeats

    # Prepare directory
    os.system(f"mkdir -p {output_dir}")

    for problem in problems:
        for linkage_model in linkage_models:
            for dimensionality in sorted(dimensionalities):
                num_failed_repeats = 0

                for repeat in range(num_repeats):
                    print(f"[Problem] {problem:<15}  "
                          f"[Linkage] {linkage_model:<15}  "
                          f"[Dim] {dimensionality:4}  "
                          f"[Repeat] {repeat:4} -> ", end="", flush=True)
                    base_run_config = RunConfig(
                        linkage_model=linkage_model,
                        population_size=-1,
                        random_seed=-1,
                        problem=problem,
                        dimensionality=dimensionality,
                    )

                    result_population_size, result_median_num_evaluations, result_corrected_num_evaluations = run_bisection(
                        os.path.join(output_dir,
                                     f"{problem},{linkage_model},{dimensionality:04},{repeat:04}"),
                        base_run_config, DEFAULT_NUM_REPEATS_PER_BISECTION_TEST, bisection_repeat=repeat
                    )

                    print(
                        f"[Pop] {result_population_size:4}  "
                        f"[Evals] {int(result_median_num_evaluations):8}  "
                        f"[Corr-Evals] {int(result_corrected_num_evaluations):8}"
                    )

                    if result_corrected_num_evaluations >= DEFAULT_MAX_NUM_EVALUATIONS:
                        num_failed_repeats += 1

                        if num_failed_repeats >= num_repeats * 0.5:
                            break

                if num_failed_repeats >= num_repeats * 0.5:
                    print("Majority of all repeats did not pass, abandoning all larger dimensionalities")
                    break


if __name__ == '__main__':
    main()
