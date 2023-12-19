import argparse
import os

import pandas as pd

from rvgomea.defaults import DEFAULT_LINKAGE_MODEL, DEFAULT_PROBLEM, DEFAULT_DIMENSIONALITY, \
    DEFAULT_POPULATION_SIZE, DEFAULT_MAX_NUM_EVALUATIONS, DEFAULT_NUM_BISECTION_REPEATS
from rvgomea.run_config import RunConfig
from rvgomea.run_algorithm import run_algorithm


def main():
    parser = argparse.ArgumentParser(
        prog='Sweep',
        description='Run a simple sweep')

    parser.add_argument('-o', '--output-dir', type=str, required=True)
    parser.add_argument('-p', '--problems', type=str,
                        default=DEFAULT_PROBLEM)
    parser.add_argument('-l', '--linkage-models', type=str,
                        default=DEFAULT_LINKAGE_MODEL)
    parser.add_argument('-d', '--dimensionalities', type=str,
                        default=DEFAULT_DIMENSIONALITY)
    parser.add_argument('-r', '--num-repeats', type=int,
                        default=DEFAULT_NUM_BISECTION_REPEATS)
    parser.add_argument('-s', '--population-size', type=int,
                        default=DEFAULT_POPULATION_SIZE)

    args = parser.parse_args()

    output_dir = args.output_dir
    linkage_models = [t.strip() for t in args.linkage_models.split(",") if len(t.strip()) > 0]
    problems = [t.strip() for t in args.problems.split(",") if len(t.strip()) > 0]
    dimensionalities = [int(t) for t in args.dimensionalities.split(",") if len(t.strip()) > 0]
    num_repeats = args.num_repeats
    population_size = args.population_size

    # Prepare directory
    os.system(f"mkdir -p {output_dir}")

    failed_settings = []
    results = []
    for problem in problems:
        for linkage_model in linkage_models:
            for dimensionality in dimensionalities:
                for repeat in range(num_repeats):
                    print(f"[Problem] {problem:<15}  "
                          f"[Linkage] {linkage_model:<15}  "
                          f"[Dim] {dimensionality:4}  "
                          f"[Repeat] {repeat:4} -> ", end="", flush=True)
                    config = RunConfig(
                        linkage_model=linkage_model,
                        population_size=population_size,
                        random_seed=repeat,
                        problem=problem,
                        dimensionality=dimensionality,
                    )

                    result = run_algorithm(
                        config,
                        in_dir=os.path.join(output_dir,
                                            f"{problem},{linkage_model},{dimensionality:04},{repeat:04}"),
                    )

                    results.append({
                        "problem": problem,
                        "linkage_model": linkage_model,
                        "dimensionality": dimensionality,
                        "repeat": repeat,
                        "population_size": population_size,
                        "median_num_evaluations": result.statistics["evaluations"].iloc[-1],
                    })

                    print(f"[Evals] {int(results[-1]['median_num_evaluations']):8}")

                    if results[-1]["median_num_evaluations"] >= DEFAULT_MAX_NUM_EVALUATIONS:
                        failed_settings.append(results[-1])

    def filter_dict(d):
        return {key: d[key] for key in ("problem", "linkage_model", "dimensionality")}

    for f in failed_settings:
        results = [r for r in results
                   if filter_dict(r) != filter_dict(f)]

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "aggregated_results.csv"))


if __name__ == '__main__':
    main()
