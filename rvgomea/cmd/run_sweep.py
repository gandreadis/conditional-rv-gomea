import argparse
import os

import pandas as pd

from rvgomea.defaults import DEFAULT_LINKAGE_MODEL, DEFAULT_PROBLEM, DEFAULT_DIMENSIONALITY, \
    DEFAULT_NUM_REPEATS_PER_BISECTION_TEST, DEFAULT_BLACK_BOX, DEFAULT_POPULATION_SIZE, DEFAULT_MAX_NUM_EVALUATIONS
from rvgomea.experiments.bisection_runner import run_bisection
from rvgomea.run_config import RunConfig
from rvgomea.run_rvgomea import run_rvgomea


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
    parser.add_argument('-b', '--black-box', action="store_true",
                        default=DEFAULT_BLACK_BOX)
    parser.add_argument('-r', '--num-repeats', type=int,
                        default=5)
    parser.add_argument('-s', '--population-size', type=int,
                        default=DEFAULT_POPULATION_SIZE)

    args = parser.parse_args()

    output_dir = args.output_dir
    linkage_models = [t.strip() for t in args.linkage_models.split(",") if len(t.strip()) > 0]
    problems = [t.strip() for t in args.problems.split(",") if len(t.strip()) > 0]
    dimensionalities = [int(t) for t in args.dimensionalities.split(",") if len(t.strip()) > 0]
    black_box = args.black_box
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
                          f"[BBO] {str(black_box):5}  "
                          f"[Repeat] {repeat:4} -> ", end="", flush=True)
                    config = RunConfig(
                        linkage_model=linkage_model,
                        population_size=population_size,
                        random_seed=repeat,
                        problem=problem,
                        dimensionality=dimensionality,
                        black_box=black_box,
                    )

                    result = run_rvgomea(
                        config,
                        in_dir=os.path.join(output_dir, f"{problem},{linkage_model},{dimensionality:04},{black_box},{repeat:04}"),
                    )

                    results.append({
                        "problem": problem,
                        "linkage_model": linkage_model,
                        "dimensionality": dimensionality,
                        "black_box": black_box,
                        "repeat": repeat,
                        "population_size": population_size,
                        "median_num_evaluations": result.statistics["evaluations"].iloc[-1],
                    })

                    print(f"[Evals] {int(results[-1]['median_num_evaluations']):8}")

                    if int(results[-1]["median_num_evaluations"]) >= int(DEFAULT_MAX_NUM_EVALUATIONS):
                        failed_settings.append(results[-1])
                        break

    def filter_dict(d):
        return {key: d[key] for key in ("problem", "linkage_model", "dimensionality", "black_box")}

    for f in failed_settings:
        results = [r for r in results
                   if filter_dict(r) != filter_dict(f)]

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "aggregated_results.csv"))


if __name__ == '__main__':
    main()
