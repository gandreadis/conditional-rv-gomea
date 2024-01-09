import os
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from rvgomea.cmd.run_parallel_set_of_bisections import PROBLEM_DIMENSIONS, LINKAGE_MODELS
from rvgomea.defaults import DEFAULT_MAX_NUM_EVALUATIONS
from rvgomea.run_algorithm import run_algorithm
from rvgomea.run_config import RunConfig

DIMENSIONALITY_EXTENSIONS = {
    "sphere": [160, 320],
    "rosenbrock": [160, 320],
    "reb2-chain-weak": [160, 320],
    "reb2-chain-strong": [160, 320],
    "reb2-chain-alternating": [160, 320],
    "reb5-no-overlap": [160, 320],
    "reb5-small-overlap": [161, 321],
    "reb5-small-overlap-alternating": [161, 321],
    "osoreb": [160, 320],
    "reb5-large-overlap": [160, 320],
    "reb5-disjoint-pairs": [162, 324],
    "reb-grid": [169, 324],
}


def sweep_worker(run_config: RunConfig):
    result = run_algorithm(run_config, run_config.base_dir, save_statistics=False)
    os.system(f"rm -rf {run_config.base_dir}")
    return result


def main():
    aggregated_base_dir_stub = sys.argv[1]
    extended_base_dir = sys.argv[2]

    rows = []

    for problem, dimensions in PROBLEM_DIMENSIONS:
        print(f"Problem: {problem}")
        df = pd.read_csv(os.path.join(aggregated_base_dir_stub + problem, "aggregated_results.csv"))

        for linkage_model in LINKAGE_MODELS:
            print(f"Linkage model: {linkage_model}")
            lm_df = df[df["linkage_model"] == linkage_model]
            rows_first_dim = lm_df[lm_df["dimensionality"] == dimensions[0]]
            rows_last_dim = lm_df[lm_df["dimensionality"] == dimensions[-1]]

            if len(rows_first_dim) == 0 or len(rows_last_dim) == 0:
                continue

            pop_size_first = np.median(rows_first_dim["population_size"])
            pop_size_last = np.median(rows_last_dim["population_size"])

            if pop_size_last <= pop_size_first:
                next_pop_sizes = [pop_size_first, pop_size_first]
            else:
                log_pop_size_diff = np.log10(pop_size_last) - np.log10(pop_size_first)
                log_dist_diff = np.log10(dimensions[-1]) - np.log10(dimensions[0])

                log_slope = log_pop_size_diff / log_dist_diff
                assert np.power(10, log_slope) >= 0, "Negative slope"

                next_pop_sizes = []
                for next_dim in DIMENSIONALITY_EXTENSIONS[problem]:
                    log_dim_diff = np.log10(next_dim) - np.log10(dimensions[-1])
                    log_projection = np.log10(pop_size_last) + log_slope * log_dim_diff
                    next_pop_size = np.power(10, log_projection)
                    next_pop_size = min(next_pop_size, 10000)
                    next_pop_sizes.append(next_pop_size)

            for d, p in zip(DIMENSIONALITY_EXTENSIONS[problem], next_pop_sizes):
                configs = []
                num_repeats_per_config = 30

                for repeat in range(num_repeats_per_config):
                    derived_config = RunConfig(
                        linkage_model=linkage_model,
                        population_size=p,
                        random_seed=repeat + 1,
                        problem=problem,
                        dimensionality=d,
                        base_dir=os.path.join(extended_base_dir, str(repeat)),
                    )
                    configs.append(derived_config)

                with Pool(5) as pool:
                    results = list(pool.imap_unordered(sweep_worker, configs))

                median_num_evaluations = np.median([
                    (r.statistics["evaluations"].iloc[-1] if r.succeeded else DEFAULT_MAX_NUM_EVALUATIONS)
                    for r in results
                ])

                succeeded_run_evals = []
                succeeded_run_evals_sum = 0
                for r in results:
                    if r.succeeded:
                        succeeded_run_evals.append(r.statistics["evaluations"].iloc[-1])
                        succeeded_run_evals_sum += succeeded_run_evals[-1]

                if len(succeeded_run_evals) == 0:
                    corrected_num_evaluations = DEFAULT_MAX_NUM_EVALUATIONS
                else:
                    success_rate = float(len(succeeded_run_evals)) / float(num_repeats_per_config)
                    corrected_num_evaluations = (succeeded_run_evals_sum / float(
                        len(succeeded_run_evals))) / success_rate

                print(f"Dimensionality: {d:4}   Pop. size: {p:4}   Corr. num. evals: {corrected_num_evaluations:12.0f}")
                rows.append({
                    "problem": problem,
                    "linkage_model": linkage_model,
                    "dimensionality": d,
                    "population_size": p,
                    "median_num_evaluations": median_num_evaluations,
                    "corrected_num_evaluations": corrected_num_evaluations,
                })

            print()

        print("-----------")

    pd.DataFrame(rows).to_csv(os.path.join(extended_base_dir, "extended_results.csv"))


if __name__ == '__main__':
    main()
