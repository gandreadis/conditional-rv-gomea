import multiprocessing
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

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
    "osoreb-big-strong": [160, 320],
    "osoreb-small-strong": [160, 320],
    "reb5-large-overlap": [160, 320],
    "reb5-disjoint-pairs": [162, 324],
    "reb-grid": [169, 324],
}

PROBLEM_DIMENSIONS = [
    ("sphere", [10, 20, 40, 80]),
    ("rosenbrock", [10, 20, 40, 80]),
    ("reb2-chain-weak", [10, 20, 40, 80]),
    ("reb2-chain-strong", [10, 20, 40, 80]),
    ("reb2-chain-alternating", [10, 20, 40, 80]),
    ("reb5-no-overlap", [10, 20, 40, 80]),
    ("reb5-small-overlap", [9, 21, 41, 81]),
    ("reb5-small-overlap-alternating", [9, 21, 41, 81]),
    ("osoreb", [10, 20, 40, 80]),
    ("osoreb-big-strong", [10, 20, 40, 80]),
    ("osoreb-small-strong", [10, 20, 40, 80]),
    ("reb5-large-overlap", [10, 20, 40, 80]),
    ("reb5-disjoint-pairs", [9, 18, 36, 72]),
    ("reb-grid", [16, 36, 64, 81]),
]

LINKAGE_MODELS = [
    "uni-hg-gbo-without_clique_seeding-conditional",
    "mp-hg-gbo-without_clique_seeding-conditional",
    "mp-hg-gbo-with_clique_seeding-conditional",
    "uni-hg-fb_no_order-without_clique_seeding-conditional",
    "mp-hg-fb_no_order-without_clique_seeding-conditional",
    "mp-hg-fb_no_order-with_clique_seeding-conditional",
    "lt-fb-online-pruned",
    "univariate",
    "vkd-cma",
    "full",
]


def sweep_worker(run_config: RunConfig):
    result = run_algorithm(run_config, run_config.base_dir, save_statistics=False)
    os.system(f"rm -rf {run_config.base_dir}")
    return result


def main():
    aggregated_base_dir_stub = sys.argv[1]
    extended_base_dir = sys.argv[2]
    filter_problem_index = int(sys.argv[3])
    global_repeat = filter_problem_index // len(DIMENSIONALITY_EXTENSIONS.items())
    filter_problem_index = filter_problem_index % len(DIMENSIONALITY_EXTENSIONS.items())

    os.system(f"mkdir -p {extended_base_dir}")

    for problem_index, (problem, dimensions) in enumerate(PROBLEM_DIMENSIONS):
        if problem_index != filter_problem_index:
            continue

        print(f"Problem: {problem}    - Repeat: {global_repeat}")
        df = pd.read_csv(os.path.join(aggregated_base_dir_stub + problem, "aggregated_results.csv"))

        problem_rows = []

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
                next_pop_sizes = [pop_size_last, pop_size_last]
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
                    next_pop_size = min(int(round(next_pop_size)), 10000)
                    next_pop_sizes.append(next_pop_size)

            for d, p in zip(DIMENSIONALITY_EXTENSIONS[problem], next_pop_sizes):
                configs = []
                num_repeats_per_config = 30

                for repeat in range(num_repeats_per_config):
                    derived_config = RunConfig(
                        linkage_model=linkage_model,
                        population_size=p,
                        random_seed=global_repeat * 5000 + repeat + 1,
                        problem=problem,
                        dimensionality=d,
                        base_dir=os.path.join(extended_base_dir, problem, str(global_repeat), str(repeat)),
                    )
                    configs.append(derived_config)

                with Pool(multiprocessing.cpu_count()) as pool:
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

                print(f"Dimensionality: {d:3} | Pop. size: {p:4} | Corr. num. evals: {corrected_num_evaluations:12.0f}")
                problem_rows.append({
                    "problem": problem,
                    "repeat": global_repeat,
                    "linkage_model": linkage_model,
                    "dimensionality": d,
                    "population_size": p,
                    "median_num_evaluations": median_num_evaluations,
                    "corrected_num_evaluations": corrected_num_evaluations,
                })

                if corrected_num_evaluations >= DEFAULT_MAX_NUM_EVALUATIONS:
                    print("Skipping second dimension as first failed")
                    break

            print()

        print("-----------")

        pd.DataFrame(problem_rows).to_csv(os.path.join(aggregated_base_dir_stub + problem,
                                                       f"extrapolated_results_{global_repeat}.csv"))


if __name__ == '__main__':
    main()
