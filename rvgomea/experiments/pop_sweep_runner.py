import multiprocessing
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from rvgomea.defaults import DEFAULT_MAX_NUM_EVALUATIONS
from rvgomea.run_config import RunConfig
from rvgomea.run_algorithm import run_algorithm


def bisection_worker(run_config: RunConfig):
    result = run_algorithm(run_config, run_config.base_dir, save_statistics=False)
    os.system(f"rm -rf {run_config.base_dir}")
    return result


def run_pop_sweep(base_dir: str, base_run_config: RunConfig, num_repeats_per_config: int,
                  num_cpus: int = multiprocessing.cpu_count(), log_progress=False, bisection_repeat: int = 0):
    config_counter = [0]
    history = []
    history_counter = [0]

    def save_history():
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(base_dir, "pop_size_history.csv"))

    def test_population_size(population_size: int):
        configs = []
        for repeat in range(num_repeats_per_config):
            derived_config = base_run_config.copy()
            derived_config.config_id = config_counter[0]
            derived_config.base_dir = os.path.join(base_dir, str(derived_config.config_id))
            derived_config.population_size = population_size
            derived_config.random_seed = bisection_repeat * 100000 + repeat
            configs.append(derived_config)

            config_counter[0] += 1

        with Pool(num_cpus) as pool:
            results = list(pool.imap_unordered(bisection_worker, configs))

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
            corrected_num_evaluations = (succeeded_run_evals_sum / float(len(succeeded_run_evals))) / success_rate

        cholesky_fails = np.median([
            r.cholesky_fails for r in results
        ])

        if log_progress:
            print(
                f"[POP-SIZE] {population_size:5}  "
                f"[NUM-EVALS] {median_num_evaluations:8}  "
                f"[CORR-NUM-EVALS] {corrected_num_evaluations:8} "
                f"[CHOL] {cholesky_fails:6}"
            )

        history.append({
            "iteration": history_counter[0],
            "population_size": population_size,
            "median_num_evaluations": median_num_evaluations,
            "corrected_num_evaluations": corrected_num_evaluations,
            "num_failures": num_repeats_per_config - len(succeeded_run_evals),
            "median_num_cholesky_fails": cholesky_fails,
        })
        history_counter[0] += 1

        return median_num_evaluations, corrected_num_evaluations

    # Prepare base directory
    os.system(f"rm -rf {base_dir}")
    os.system(f"mkdir -p {base_dir}")

    for pop_size in range(50, 500, 10):
        test_population_size(pop_size)

    save_history()
