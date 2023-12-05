import json
import multiprocessing
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from rvgomea.defaults import DEFAULT_MAX_NUM_EVALUATIONS
from rvgomea.run_config import RunConfig
from rvgomea.run_rvgomea import run_rvgomea

MAX_BISECTION_POPULATION = 2048
INITIAL_BISECTION_POPULATION = 8

BISECTION_FAILURE = (MAX_BISECTION_POPULATION, DEFAULT_MAX_NUM_EVALUATIONS)


def bisection_worker(run_config: RunConfig):
    result = run_rvgomea(run_config, run_config.base_dir, save_statistics=False)
    os.system(f"rm -rf {run_config.base_dir}")
    return result


def run_bisection(base_dir: str, base_run_config: RunConfig, num_repeats_per_config: int,
                  num_cpus: int = multiprocessing.cpu_count() - 1, log_progress=False, bisection_repeat: int = 0):
    config_counter = [0]
    history = []
    history_counter = [0]
    population_size_cache = {}

    def save_history(population_size: int, median_num_evaluations: float):
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(base_dir, "bisection_history.csv"))

        with open(os.path.join(base_dir, "bisection_result.json"), "w") as f:
            f.write(json.dumps({
                "population_size": population_size,
                "median_num_evaluations": median_num_evaluations,
            }, indent=4))

    def test_population_size(population_size: int):
        if population_size in population_size_cache.keys():
            median_num_evaluations = population_size_cache[population_size]
        else:
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

            all_passed = all(r.succeeded for r in results)
            if all_passed:
                median_num_evaluations = np.median([r.statistics["evaluations"].iloc[-1] for r in results])
            else:
                median_num_evaluations = DEFAULT_MAX_NUM_EVALUATIONS

            population_size_cache[population_size] = median_num_evaluations

        if log_progress:
            print(f"[POP-SIZE] {population_size:5}  [NUM-EVALS] {median_num_evaluations:8}")

        history.append({
            "iteration": history_counter[0],
            "population_size": population_size,
            "median_num_evaluations": median_num_evaluations,
        })
        history_counter[0] += 1

        return median_num_evaluations

    # Prepare base directory
    os.system(f"rm -rf {base_dir}")
    os.system(f"mkdir -p {base_dir}")

    # Start out with initial set of populations
    pop_size_a = INITIAL_BISECTION_POPULATION
    evals_a = test_population_size(pop_size_a)
    pop_size_b = pop_size_a * 2
    evals_b = test_population_size(pop_size_b)
    pop_size_d = pop_size_a * 4
    evals_d = test_population_size(pop_size_d)

    # Increase population size range
    while evals_d <= evals_b or evals_b >= DEFAULT_MAX_NUM_EVALUATIONS:
        pop_size_b = pop_size_d
        evals_b = evals_d
        pop_size_d *= 2
        evals_d = test_population_size(pop_size_d)

        if pop_size_d > MAX_BISECTION_POPULATION:
            save_history(*BISECTION_FAILURE)
            return BISECTION_FAILURE

    # Conduct bisection
    num_iterations = 0
    while num_iterations < 100:
        pop_size_b = int(pop_size_a + (pop_size_d - pop_size_a) / 3.0)
        evals_b = test_population_size(pop_size_b)
        pop_size_c = int(pop_size_a + 2.0 * (pop_size_d - pop_size_a) / 3.0)
        evals_c = test_population_size(pop_size_c)

        if abs(pop_size_c - pop_size_b) <= 1:
            if evals_a < evals_b and evals_a < evals_c:
                save_history(pop_size_a, evals_a)
                return pop_size_a, evals_a

            if evals_b < evals_c:
                save_history(pop_size_b, evals_b)
                return pop_size_b, evals_b
            else:
                save_history(pop_size_c, evals_c)
                return pop_size_c, evals_c

        if evals_b < evals_c:
            pop_size_d = pop_size_c
            evals_d = evals_c
        elif evals_c <= evals_b:
            pop_size_a = pop_size_b
            evals_a = evals_b

        num_iterations += 1

    save_history(*BISECTION_FAILURE)
    return BISECTION_FAILURE
