import json
import math
import multiprocessing
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from rvgomea.defaults import DEFAULT_MAX_NUM_EVALUATIONS
from rvgomea.run_config import RunConfig
from rvgomea.run_algorithm import run_algorithm

MAX_BISECTION_POPULATION = 2048
MIN_BISECTION_POPULATION = 8

BISECTION_FAILURE = (MAX_BISECTION_POPULATION, DEFAULT_MAX_NUM_EVALUATIONS, DEFAULT_MAX_NUM_EVALUATIONS)


def bisection_worker(run_config: RunConfig):
    result = run_algorithm(run_config, run_config.base_dir, save_statistics=False)
    os.system(f"rm -rf {run_config.base_dir}")
    return result


def run_bisection(base_dir: str, base_run_config: RunConfig, num_repeats_per_config: int,
                  num_cpus: int = multiprocessing.cpu_count(), log_progress=False, bisection_repeat: int = 0):
    config_counter = [0]
    history = []
    history_counter = [0]
    population_size_cache = {}

    def save_history(population_size: int, median_num_evaluations: float, corrected_num_evaluations: float):
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(base_dir, "bisection_history.csv"))

        with open(os.path.join(base_dir, "bisection_result.json"), "w") as f:
            f.write(json.dumps({
                "population_size": population_size,
                "median_num_evaluations": median_num_evaluations,
                "corrected_num_evaluations": corrected_num_evaluations,
            }, indent=4))

    def test_population_size(population_size: int):
        if population_size in population_size_cache.keys():
            median_num_evaluations, corrected_num_evaluations = population_size_cache[population_size]
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

            population_size_cache[population_size] = median_num_evaluations, corrected_num_evaluations

        if log_progress:
            print(
                f"[POP-SIZE] {population_size:5}  [NUM-EVALS] {median_num_evaluations:10.4}  [CORR-NUM-EVALS] {corrected_num_evaluations:12.4}")

        history.append({
            "iteration": history_counter[0],
            "population_size": population_size,
            "median_num_evaluations": median_num_evaluations,
            "corrected_num_evaluations": corrected_num_evaluations,
        })
        history_counter[0] += 1

        return median_num_evaluations, corrected_num_evaluations

    # Prepare base directory
    os.system(f"rm -rf {base_dir}")
    os.system(f"mkdir -p {base_dir}")

    # Start out with guideline
    pop_size_guideline = min(int(17 + 3 * math.pow(base_run_config.dimensionality, 1.5)), MAX_BISECTION_POPULATION)
    pop_size_upper = pop_size_guideline
    med_evals_upper, cor_evals_upper = test_population_size(pop_size_upper)
    med_evals_guideline, cor_evals_guideline = med_evals_upper, cor_evals_upper

    # Need to search for higher pop size, despite guideline
    while cor_evals_upper >= DEFAULT_MAX_NUM_EVALUATIONS:
        pop_size_upper *= 2
        if pop_size_upper > MAX_BISECTION_POPULATION:
            pop_size_upper = pop_size_guideline
            med_evals_upper, cor_evals_upper = med_evals_guideline, cor_evals_guideline
            break

        med_evals_upper, cor_evals_upper = test_population_size(pop_size_upper)

    pop_size_lower = int(pop_size_upper * 0.5)
    med_evals_lower, cor_evals_lower = test_population_size(pop_size_lower)

    while cor_evals_lower < cor_evals_upper:
        pop_size_lower = int(pop_size_lower * 0.5)
        if pop_size_lower < MIN_BISECTION_POPULATION:
            pop_size_lower = MIN_BISECTION_POPULATION
            med_evals_lower, cor_evals_lower = test_population_size(pop_size_lower)
            break

        med_evals_lower, cor_evals_lower = test_population_size(pop_size_lower)

    pop_size_a = pop_size_lower
    med_evals_a, cor_evals_a = med_evals_lower, cor_evals_lower
    pop_size_d = pop_size_upper
    med_evals_d, cor_evals_d = med_evals_upper, cor_evals_upper

    # Conduct bisection
    num_iterations = 0
    while num_iterations < 100:
        pop_size_b = int(pop_size_a + (pop_size_d - pop_size_a) / 3.0)
        med_evals_b, cor_evals_b = test_population_size(pop_size_b)
        pop_size_c = int(pop_size_a + 2.0 * (pop_size_d - pop_size_a) / 3.0)
        med_evals_c, cor_evals_c = test_population_size(pop_size_c)

        if abs(pop_size_c - pop_size_b) <= 1:
            if cor_evals_a < cor_evals_b and cor_evals_a < cor_evals_c:
                save_history(pop_size_a, med_evals_a, cor_evals_a)
                return pop_size_a, med_evals_a, cor_evals_a

            if cor_evals_b < cor_evals_c:
                save_history(pop_size_b, med_evals_b, cor_evals_b)
                return pop_size_b, med_evals_b, cor_evals_b
            else:
                save_history(pop_size_c, med_evals_c, cor_evals_c)
                return pop_size_c, med_evals_c, cor_evals_c

        if cor_evals_b < cor_evals_c:
            pop_size_d = pop_size_c
            med_evals_d = med_evals_c
            cor_evals_d = cor_evals_c
        elif cor_evals_c <= cor_evals_b:
            pop_size_a = pop_size_b
            med_evals_a = med_evals_b
            cor_evals_a = cor_evals_b

        num_iterations += 1

    save_history(*BISECTION_FAILURE)
    return BISECTION_FAILURE
