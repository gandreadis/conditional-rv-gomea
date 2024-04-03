import os
import sys

from rvgomea.defaults import DEFAULT_NUM_REPEATS_PER_BISECTION_TEST
from rvgomea.experiments.bisection_runner import run_bisection
from rvgomea.run_config import RunConfig

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
    ("reb5-large-overlap", [10, 20, 40, 80]),
    ("reb5-disjoint-pairs", [9, 18, 36, 72]),
    ("reb-grid", [16, 36, 64, 81]),
]

LINKAGE_MODELS = [
    "mp-hg-gbo-without_clique_seeding-conditional",
    "mp-hg-gbo-with_clique_seeding-conditional",
    "mp-hg-fb_no_order-without_clique_seeding-conditional",
    "mp-hg-fb_no_order-with_clique_seeding-conditional",
    "univariate",
    "lt-fb-online-pruned",
    "vkd-cma",
    "full",
]

BASE_DIR = "data/scalability-bisection-"

REPEATS = 5

SETTINGS = []

for repeat in range(1, REPEATS + 1):
    for problem, dimensions in PROBLEM_DIMENSIONS:
        for dimension in dimensions:
            for linkage_model in LINKAGE_MODELS:
                SETTINGS.append({
                    "repeat": repeat,
                    "problem": problem,
                    "dimension": dimension,
                    "linkage_model": linkage_model,
                })

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
]

for repeat in range(1, REPEATS + 1):
    for problem, dimensions in PROBLEM_DIMENSIONS:
        for dimension in dimensions:
            for linkage_model in LINKAGE_MODELS:
                SETTINGS.append({
                    "repeat": repeat,
                    "problem": problem,
                    "dimension": dimension,
                    "linkage_model": linkage_model,
                })

PROBLEM_DIMENSIONS = [
    ("osoreb", [10, 20, 40, 80]),
    ("osoreb-big-strong", [10, 20, 40, 80]),
    ("osoreb-small-strong", [10, 20, 40, 80]),
]

LINKAGE_MODELS = [
    "univariate",
    "vkd-cma",
    "full",
]

for repeat in range(1, REPEATS + 1):
    for problem, dimensions in PROBLEM_DIMENSIONS:
        for dimension in dimensions:
            for linkage_model in LINKAGE_MODELS:
                SETTINGS.append({
                    "repeat": repeat,
                    "problem": problem,
                    "dimension": dimension,
                    "linkage_model": linkage_model,
                })


def main():
    if len(sys.argv) < 2:
        print("Jobs in array:")
        for i, setting in enumerate(SETTINGS):
            repeat = setting["repeat"]
            problem = setting["problem"]
            dimension = setting["dimension"]
            linkage_model = setting["linkage_model"]
            print(f"[{i + 1:4}] {problem},{linkage_model},{dimension:04},{repeat:04}")
            # print(f"[{i+1:4}] {setting}")
        exit(1)

    job_array_index = int(sys.argv[1])
    setting = SETTINGS[job_array_index - 1]

    repeat = setting["repeat"]
    problem = setting["problem"]
    dimension = setting["dimension"]
    linkage_model = setting["linkage_model"]

    output_dir = BASE_DIR + setting["problem"]
    os.system(f"mkdir -p {output_dir}")

    print(f"[Problem] {problem:<15}  "
          f"[Linkage] {linkage_model:<15}  "
          f"[Dim] {dimension:4}  "
          f"[Repeat] {repeat:4}", flush=True)

    base_run_config = RunConfig(
        linkage_model=linkage_model,
        population_size=-1,
        random_seed=-1,
        problem=problem,
        dimensionality=dimension,
    )

    result_population_size, result_median_num_evaluations, result_corrected_num_evaluations = run_bisection(
        os.path.join(output_dir, f"{problem},{linkage_model},{dimension:04},{repeat:04}"),
        base_run_config, DEFAULT_NUM_REPEATS_PER_BISECTION_TEST, bisection_repeat=repeat
    )

    print(
        f"[Pop] {result_population_size:4}  "
        f"[Evals] {int(result_median_num_evaluations):8}  "
        f"[Corr-Evals] {int(result_corrected_num_evaluations):8}"
    )


if __name__ == '__main__':
    main()
