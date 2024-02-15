import os

import pandas as pd

from rvgomea.convert_statistics import convert_statistics
from rvgomea.run_config import RunConfig
from rvgomea.run_result import RunResult

LINKAGE_MODEL_CODES = {
    "univariate": 1,
    "full": -1,
    "vkd-cma": -100,
    "lt-static-gbo": -4,
    "lt-fb-online-unpruned": -5,
    "lt-fb-online-pruned": -6,
    "mp-fb-online-gg": -7,
    "mp-fb-online-fg": -8,
    "mp-fb-online-hg": -9,
}

# Generate linkage model combinations
# Label encoding: max clique - factorized elements - full element - fitness-based - seed cliques - conditional
for max_clique_label, max_clique_size in (("uni", "1"), ("mp", "100")):
    for factorization_label, factorization in (("lt", "00"), ("gg", "01"), ("fg", "10"), ("hg", "11")):
        for fitness_based_label, fitness_based in (("gbo", "0"), ("fb", "1"), ("fb_no_order", "2")):
            for seed_cliques_label, seed_cliques in (
            ("without_clique_seeding", "0"), ("with_clique_seeding", "1"), ("with_clique_seeding_and_uni", "2"),
            ("pruned", "3")):
                for conditional_label, conditional in (("non_conditional", "0"), ("conditional", "1")):
                    for set_cover_label, set_cover in (("", "0"), ("-set_cover", "1")):
                        if "with_clique_seeding" in seed_cliques_label and factorization_label == "lt":
                            continue
                        if "pruned" in seed_cliques_label and factorization_label != "lt":
                            continue

                        LINKAGE_MODEL_CODES[
                            f"{max_clique_label}-{factorization_label}-{fitness_based_label}-{seed_cliques_label}-{conditional_label}{set_cover_label}"
                        ] = f"-{max_clique_size}{factorization}{fitness_based}{seed_cliques}{conditional}{set_cover}"

PROBLEM_CODES = {
    "sphere": 0,
    "rosenbrock": 7,
    "reb2-chain-alternating": 216191,
    "reb5-no-overlap": 506699,
    "reb5-small-overlap": 516699,
    "reb5-small-overlap-alternating": 516191,
    "reb5-large-overlap": 546699,
    "reb5-disjoint-pairs": 14,
    "osoreb": 16,
    "osoreb-big-strong": 17,
    "osoreb-small-strong": 18,
    "reb-grid": 20,  # only square
}

# For the reb5 problems, the following problem sizes are compatible:
# reb5-no-overlap: multiples of 5
# reb5-small-overlap*: 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81
# reb5-large-overlap: anything >= 5
# reb5-disjoint-pairs: multiples of 9

for rot_angle in range(10):
    for cond_number in range(1, 7):
        PROBLEM_CODES[
            f"reb-chain-condition-{cond_number}-rotation-{rot_angle * 5}"] = 210000 + cond_number * 1000 + cond_number * 100 + rot_angle * 10 + rot_angle

PROBLEM_CODES["reb2-chain-weak"] = PROBLEM_CODES["reb-chain-condition-1-rotation-5"]
PROBLEM_CODES["reb2-chain-strong"] = PROBLEM_CODES["reb-chain-condition-6-rotation-45"]

INIT_RANGES = {
    "sphere": [-115, -100],
    "rosenbrock": [-115, -100],
    "reb-grid": [-115, -100],
    "osoreb": [-115, -100],
    "osoreb-big-strong": [-115, -100],
    "osoreb-small-strong": [-115, -100],
    "reb2": [-115, -100],
    "reb5": [-115, -100],
}


def run_algorithm(config: RunConfig, in_dir=None, show_output=False, save_statistics=True,
                  save_fitness_dependencies=False) -> RunResult:
    command = ""

    if in_dir is not None:
        command += f"mkdir -p {in_dir} && cd {in_dir} && "

    processed_base_dir = os.getcwd()
    if in_dir is not None:
        processed_base_dir = in_dir

    # Clear previous output files
    command += "rm -f *.dat && rm -f *.csv && rm -f *.json && "

    # Intercept VkD-CMA runs
    if config.linkage_model == "vkd-cma":
        command += f"cd {os.getcwd()} && "
        command += f"python rvgomea/experiments/vkdcma.py "

        command += f"-i {processed_base_dir} "
        command += f"-p {config.problem} "
        command += f"-d {config.dimensionality} "
        command += f"-s {config.population_size} "
        if show_output:
            command += f"-o "
        command += f"-r {config.random_seed} "

        output = os.popen(command).read()

        if show_output:
            print(output)

        statistics = pd.read_csv(os.path.join(processed_base_dir, "statistics.csv"))

        succeeded = (statistics["best_objective"].iloc[-1] <= config.vtr and
                     statistics["evaluations"].iloc[-1] < config.max_num_evaluations and
                     statistics["seconds"].iloc[-1] < config.max_num_seconds)

        if not save_statistics:
            os.system(f"rm {os.path.join(processed_base_dir, 'statistics.csv')}")

        if show_output:
            print(f"Succeeded: {succeeded}")

        return RunResult(config, statistics, -1, succeeded)

    exe_path = os.path.join(os.path.dirname(__file__), "..", "RV-GOMEA")
    command += f"{exe_path} "

    # Write generational stats
    command += "-s "

    if save_fitness_dependencies:
        command += "-d "

    # Set linkage model
    if config.linkage_model.lower() not in LINKAGE_MODEL_CODES.keys():
        raise Exception(f"Unknown linkage model: {config.linkage_model}")
    linkage_model_code = LINKAGE_MODEL_CODES[config.linkage_model.lower()]
    command += f"-f {linkage_model_code} "

    # Set random seed
    if config.random_seed >= 0:
        if config.random_seed == 0:
            config.random_seed = 1928374

        command += f"-S {config.random_seed} "

    # Set black-box
    if config.black_box:
        command += f"-b "

    # Enable value to reach
    command += "-r "

    # Set problem
    if config.problem.lower() not in PROBLEM_CODES.keys():
        raise Exception(f"Unknown problem: {config.problem}")
    problem_code = PROBLEM_CODES[config.problem.lower()]
    command += f"{problem_code} "

    if "reb2" in config.problem.lower() or "reb-chain" in config.problem.lower():
        config.lower_init_bound, config.upper_init_bound = INIT_RANGES["reb2"]
    elif "reb5" in config.problem.lower():
        config.lower_init_bound, config.upper_init_bound = INIT_RANGES["reb5"]
    else:
        config.lower_init_bound, config.upper_init_bound = INIT_RANGES[config.problem.lower()]

    # Set further parameters
    command += f"{config.dimensionality} "
    command += f"{config.lower_init_bound} "
    command += f"{config.upper_init_bound} "
    command += f"{config.rotation_angle} "
    command += f"{config.tau} "
    command += f"{config.population_size} "
    command += f"{config.num_populations} "
    command += f"{config.distribution_multiplier_decrease} "
    command += f"{config.st_dev_threshold} "
    command += f"{config.max_num_evaluations} "
    command += f"{config.vtr} "
    command += f"{config.max_no_improvement_stretch} "
    command += f"{config.fitness_variance_tolerance} "
    command += f"{config.max_num_seconds} "

    if show_output:
        print("Command:", command)

    output = os.popen(command).read()

    # lines = output.split("\n")
    # lines = [l for l in lines if len(l.strip()) > 0]
    # cholesky_fails = int(lines[-1].split(" ")[-1])

    if show_output:
        print(output)

    config.base_dir = processed_base_dir

    # Save run config for future reference
    config.to_json(os.path.join(processed_base_dir, "run_config.json"))

    if not os.path.exists(os.path.join(processed_base_dir, "statistics.dat")):
        if show_output:
            print("Aborting, no statistics.dat found")
        return RunResult(config, None, -1, False)

    # Convert statistics to CSV
    statistics = convert_statistics(os.path.join(processed_base_dir, "statistics.dat"),
                                    os.path.join(processed_base_dir, "statistics.csv")
                                    if save_statistics else None)
    assert len(statistics) > 0, f"No generations were executed with RunConfig {config}"

    succeeded = (statistics["best_objective"].iloc[-1] <= config.vtr and
                 statistics["evaluations"].iloc[-1] <= config.max_num_evaluations and
                 statistics["seconds"].iloc[-1] <= config.max_num_seconds)

    if show_output:
        print(f"Succeeded: {succeeded}")

    return RunResult(config, statistics, -1, succeeded)
