import os

from rvgomea.convert_statistics import convert_statistics
from rvgomea.run_config import RunConfig
from rvgomea.run_result import RunResult

LINKAGE_MODEL_CODES = {
    "univariate": 1,
    "full": -1,
    "ucond-GG": -1011,
    "ucond-fg": -1101,
    "ucond-hg": -1111,
    "mcond-hg": -100111,
}

PROBLEM_CODES = {
    "sphere": 0,
    "rosenbrock": 7,
}


def run_rvgomea(config: RunConfig, in_dir=None, show_output=False, save_statistics=True) -> RunResult:
    command = ""

    if in_dir is not None:
        command += f"mkdir -p {in_dir} && cd {in_dir} && "

    # Clear previous output files
    command += "rm -f *.dat && rm -f *.csv && rm -f *.json && "

    exe_path = os.path.join(os.path.dirname(__file__), "..", "RV-GOMEA")
    command += f"{exe_path} "

    # Write generational stats
    command += "-s "

    # Set linkage model
    if config.linkage_model.lower() not in LINKAGE_MODEL_CODES.keys():
        raise Exception(f"Unknown linkage model: {config.linkage_model}")
    linkage_model_code = LINKAGE_MODEL_CODES[config.linkage_model.lower()]
    command += f"-f {linkage_model_code} "

    # Set random seed
    if config.random_seed >= 0:
        command += f"-S {config.random_seed} "

    # Enable value to reach
    command += "-r "

    # Set problem
    if config.problem.lower() not in PROBLEM_CODES.keys():
        raise Exception(f"Unknown problem: {config.problem}")
    problem_code = PROBLEM_CODES[config.problem.lower()]
    command += f"{problem_code} "

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

    output = os.popen(command).read()

    if show_output:
        print(output)

    processed_base_dir = os.getcwd()
    if in_dir is not None:
        processed_base_dir = in_dir

    config.base_dir = processed_base_dir

    # Save run config for future reference
    config.to_json(os.path.join(processed_base_dir, "run_config.json"))

    # Convert statistics to CSV
    statistics = convert_statistics(os.path.join(processed_base_dir, "statistics.dat"),
                                    os.path.join(processed_base_dir, "statistics.csv")
                                    if save_statistics else None)
    assert len(statistics) > 0, f"No generations were executed with RunConfig {config}"

    succeeded = (statistics["best_objective"].iloc[-1] <= config.vtr and
                 statistics["evaluations"].iloc[-1] <= config.max_num_evaluations)

    if show_output:
        print(f"Succeeded: {succeeded}")

    return RunResult(config, statistics, succeeded)
