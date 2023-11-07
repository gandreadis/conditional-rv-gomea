import os

from rvgomea.run_config import RunConfig


LINKAGE_MODEL_CODES = {
    "UNIVARIATE": 1,
    "FULL": -1,
    "UCond-GG": -1011,
    "UCond-FG": -1101,
    "UCond-HG": -1111,
    "MCond-HG": -100111,
}

PROBLEM_CODES = {
    "SPHERE": 0,
    "ROSENBROCK": 7,
}


def run_rvgomea(config: RunConfig):
    command = "./RV-GOMEA "

    # Set linkage model
    if config.linkage_model not in LINKAGE_MODEL_CODES.keys():
        raise Exception(f"Unknown linkage model: {config.linkage_model}")
    linkage_model_code = LINKAGE_MODEL_CODES[config.linkage_model]
    command += f"-f {linkage_model_code} "

    # Set random seed
    if config.random_seed >= 0:
        command += f"-S {config.random_seed} "

    # Enable value to reach
    command += "-r "

    # Set problem
    if config.problem not in PROBLEM_CODES.keys():
        raise Exception(f"Unknown problem: {config.problem}")
    problem_code = PROBLEM_CODES[config.problem]
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
    print(output)
