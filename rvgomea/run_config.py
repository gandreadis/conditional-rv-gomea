import json
from dataclasses import dataclass, is_dataclass, asdict

from rvgomea.defaults import *


@dataclass
class RunConfig:
    # Algorithm
    linkage_model: str = DEFAULT_LINKAGE_MODEL
    population_size: int = DEFAULT_POPULATION_SIZE
    num_populations: float = DEFAULT_NUM_POPULATIONS
    tau: float = DEFAULT_TAU
    distribution_multiplier_decrease: float = DEFAULT_DISTRIBUTION_MULTIPLIER_DECREASE
    st_dev_threshold: float = DEFAULT_ST_DEV_THRESHOLD
    random_seed: int = DEFAULT_RANDOM_SEED

    # Problem
    problem: str = DEFAULT_PROBLEM
    dimensionality: int = DEFAULT_DIMENSIONALITY
    black_box: bool = DEFAULT_BLACK_BOX
    lower_init_bound: float = DEFAULT_LOWER_INIT_BOUND
    upper_init_bound: float = DEFAULT_UPPER_INIT_BOUND
    vtr: float = DEFAULT_VTR
    rotation_angle: float = DEFAULT_ROTATION_ANGLE

    # Computation Budget
    max_num_evaluations: float = DEFAULT_MAX_NUM_EVALUATIONS
    max_num_seconds: float = DEFAULT_MAX_NUM_SECONDS
    max_no_improvement_stretch: int = DEFAULT_MAX_NO_IMPROVEMENT_STRETCH
    fitness_variance_tolerance: float = DEFAULT_FITNESS_VARIANCE_TOLERANCE

    base_dir: str = ""
    config_id: int = -1

    def to_json(self, filename: str):
        with open(filename, "w") as f:
            f.write(json.dumps(self, cls=EnhancedJSONEncoder, indent=4))

    def copy(self):
        return RunConfig(**self.__dict__)


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)
