from dataclasses import dataclass

import pandas as pd

from rvgomea.run_config import RunConfig


@dataclass
class RunResult:
    config: RunConfig
    statistics: pd.DataFrame
    cholesky_fails: int
    succeeded: bool
