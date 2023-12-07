import json
import os.path
import sys
from glob import glob

import pandas as pd

from rvgomea.defaults import DEFAULT_MAX_NUM_EVALUATIONS


def main(directory: str):
    failed_settings = []
    results = []
    for bisection_path in glob(os.path.join(directory, "*")):
        if not os.path.isdir(bisection_path) or bisection_path.endswith("plots"):
            continue

        try:
            with open(os.path.join(bisection_path, "bisection_result.json")) as f:
                result = json.load(f)
        except Exception as e:
            print(f"Skipping {bisection_path}, {e}")
            continue

        settings = bisection_path.split("/")[-1].split(",")
        results.append({
            "problem": settings[0],
            "linkage_model": settings[1],
            "dimensionality": int(settings[2]),
            "black_box": bool(settings[3]),
            "repeat": int(settings[4]),
            "population_size": result["population_size"],
            "median_num_evaluations": result["median_num_evaluations"],
        })

        if int(result["median_num_evaluations"]) == int(DEFAULT_MAX_NUM_EVALUATIONS):
            failed_settings.append(results[-1])

    def filter_dict(d):
        return {key: d[key] for key in ("problem", "linkage_model", "dimensionality", "black_box")}

    for f in failed_settings:
        results = [r for r in results
                   if filter_dict(r) != filter_dict(f)]

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(directory, "aggregated_results.csv"))


if __name__ == '__main__':
    main(sys.argv[1])
