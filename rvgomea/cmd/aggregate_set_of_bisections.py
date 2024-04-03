import json
import os.path
import sys
from glob import glob

import pandas as pd

from rvgomea.defaults import DEFAULT_MAX_NUM_EVALUATIONS, DEFAULT_NUM_BISECTION_REPEATS

FILTER_KEYS = ("problem", "linkage_model", "dimensionality")


def filter_dict(d):
    return tuple(d[key] for key in FILTER_KEYS if key in d)


def main(directory: str):
    failed_settings = {}
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

        data = {
            "problem": settings[0],
            "linkage_model": settings[1],
            "dimensionality": int(settings[2]),
            "repeat": int(settings[3]),
            "population_size": result["population_size"],
            "median_num_evaluations": result["median_num_evaluations"],
            "corrected_num_evaluations": result["corrected_num_evaluations"],
        }

        # Address REBGrid 4th dimension having slightly passing median than 3th dimension (where it already fails)
        if data["problem"] == "reb-grid" and data["linkage_model"] == "full" and data["dimensionality"] >= 40:
            continue

        results.append(data)

        if result["corrected_num_evaluations"] >= DEFAULT_MAX_NUM_EVALUATIONS:
            s = filter_dict(results[-1])
            if s not in failed_settings.keys():
                failed_settings[s] = 0
            failed_settings[s] += 1

    results = [
        r for r in results
        if not (filter_dict(r) in failed_settings.keys() and failed_settings[
            filter_dict(r)] >= 0.5 * DEFAULT_NUM_BISECTION_REPEATS)
    ]

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(directory, "aggregated_results.csv"))


if __name__ == '__main__':
    main(sys.argv[1])
