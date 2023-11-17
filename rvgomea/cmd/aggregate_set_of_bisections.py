import json
import os.path
import sys
from glob import glob

import pandas as pd


def main(directory: str):
    results = []
    for bisection_path in glob(os.path.join(directory, "*")):
        if not os.path.isdir(bisection_path):
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
            "repeat": int(settings[3]),
            "population_size": result["population_size"],
            "median_num_evaluations": result["median_num_evaluations"],
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(directory, "aggregated_results.csv"))


if __name__ == '__main__':
    main(sys.argv[1])
