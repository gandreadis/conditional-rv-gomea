import os.path
import sys
from glob import glob

FRAGMENTS = ("shark2",)  # ("shark", "snellius1", "snellius2")


def main(all_scalability_results_directory):
    for fragment in FRAGMENTS:
        directory = os.path.join(all_scalability_results_directory, fragment, "logs")
        for log_path in glob(os.path.join(directory, "*.out")):
            with open(log_path) as f:
                t = f.read()
                if "Done." not in t:
                    print(f"FAILED: {log_path}")


if __name__ == '__main__':
    main(sys.argv[1])
