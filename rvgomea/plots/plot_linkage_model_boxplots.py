import os
import sys

import pandas as pd
from matplotlib import pyplot as plt
import scienceplots
import seaborn as sns

plt.style.use('science')

# Prevent scienceplots from being purged as import
scienceplots.listdir(".")

def main(directory):
    df = pd.read_csv(os.path.join(directory, "aggregated_results.csv"))

    metric, title = ("median_num_evaluations", "Number of evaluations")

    f, ax = plt.subplots(figsize=(5, 5))

    sns.boxplot(ax=ax, data=df, x=metric, y="linkage_model")
    ax.set_xlabel("Linkage model")
    ax.set_ylabel(title)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.savefig(os.path.join(directory, f'boxplot_{metric}.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(directory, f'boxplot_{metric}.png'), bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    main(sys.argv[1])
