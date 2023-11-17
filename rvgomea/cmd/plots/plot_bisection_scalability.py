import os
import sys

import pandas as pd
import scienceplots
import seaborn as sns
from matplotlib import pyplot as plt, cm

plt.style.use('science')

# Prevent scienceplots from being purged as import
scienceplots.listdir(".")

cmap = cm.get_cmap('jet')
NUM_COLORS = 17


def get_color(index: int):
    cmap(index / (NUM_COLORS - 1))


LINKAGE_MODEL_COLORS = {
    "univariate": get_color(0),
    "full": get_color(16),
    "lt-static-gbo": get_color(1),
    "lt-fb-online-unpruned": get_color(15),
    "lt-fb-online-pruned": get_color(2),
    "ucond-gg-gbo": get_color(14),
    "ucond-fg-gbo": get_color(3),
    "ucond-hg-gbo": get_color(13),
    "mcond-hg-gbo": get_color(4),
    "ucond-gg-fb": get_color(12),
    "ucond-fg-fb": get_color(5),
    "ucond-hg-fb": get_color(11),
    "mcond-hg-fb": get_color(6),
    "ucond-gg-fb-generic": get_color(10),
    "ucond-fg-fb-generic": get_color(7),
    "ucond-hg-fb-generic": get_color(9),
    "mcond-hg-fb-generic": get_color(8),
}


def main(directory):
    df = pd.read_csv(os.path.join(directory, "aggregated_results.csv"))

    f, ax = plt.subplots(figsize=(7, 7))
    sns.lineplot(ax=ax, data=df,
                 x="dimensionality", y="population_size", hue="linkage_model",
                 style="linkage_model", markers=True, err_style="bars")
    ax.legend()
    ax.set_xlabel("Dimensionality")
    ax.set_ylabel("Population size")
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(os.path.join(directory, 'scalability_plot_population_size.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(directory, 'scalability_plot_population_size.png'), bbox_inches='tight')
    plt.clf()

    f, ax = plt.subplots(figsize=(7, 7))
    sns.lineplot(ax=ax, data=df,
                 x="dimensionality", y="median_num_evaluations", hue="linkage_model",
                 style="linkage_model", markers=True, err_style="bars")
    ax.legend()
    ax.set_xlabel("Dimensionality")
    ax.set_ylabel("Number of evaluations")
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(os.path.join(directory, 'scalability_plot_evaluations.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(directory, 'scalability_plot_evaluations.png'), bbox_inches='tight')


if __name__ == '__main__':
    main(sys.argv[1])
