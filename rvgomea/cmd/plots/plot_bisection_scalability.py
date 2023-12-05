import os
import sys

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
import scienceplots
import seaborn as sns

plt.style.use('science')

# Prevent scienceplots from being purged as import
scienceplots.listdir(".")

cmap = matplotlib.colormaps['tab20c']


MARKERS = {
    "univariate": ".",
    "full": "v",
    "lt-static-gbo": "^",
    "lt-fb-online-unpruned": "<",
    "lt-fb-online-pruned": ">",
    "mp-fb-online-fg": "o",
    "mp-fb-online-hg": "o",
    "ucond-gg-gbo": "*",
    "ucond-fg-gbo": "*",
    "ucond-hg-gbo": "*",
    "mcond-hg-gbo": "*",
    "ucond-gg-fb": "+",
    "ucond-fg-fb": "+",
    "ucond-hg-fb": "+",
    "mcond-hg-fb": "+",
    "ucond-gg-fb-generic": "x",
    "ucond-fg-fb-generic": "x",
    "ucond-hg-fb-generic": "x",
    "mcond-hg-fb-generic": "x",
}

COLOR_ORDER = [
    "ucond-gg-gbo",
    "ucond-gg-fb",
    "ucond-gg-fb-generic",
    "mp-fb-online-fg",
    "univariate",
    "ucond-fg-gbo",
    "ucond-fg-fb",
    "ucond-fg-fb-generic",
    "full",
    "ucond-hg-gbo",
    "ucond-hg-fb",
    "ucond-hg-fb-generic",
    "mp-fb-online-hg",
    "lt-static-gbo",
    "mcond-hg-gbo",
    "mcond-hg-fb",
    "mcond-hg-fb-generic",
    "lt-fb-online-unpruned",
    "lt-fb-online-pruned",
]

COLORS = {}
for index, color in enumerate(COLOR_ORDER):
    COLORS[color] = cmap(index / (len(COLOR_ORDER) - 1))


def main(directory, configuration):
    if configuration == "all":
        linkage_models = [
            "univariate",
            "full",
            "lt-static-gbo",
            "lt-fb-online-unpruned",
            "lt-fb-online-pruned",
            "ucond-gg-gbo",
            "ucond-fg-gbo",
            "ucond-hg-gbo",
            "mcond-hg-gbo",
            "ucond-gg-fb",
            "ucond-fg-fb",
            "ucond-hg-fb",
            "mcond-hg-fb",
            "ucond-gg-fb-generic",
            "ucond-fg-fb-generic",
            "ucond-hg-fb-generic",
            "mcond-hg-fb-generic",
        ]
    elif configuration == "structure-known":
        linkage_models = [
            "lt-static-gbo",
            "ucond-gg-gbo",
            "ucond-fg-gbo",
            "ucond-hg-gbo",
            "mcond-hg-gbo",
        ]
    elif configuration == "structure-unknown":
        linkage_models = [
            "univariate",
            "full",
            "lt-fb-online-unpruned",
            "lt-fb-online-pruned",
            "ucond-gg-fb",
            "ucond-fg-fb",
            "ucond-hg-fb",
            "mcond-hg-fb",
            "ucond-gg-fb-generic",
            "ucond-fg-fb-generic",
            "ucond-hg-fb-generic",
            "mcond-hg-fb-generic",
        ]
    elif configuration == "non-conditional":
        linkage_models = [
            "univariate",
            "full",
            "lt-static-gbo",
            "lt-fb-online-unpruned",
            "lt-fb-online-pruned",
        ]
    elif configuration == "conditional":
        linkage_models = [
            "ucond-gg-gbo",
            "ucond-fg-gbo",
            "ucond-hg-gbo",
            "mcond-hg-gbo",
            "ucond-gg-fb",
            "ucond-fg-fb",
            "ucond-hg-fb",
            "mcond-hg-fb",
            "ucond-gg-fb-generic",
            "ucond-fg-fb-generic",
            "ucond-hg-fb-generic",
            "mcond-hg-fb-generic",
        ]
    elif configuration == "temp":
        linkage_models = [
            "mp-fb-online-fg",
            "mp-fb-online-hg",
        ]
    else:
        raise Exception("Unknown linkage model configuration")

    df = pd.read_csv(os.path.join(directory, "aggregated_results.csv"))

    for metric, title in [("population_size", "Population size"), ("median_num_evaluations", "Number of evaluations")]:
        f, ax = plt.subplots(figsize=(6, 6))
        # sns.lineplot(ax=ax, data=df[df["linkage_model"].str.contains("gbo")], x="dimensionality", y=metric, hue="linkage_model", style="linkage_model")

        for model in linkage_models:
            d = df[df["linkage_model"] == model].groupby(["dimensionality"]).median(numeric_only=True).reset_index()
            ax.plot(d["dimensionality"], d[metric], linestyle='-', color=COLORS[model], marker=MARKERS[model],
                    label=model)

        ax.legend(title='Linkage model', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xlabel("Dimensionality")
        ax.set_ylabel(title)
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(os.path.join(directory, f'scalability_plot_{metric}_{configuration}.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(directory, f'scalability_plot_{metric}_{configuration}.png'), bbox_inches='tight')
        plt.clf()


if __name__ == '__main__':
    for mode in ("temp",): #("all", "structure-known", "structure-unknown", "non-conditional", "conditional"):
        main(sys.argv[1], mode)
