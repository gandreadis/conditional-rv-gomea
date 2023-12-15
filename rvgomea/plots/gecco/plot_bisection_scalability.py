import os
import sys

import matplotlib
import pandas as pd
import scienceplots
from matplotlib import pyplot as plt

plt.style.use('science')

# Prevent scienceplots from being purged as import
scienceplots.listdir(".")

cmap = matplotlib.colormaps['tab10']

LABELS = {
    "univariate": "Univariate",
    "full": "Full",
    "mp-hg-gbo-without_clique_seeding-conditional": "Static-MCond-HG",
    "mp-hg-gbo-with_clique_seeding-conditional": "Static-MCond-HG-CS",
    "lt-fb-online-pruned": "FB-LT",
    "uni-hg-fb_no_order-without_clique_seeding-conditional": "FB-UCond-HG",
    "mp-hg-fb_no_order-without_clique_seeding-conditional": "FB-MCond-HG",
    "mp-hg-fb_no_order-with_clique_seeding-conditional": "FB-MCond-HG-CS",
    "mp-fg-gbo-without_clique_seeding-non_conditional-set_cover": "Static-EdgeCover",
    "mp-hg-gbo-without_clique_seeding-conditional-set_cover": "Static-EdgeCover-MCond-HG",
}

MARKERS = {
    "univariate": "+",
    "full": "*",
    "mp-hg-gbo-without_clique_seeding-conditional": ">",
    "mp-hg-gbo-with_clique_seeding-conditional": "<",
    "lt-fb-online-pruned": "o",
    "uni-hg-fb_no_order-without_clique_seeding-conditional": "s",
    "mp-hg-fb_no_order-without_clique_seeding-conditional": "x",
    "mp-hg-fb_no_order-with_clique_seeding-conditional": "X",
    "mp-fg-gbo-without_clique_seeding-non_conditional-set_cover": "2",
    "mp-hg-gbo-without_clique_seeding-conditional-set_cover": "d",
}

COLOR_ORDER = {
    "univariate": 7,
    "full": 6,
    "mp-hg-gbo-without_clique_seeding-conditional": 5,
    "mp-hg-gbo-with_clique_seeding-conditional": 4,
    "lt-fb-online-pruned": 3,
    "uni-hg-fb_no_order-without_clique_seeding-conditional": 2,
    "mp-hg-fb_no_order-without_clique_seeding-conditional": 1,
    "mp-hg-fb_no_order-with_clique_seeding-conditional": 0,
    "mp-fg-gbo-without_clique_seeding-non_conditional-set_cover": 8,
    "mp-hg-gbo-without_clique_seeding-conditional-set_cover": 9,
}

COLORS = {}
for lm, color_index in COLOR_ORDER.items():
    COLORS[lm] = cmap(color_index / (len(COLOR_ORDER) - 1))


def main(base_directory, problem_ids, problem_labels, linkage_models):
    assert len(problem_ids) == len(problem_labels)

    if len(problem_ids) == 1:
        plot_directory = base_directory
    else:
        plot_directory = base_directory[:-1]
        os.system(f"mkdir -p {plot_directory}")

    for metric, metric_label in [("population_size", "Population size"),
                                 ("corrected_num_evaluations", "Corrected num. evaluations")]:

        handles, labels = None, None

        if len(problem_ids) == 1:
            fig, ax = plt.subplots(1, 1, figsize=(3, 2))
            handles, labels = make_one_plot(ax, base_directory, linkage_models, metric, None)
        else:
            fig, axs = plt.subplots(3, 4, figsize=(10, 7), sharex=True, sharey=True)

            for i, (problem_id, problem_label) in enumerate(zip(problem_ids, problem_labels)):
                ax = axs[i // 4, i % 4]

                handles, labels = make_one_plot(ax, base_directory + problem_id, linkage_models, metric, problem_label)

        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 1.01), ncol=len(linkage_models) // 2)

        # Global labels
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel('Dimensionality')
        plt.ylabel(metric_label)


        if len(problem_ids) == 1:
            prefix = "set_cover_scalability"
        else:
            prefix = "scalability"

        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, f'{prefix}_{metric}.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(plot_directory, f'{prefix}_{metric}.png'), bbox_inches='tight')
        plt.clf()
        plt.close(fig)


def make_one_plot(ax, directory, linkage_models, metric, problem_label):
    df = pd.read_csv(os.path.join(directory, "aggregated_results.csv"))

    for lm in linkage_models:
        d = df[df["linkage_model"] == lm].groupby(["dimensionality"]).median(numeric_only=True).reset_index()
        ax.plot(d["dimensionality"], d[metric], linestyle='-', color=COLORS[lm], marker=MARKERS[lm],
                label=LABELS[lm])

    handles, labels = ax.get_legend_handles_labels()
    if problem_label is not None:
        ax.set_title(problem_label)

    plt.xscale('log')
    plt.yscale('log')
    return handles, labels


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(","), sys.argv[4].split(","))
