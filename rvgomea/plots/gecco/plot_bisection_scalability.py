import os
import sys

import matplotlib
import pandas as pd
import scienceplots
from matplotlib import pyplot as plt

plt.style.use('science')

# Prevent scienceplots from being purged as import
scienceplots.listdir(".")

cmap = matplotlib.colormaps['tab20c']

LABELS = {
    "univariate": "Univariate",
    "full": "Full",
    "mp-hg-gbo-without_clique_seeding-conditional": "Static-MCond-HG",
    "mp-hg-gbo-with_clique_seeding-conditional": "Static-MCond-HG-MC",
    "lt-fb-online-pruned": "FB-LT",
    "uni-hg-fb_no_order-without_clique_seeding-conditional": "FB-UCond-HG",
    "mp-hg-fb_no_order-without_clique_seeding-conditional": "FB-MCond-HG",
    "mp-hg-fb_no_order-with_clique_seeding-conditional": "FB-MCond-HG-MC",
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
}

COLOR_ORDER = {
    "univariate": 0,
    "full": 4,
    "mp-hg-gbo-without_clique_seeding-conditional": 1,
    "mp-hg-gbo-with_clique_seeding-conditional": 5,
    "lt-fb-online-pruned": 2,
    "uni-hg-fb_no_order-without_clique_seeding-conditional": 6,
    "mp-hg-fb_no_order-without_clique_seeding-conditional": 3,
    "mp-hg-fb_no_order-with_clique_seeding-conditional": 7,
}

COLORS = {}
for lm, color_index in COLOR_ORDER.items():
    COLORS[lm] = cmap(color_index / (len(COLOR_ORDER) - 1))


def main(base_directory, problem_ids, problem_labels):
    num_problems = len(problem_ids)
    assert num_problems == 12
    assert len(problem_ids) == len(problem_labels)
    plot_directory = base_directory[:-1]
    os.system(f"mkdir -p {plot_directory}")

    for metric, metric_label in [("population_size", "Population size"),
                                 ("median_num_evaluations", "Num. evaluations")]:
        fig, axs = plt.subplots(3, 4, figsize=(10, 7), sharex=True, sharey=True)

        handles, labels = None, None
        for i, (problem_id, problem_label) in enumerate(zip(problem_ids, problem_labels)):
            if problem_id == "summation-cancellation":
                continue

            ax = axs[i // 4, i % 4]
            df = pd.read_csv(os.path.join(base_directory + problem_id, "aggregated_results.csv"))

            for lm in LABELS.keys():
                d = df[df["linkage_model"] == lm].groupby(["dimensionality"]).median(numeric_only=True).reset_index()
                ax.plot(d["dimensionality"], d[metric], linestyle='-', color=COLORS[lm], marker=MARKERS[lm],
                        label=LABELS[lm])

            handles, labels = ax.get_legend_handles_labels()

            ax.set_title(problem_label)
            plt.xscale('log')
            plt.yscale('log')

        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 1.01), ncol=len(LABELS) // 2)

        # Global labels
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel('Dimensionality')
        plt.ylabel(metric_label)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, f'scalability_{metric}.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(plot_directory, f'scalability_{metric}.png'), bbox_inches='tight')
        plt.clf()
        plt.close(fig)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(","))
