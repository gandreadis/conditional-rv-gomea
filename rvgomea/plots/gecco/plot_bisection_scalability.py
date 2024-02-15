import math
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import scienceplots
from matplotlib import pyplot as plt
from scipy.stats import mannwhitneyu

from rvgomea.defaults import DEFAULT_MAX_NUM_EVALUATIONS

plt.style.use('science')

# Prevent scienceplots from being purged as import
scienceplots.listdir(".")

cmap = matplotlib.colormaps['tab10']

LABELS = {
    "univariate": "Univariate",
    "full": "Full",
    "uni-hg-gbo-without_clique_seeding-conditional": "Static-UCond-HG",
    "mp-hg-gbo-without_clique_seeding-conditional": "Static-MCond-HG",
    "mp-hg-gbo-with_clique_seeding-conditional": "Static-MCond-HG-CS",
    "lt-fb-online-pruned": "FB-LT",
    "uni-hg-fb_no_order-without_clique_seeding-conditional": "FB-UCond-HG",
    "mp-hg-fb_no_order-without_clique_seeding-conditional": "FB-MCond-HG",
    "mp-hg-fb_no_order-with_clique_seeding-conditional": "FB-MCond-HG-CS",
    "vkd-cma": "VkD-CMA",
}

MARKERS = {
    "univariate": "+",
    "full": "*",
    "uni-hg-gbo-without_clique_seeding-conditional": ">",
    "mp-hg-gbo-without_clique_seeding-conditional": "<",
    "mp-hg-gbo-with_clique_seeding-conditional": "d",
    "lt-fb-online-pruned": "s",
    "uni-hg-fb_no_order-without_clique_seeding-conditional": "2",
    "mp-hg-fb_no_order-without_clique_seeding-conditional": "x",
    "mp-hg-fb_no_order-with_clique_seeding-conditional": "X",
    "vkd-cma": "o",
}

COLOR_ORDER = {
    "univariate": 8,
    "full": 7,
    "uni-hg-gbo-without_clique_seeding-conditional": 6,
    "mp-hg-gbo-without_clique_seeding-conditional": 5,
    "mp-hg-gbo-with_clique_seeding-conditional": 4,
    "lt-fb-online-pruned": 2,
    "uni-hg-fb_no_order-without_clique_seeding-conditional": 1,
    "mp-hg-fb_no_order-without_clique_seeding-conditional": 0,
    "mp-hg-fb_no_order-with_clique_seeding-conditional": 9,
    "vkd-cma": 3,
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
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            handles, labels = make_one_plot(ax, base_directory, problem_ids[0], linkage_models, metric, None)
        else:
            fig, axs = plt.subplots(3, 4, figsize=(10, 6), sharex=True, sharey=True)

            for i, (problem_id, problem_label) in enumerate(zip(problem_ids, problem_labels)):
                ax = axs[i // 4, i % 4]

                handles, labels = make_one_plot(ax, base_directory + problem_id, problem_id, linkage_models, metric,
                                                problem_label)

        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 1.01), ncol=math.ceil(len(linkage_models) / 2))

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


def make_one_plot(ax, directory, problem_id, linkage_models, metric, problem_label):
    df = pd.read_csv(os.path.join(directory, "aggregated_results.csv"))

    dfs = []
    for i in range(5):
        extrapolated_results_name = os.path.join(directory, f"extrapolated_results_{i}.csv")
        extrapolated_df = pd.read_csv(extrapolated_results_name)
        dfs.append(extrapolated_df)
    extrapolated_df = pd.concat(dfs)
    extrapolated_df = extrapolated_df[extrapolated_df["corrected_num_evaluations"] < DEFAULT_MAX_NUM_EVALUATIONS]

    for lm in linkage_models:
        d = df[df["linkage_model"] == lm].groupby(["dimensionality"]).median(numeric_only=True).reset_index()

        ax.plot(d["dimensionality"], d[metric], linestyle='-', color=COLORS[lm], marker=MARKERS[lm],
                label=LABELS[lm])

        # One or more dimensions were too difficult
        if len(d["dimensionality"]) < 4 or extrapolated_df is None:
            continue

        last_dim = np.max(d["dimensionality"])
        last_value = np.median(d[d["dimensionality"] == last_dim][metric].tolist())

        extra_dimensions = [last_dim]
        extra_values = [last_value]

        e = extrapolated_df[extrapolated_df["problem"] == problem_id]
        e = e[e["linkage_model"] == lm]
        e = e.groupby(["dimensionality"]).median(numeric_only=True).reset_index().sort_values(by=["dimensionality"])

        for extra_dim, extra_value, corr_eval in zip(e["dimensionality"].tolist(),
                                                     e[metric].tolist(),
                                                     e["corrected_num_evaluations"].tolist()):
            if corr_eval >= DEFAULT_MAX_NUM_EVALUATIONS:
                break

            extra_dimensions.append(extra_dim)
            extra_values.append(extra_value)

        line_style = "--" if metric == "population_size" else "-"
        ax.plot(extra_dimensions, extra_values, linestyle=line_style, color=COLORS[lm], marker=MARKERS[lm],
                label="_" + LABELS[lm])

    handles, labels = ax.get_legend_handles_labels()

    if problem_label is not None:
        ax.set_title(problem_label)

    plt.xscale('log')
    plt.yscale('log')

    if metric == "corrected_num_evaluations":
        with_vig = ["uni-hg-gbo-without_clique_seeding-conditional", "mp-hg-gbo-without_clique_seeding-conditional",
                    "mp-hg-gbo-with_clique_seeding-conditional", ]
        without_vig = ["univariate", "lt-fb-online-pruned", "uni-hg-fb_no_order-without_clique_seeding-conditional",
                       "mp-hg-fb_no_order-without_clique_seeding-conditional",
                       "mp-hg-fb_no_order-with_clique_seeding-conditional", "vkd-cma"]
        largest_dim = sorted(list(
            extrapolated_df[extrapolated_df["linkage_model"] == "mp-hg-fb_no_order-with_clique_seeding-conditional"][
                "dimensionality"]))[-1]

        eval_sets = []
        median_evals = []

        for lm in linkage_models:
            d = extrapolated_df[extrapolated_df["linkage_model"] == lm]
            filtered = d[d["dimensionality"] == largest_dim]
            if len(filtered) == 0:
                eval_set = [1e7] * 5
            else:
                eval_set = list(filtered["corrected_num_evaluations"])

            eval_sets.append(eval_set)
            evals = np.median(eval_set)
            median_evals.append(evals)

        best_lm_with = None
        best_lm_without = None
        p_with = None
        p_without = None

        for mode in ("WITH", "WITHOUT"):
            filtered_lm_list = []
            filtered_lm_index_list = []
            filtered_eval_sets = []
            filtered_median_evals = []
            vig_list = with_vig if mode == "WITH" else without_vig
            for i, lm in enumerate(linkage_models):
                if lm in vig_list:
                    filtered_lm_index_list.append(i)
                    filtered_lm_list.append(lm)
                    filtered_eval_sets.append(eval_sets[i])
                    filtered_median_evals.append(median_evals[i])

            if len(np.unique(filtered_median_evals)) == 1:
                if mode == "WITH":
                    assert problem_id == "sphere"
                    best_lm_with = LABELS["uni-hg-gbo-without_clique_seeding-conditional"]
                    p_with = "1.000"
                else:
                    raise Exception("Not expected")

                continue

            best_index = np.argmin(filtered_median_evals)
            best_lm = filtered_lm_list[best_index]

            p_values = []
            for i, lm in enumerate(filtered_lm_list):
                if i == best_index:
                    continue
                _, p = mannwhitneyu(filtered_eval_sets[i], filtered_eval_sets[best_index])
                if p > 0.99999:
                    continue
                p_values.append(p)

            max_p = np.max(p_values)

            if max_p < 0.001:
                p_string = "{\\bftab $<0.001$}"
            elif max_p < 0.05 / (len(filtered_eval_sets) - 1):
                p_string = "{\\bftab " + f"{max_p:0.3f}" "}"
            else:
                p_string = f"{max_p:0.3f}"

            if mode == "WITH":
                best_lm_with = LABELS[best_lm]
                p_with = p_string
            else:
                best_lm_without = LABELS[best_lm]
                p_without = p_string

        print(f"{problem_label[:3]} & {best_lm_with} & {p_with} & {best_lm_without} & {p_without} \\\\")

    return handles, labels


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(","), sys.argv[4].split(","))
