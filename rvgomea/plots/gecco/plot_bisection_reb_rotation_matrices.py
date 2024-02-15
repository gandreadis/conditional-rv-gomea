import os
import sys

import numpy as np
import pandas as pd
import scienceplots
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import ImageGrid

from rvgomea.defaults import DEFAULT_MAX_NUM_EVALUATIONS

plt.style.use('science')

# Prevent scienceplots from being purged as import
scienceplots.listdir(".")

ranges = {
    "lt-vs-mcond": [1e3, 1e6],
}
palettes = {
    "lt-vs-mcond": "inferno",
}


def main(directory, linkage_model_ids, linkage_model_labels):
    assert len(linkage_model_ids) == len(linkage_model_labels)

    run_id = directory.split('-bisection-')[-1]

    num_models = len(linkage_model_ids)
    df_full = pd.read_csv(os.path.join(directory, "aggregated_results.csv"))
    plot_directory = os.path.join(directory, "plots")
    os.system(f"mkdir -p {plot_directory}")

    fig = plt.figure(figsize=(5, 2.25))

    grid = ImageGrid(fig, 111,
                     nrows_ncols=(1, num_models),
                     share_all=True,
                     axes_pad=0.05,
                     cbar_mode='single',
                     cbar_location='right',
                     cbar_pad=0.1,
                     aspect=False)

    cb = None
    for i, ax in enumerate(grid):
        matrix = np.zeros((10, 6))

        df = df_full[df_full["linkage_model"] == linkage_model_ids[i]]
        for rot_angle in range(10):
            actual_angle = rot_angle * 5
            for cond_number in range(1, 7):
                d = df[df["problem"] == f"reb-chain-condition-{cond_number}-rotation-{actual_angle}"]
                if len(d) == 0:
                    result = -1  # DEFAULT_MAX_NUM_EVALUATIONS
                else:
                    result = np.median(d["corrected_num_evaluations"])
                matrix[rot_angle, cond_number - 1] = result

        matrix_flipped = np.flipud(matrix)

        ax.set_title(linkage_model_labels[i].strip())

        cb = ax.imshow(matrix_flipped, cmap=palettes[run_id],
                       norm=LogNorm(vmin=ranges[run_id][0], vmax=ranges[run_id][1]), extent=[0.5, 6.5, -2.5, 47.5],
                       aspect="auto")
        ax.tick_params(axis=u'both', which=u'both', length=0)
        ax.set_xticks([2, 4, 6])
        ax.set_yticks([0, 15, 30, 45])

        # Uncomment this to show text in matrix fields
        # for rot_angle in range(10):
        #     actual_angle = rot_angle * 5
        #     for cond_number in range(1, 7):
        #         text = ax.text(cond_number + 0.33, actual_angle,
        #                        f"{int(round(matrix[rot_angle, cond_number - 1] / 1e3))}k",
        #                        ha="right", va="center", color="w", fontsize="x-small")

    grid.cbar_axes[0].colorbar(cb, label="Corr. num. evaluations")

    # Global labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('condition number c')
    plt.ylabel('rotation angle $\\theta$')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_directory, f"bisection_matrices_{run_id}.png"), bbox_inches='tight')
    plt.savefig(os.path.join(plot_directory, f"bisection_matrices_{run_id}.pdf"), bbox_inches='tight')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(","))
