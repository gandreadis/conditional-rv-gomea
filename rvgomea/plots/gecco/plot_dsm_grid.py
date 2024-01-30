import os.path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import ImageGrid

plt.style.use('science')

# Prevent scienceplots from being purged as import
scienceplots.listdir(".")

AGGREGATE = True

cmap = matplotlib.colormaps["coolwarm"] #LinearSegmentedColormap.from_list('', ['blue', 'purple'])
cmap.set_under('white')


def get_matrix(generation: int, df):
    string = df[df["generation"] == generation].to_records()[0]["matrix"]
    flat_matrix = [float(s) for s in string.split("|")]
    return np.reshape(np.array(flat_matrix), (int(np.sqrt(len(flat_matrix))), int(np.sqrt(len(flat_matrix)))))


def main(base_directory, problem_ids, problem_labels):
    assert len(problem_ids) == len(problem_labels)

    if AGGREGATE:
        for problem_id in problem_ids:
            matrices = []
            for i in range(1, 31):
                df = pd.read_csv(os.path.join(base_directory, problem_id, str(i),
                                              "fitness_dependency_monitoring_per_generation.dat"))
                num_generations = int(np.max(df["generation"]))
                matrix = get_matrix(int(num_generations * 0.5), df)[:20,:20]
                matrices.append(matrix)

            mean_dsm = np.mean(matrices, axis=0)

            zero_counter = np.zeros_like(matrices[-1])
            for m in matrices:
                zero_counter += m == 0

            mean_dsm = np.where(zero_counter > 25, 0, mean_dsm)
            max_dep = np.max(mean_dsm)
            if max_dep > 0:
                mean_dsm /= max_dep
            mean_dsm.tofile(os.path.join(base_directory, problem_id, "dsm.dat"))

    fig = plt.figure(figsize=(10, 4))

    axs = ImageGrid(fig, 111,
                    nrows_ncols=(2, 6),
                    share_all=True,
                    axes_pad=(0.3,0.3),
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_size=0.1,
                    cbar_pad=0.1,
                    aspect=True)

    cb = None
    for i, (problem_id, problem_label) in enumerate(zip(problem_ids, problem_labels)):
        ax = axs[i]

        dsm = np.fromfile(os.path.join(base_directory, problem_id, "dsm.dat"))
        dsm = np.reshape(dsm, (20, 20))

        ax.set_title(problem_label, fontsize=11)
        ax.tick_params(axis=u'both', which=u'both', length=0)
        ax.set_xticks(list(range(0, 20, 5)))
        ax.set_yticks(list(range(0, 20, 5)))
        ax.set_xticklabels(list(range(0, 20, 5)), fontsize=10)
        ax.set_yticklabels(list(range(0, 20, 5)), fontsize=10)
        cb = ax.imshow(dsm, cmap=cmap, vmin=1e-6, vmax=1)

    cbar = axs.cbar_axes[0].colorbar(cb, label="Dependency strength", extend='min',
                                     ticks=[1e-6, 0.2, 0.4, 0.6, 0.8, 1.0],)
    cbar.ax.set_yticklabels(['$<\\eta$', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)


    # Version in one column:
    #
    # fig = plt.figure(figsize=(6, 4))
    #
    # axs = ImageGrid(fig, 111,
    #                 nrows_ncols=(3, 4),
    #                 share_all=True,
    #                 axes_pad=(0.3,0.3),
    #                 cbar_mode='single',
    #                 cbar_location='right',
    #                 cbar_size=0.1,
    #                 cbar_pad=0.1,
    #                 aspect=True)
    #
    #       ax.set_title(problem_label, fontsize=9)
    #       ax.tick_params(axis=u'both', which=u'both', length=0)
    #       ax.set_xticks(list(range(0, 20, 5)))
    #       ax.set_yticks(list(range(0, 20, 5)))
    #       ax.set_xticklabels(list(range(0, 20, 5)), fontsize=9)
    #       ax.set_yticklabels(list(range(0, 20, 5)), fontsize=9)
    #       cb = ax.imshow(dsm, cmap=cmap, vmin=1e-6, vmax=1)
    #
    # cbar = axs.cbar_axes[0].colorbar(cb, label="Dependency strength", extend='min',
    #                                  ticks=[1e-6, 0.2, 0.4, 0.6, 0.8, 1.0],)
    # cbar.ax.set_yticklabels(['$<\\epsilon$', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)

    plt.tight_layout()

    plt.savefig(os.path.join(base_directory, "dsm_grid.png"), bbox_inches="tight")
    plt.savefig(os.path.join(base_directory, "dsm_grid.pdf"), bbox_inches="tight")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(","))
