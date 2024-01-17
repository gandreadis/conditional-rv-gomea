import os.path
import sys

import matplotlib
import numpy as np
import scienceplots
from matplotlib import pyplot as plt

num_objectives = 3
num_points = 100
num_vars = num_points * 3 * 2

plt.style.use('science')

# Prevent scienceplots from being purged as import
scienceplots.listdir(".")

cmap = matplotlib.colormaps["coolwarm"]
cmap.set_under('white')


def main(base_dir: str):
    figure_dir = os.path.join(base_dir, "fitness_dependency_renders")
    os.system(f"mkdir -p {figure_dir}")

    dependency_matrices = []
    dependency_matrices_per_point = []
    for obj in range(num_objectives):
        dependency_matrices.append(
            np.reshape(np.fromfile(os.path.join(base_dir, f"fitness_dependency_matrix_processed_obj{obj}.np")),
                       (num_vars, num_vars)))

        dependency_matrices_per_point.append(np.zeros((num_points, num_points)))
        for i in range(num_points):
            for j in range(num_points):
                dependency_matrices_per_point[obj][i, j] = np.average(
                    dependency_matrices[obj][i * 3:(i + 1) * 3, j * 3:(j + 1) * 3])

    adjacency_matrix_per_coordinate = []
    for i in range(3):
        adjacency_matrix_per_coordinate.append(
            np.reshape(np.fromfile(os.path.join(base_dir, f"adjacency_matrix_coord{i}.np")), (num_points, num_points))
        )

    for obj in range(num_objectives):
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        cb = ax.imshow(dependency_matrices[obj], interpolation="nearest", cmap=cmap,
                       extent=(0, dependency_matrices[obj].shape[0], dependency_matrices[obj].shape[1], 0), vmin=1e-6,
                       vmax=1.0)

        cbar = fig.colorbar(cb, label="Dependency strength", extend='min', ticks=[1e-6, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.ax.set_yticklabels(['$<10^{-6}$', '0.2', '0.4', '0.6', '0.8', '1.0'])

        ax.tick_params(axis='both', which='minor', length=0)
        ax.set_xticks(list(range(0, 600, 200)))
        ax.set_yticks(list(range(0, 600, 200)))

        plt.savefig(os.path.join(figure_dir, f"dependency_matrix_obj{obj}.png"), bbox_inches="tight", dpi=300)
        plt.savefig(os.path.join(figure_dir, f"dependency_matrix_obj{obj}.pdf"), bbox_inches="tight")
        plt.close(fig)

        # Zoomin
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        cb = ax.imshow(dependency_matrices[obj], interpolation="nearest", cmap=cmap,
                       extent=(0, dependency_matrices[obj].shape[0], dependency_matrices[obj].shape[1], 0), vmin=1e-6,
                       vmax=1.0)
        ax.tick_params(axis='both', which='minor', length=0)
        ax.set_xlim(26 * 3, 50 * 3)
        ax.set_ylim(26 * 3, 50 * 3)
        # fig.colorbar(cb, ax=ax)

        plt.savefig(os.path.join(figure_dir, f"dependency_matrix_zoomin_obj{obj}.png"), bbox_inches="tight")
        plt.savefig(os.path.join(figure_dir, f"dependency_matrix_zoomin_obj{obj}.pdf"), bbox_inches="tight")
        plt.close(fig)

        # One mesh
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        cb = ax.imshow(dependency_matrices[obj][:num_points * 3, :num_points * 3], interpolation="nearest", cmap=cmap,
                       extent=(0, dependency_matrices[obj][:num_points * 3, :num_points * 3].shape[0],
                               dependency_matrices[obj][:num_points * 3, :num_points * 3].shape[1], 0), vmin=1e-6,
                       vmax=1.0)
        fig.colorbar(cb, ax=ax)

        plt.savefig(os.path.join(figure_dir, f"dependency_matrix_one_mesh_obj{obj}.png"), bbox_inches="tight")
        plt.savefig(os.path.join(figure_dir, f"dependency_matrix_one_mesh_obj{obj}.pdf"), bbox_inches="tight")
        plt.close(fig)


if __name__ == '__main__':
    main(sys.argv[1])
