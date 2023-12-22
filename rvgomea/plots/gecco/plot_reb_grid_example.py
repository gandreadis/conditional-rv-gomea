import os.path
import sys
import types

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scienceplots
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import ImageGrid

plt.style.use('science')

# Prevent scienceplots from being purged as import
scienceplots.listdir(".")

cond_dir = os.path.join(sys.argv[1], "cond")
mp_dir = os.path.join(sys.argv[1], "mp")
omp_dir = os.path.join(sys.argv[1], "omp")
lt_dir = os.path.join(sys.argv[1], "lt")

df = pd.read_csv(os.path.join(cond_dir, "fitness_dependency_monitoring_per_generation.dat"))
num_generations = int(np.max(df["generation"]))


def get_matrix(generation: int):
    string = df[df["generation"] == generation].to_records()[0]["matrix"]
    flat_matrix = [float(s) for s in string.split("|")]
    return np.reshape(np.array(flat_matrix), (int(np.sqrt(len(flat_matrix))), int(np.sqrt(len(flat_matrix)))))


fig = plt.figure(figsize=(3, 3))

grid = ImageGrid(fig, 111,
                 nrows_ncols=(1, 1),
                 share_all=True,
                 axes_pad=0.05,
                 cbar_mode='single',
                 cbar_location='right',
                 cbar_pad=0.1,
                 aspect=True
                 )

ax = grid[0]
cmap = matplotlib.colormaps["coolwarm"]
cmap.set_under('white')

dsm = get_matrix(num_generations)
cb = ax.imshow(dsm, cmap=cmap, vmin=1e-6, vmax=1)
cbar = grid.cbar_axes[0].colorbar(cb, label="Dependency strength", extend='min', ticks=[1e-6, 0.2, 0.4, 0.6, 0.8, 1.0])
cbar.ax.set_yticklabels(['$<10^{-6}$', '0.2', '0.4', '0.6', '0.8', '1.0'])

ax.tick_params(axis=u'both', which=u'both', length=0)
ax.set_xlabel("Variable")
ax.set_ylabel("Variable")
ax.set_xticks(list(range(0, 16, 2)))
ax.set_yticks(list(range(0, 16, 2)))
# ax.set_xlim(-0.5, 19.5)
# ax.set_ylim(19.5, -0.5)

plt.tight_layout()

plt.savefig(os.path.join(sys.argv[1], "dsm.png"), bbox_inches="tight")
plt.savefig(os.path.join(sys.argv[1], "dsm.pdf"), bbox_inches="tight")
plt.clf()
plt.close(fig)

# VIG drawing

fig, ax = plt.subplots(1, 1, figsize=(3, 3))

# m = np.zeros((16, 16))
#
# for x in range(4):
#     for y in range(4):
#         i = x * 4 + y
#         if x > 0:
#             j = (x - 1) * 4 + y
#             m[i, j] = 1
#             m[j, i] = 1
#         if x < 3:
#             j = (x + 1) * 4 + y
#             m[i, j] = 1
#             m[j, i] = 1
#         if y > 0:
#             j = x * 4 + y - 1
#             m[i, j] = 1
#             m[j, i] = 1
#         if y < 3:
#             j = x * 4 + y + 1
#             m[i, j] = 1
#             m[j, i] = 1

G = nx.from_numpy_array(dsm)
pos = {}

for x in range(4):
    for y in range(4):
        pos[x * 4 + y] = (y, 3 - x)

nx.draw_networkx_nodes(G, pos=pos, ax=ax, edgecolors='black', node_color='white', )
nx.draw_networkx_edges(G, pos=pos, ax=ax)

# Fix label alignment
for k, v in pos.items():
    pos[k] = (v[0] + 0.01, v[1] - 0.01)
nx.draw_networkx_labels(G, pos=pos, ax=ax)

ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()

plt.savefig(os.path.join(sys.argv[1], f"vig.png"), bbox_inches="tight")
plt.savefig(os.path.join(sys.argv[1], f"vig.pdf"), bbox_inches="tight")

# FOS drawing

for mode, directory in (("mp", mp_dir), ("omp", omp_dir), ("lt", lt_dir)):
    df = pd.read_csv(os.path.join(directory, "fitness_dependency_monitoring_per_generation.dat"))
    num_generations = int(np.max(df["generation"]))


    def get_fos(generation: int):
        string = df[df["generation"] == generation].to_records()[0]["fos"]
        fos_sets = [eval(s.replace("^", ",")) for s in string.split("|")]

        fos_sets = sorted(fos_sets, key=lambda x: (len(x), x))
        return fos_sets


    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    cmap = plt.colormaps.get_cmap('viridis')
    fosses = get_fos(num_generations)

    max_val = 0
    for ind, f in enumerate(fosses):
        for v in f:
            max_val = max(v, max_val)
            rect = Rectangle((v, ind),
                             1, 1,
                             color=cmap(ind / (len(fosses) - 1)))

            ax.add_patch(rect)

    ax.set_xlim([0, max_val + 1])
    ax.set_ylim([0, len(fosses)])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Variable")
    ax.set_ylabel("FOS set")
    ax.set_xticks(list(range(0, 16, 2)))
    ax.tick_params(axis='both', which='both', length=0)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.grid(which="both")

    for label in ax.xaxis.get_majorticklabels():
        label.customShiftValue = 0.5
        label.set_x = types.MethodType(lambda self, x: matplotlib.text.Text.set_x(self, x + self.customShiftValue),
                                       label)

    plt.savefig(os.path.join(sys.argv[1], f"fos_{mode}.png"), bbox_inches="tight")
    plt.savefig(os.path.join(sys.argv[1], f"fos_{mode}.pdf"), bbox_inches="tight")
