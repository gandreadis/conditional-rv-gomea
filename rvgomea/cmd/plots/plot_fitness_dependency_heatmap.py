import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

df = pd.read_csv(os.path.join(sys.argv[1], "fitness_dependency_monitoring_per_generation.dat"))
num_generations = int(np.max(df["generation"]))


def get_matrix(generation: int):
    string = df[df["generation"] == generation].to_records()[0]["matrix"]
    flat_matrix = [float(s) for s in string.split("|")]
    return np.reshape(np.array(flat_matrix), (int(np.sqrt(len(flat_matrix))), int(np.sqrt(len(flat_matrix)))))


fig, ax = plt.subplots(1, 1, figsize=(5, 5))

cmap = LinearSegmentedColormap.from_list('', ['white', 'darkblue'])

ax.imshow(get_matrix(num_generations), cmap=cmap)
ax.set_xticks([])
ax.set_yticks([])

plt.savefig(os.path.join(sys.argv[1], "dependency_matrix.png"), bbox_inches="tight")
