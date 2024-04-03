import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import images2gif.images2gif as igf
from matplotlib.colors import LinearSegmentedColormap
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

df = pd.read_csv(os.path.join(sys.argv[1], "fitness_dependency_monitoring_per_generation.dat"))
num_generations = int(np.max(df["generation"]))


def get_matrix(generation: int):
    string = df[df["generation"] == generation].to_records()[0]["matrix"]
    flat_matrix = [float(s) for s in string.split("|")]
    return np.reshape(np.array(flat_matrix), (int(np.sqrt(len(flat_matrix))), int(np.sqrt(len(flat_matrix)))))


if sys.argv[2] == "single":
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    cmap = LinearSegmentedColormap.from_list('', ['white', 'darkblue'])
    cmap.set_under('darkred')

    ax.imshow(get_matrix(num_generations), cmap=cmap, vmin=1e-10, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(os.path.join(sys.argv[1], "dependency_matrix.png"), bbox_inches="tight")

elif sys.argv[2] == "animation":
    frame_dir = os.path.join(sys.argv[1], "dependency_matrix_frames")
    os.system(f"rm -rf {frame_dir} && mkdir -p {frame_dir}")

    images = []
    for n in range(num_generations + 1):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        cmap = LinearSegmentedColormap.from_list('', ['white', 'darkblue'])
        cmap.set_under('darkred')

        matrix = get_matrix(n)

        if np.sum(np.abs(matrix)) == 0:
            continue

        ax.imshow(matrix, cmap=cmap, vmin=1e-10, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Generation {n:03}")

        filename = os.path.join(frame_dir, f"{n}.png")
        plt.savefig(filename, bbox_inches="tight")
        images.append(filename)
        plt.clf()
        plt.close(fig)

    clip = ImageSequenceClip(images, fps=5)

    clip.write_gif(os.path.join(sys.argv[1], "fitness_dependency_matrix_progression.gif"), fps=5)
