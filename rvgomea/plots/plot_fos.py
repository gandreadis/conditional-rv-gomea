import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

df = pd.read_csv(os.path.join(sys.argv[1], "fitness_dependency_monitoring_per_generation.dat"))
num_generations = int(np.max(df["generation"]))


def get_fos(generation: int):
    string = df[df["generation"] == generation].to_records()[0]["fos"]
    fos = [eval(s.replace("^", ",")) for s in string.split("|")]
    fos = sorted(fos, key=lambda x: (len(x), x))
    return fos


if sys.argv[2] == "single":
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    cmap = plt.colormaps.get_cmap('viridis')
    fos_sets = get_fos(num_generations)

    max_val = 0
    for ind, f in enumerate(fos_sets):
        for v in f:
            max_val = max(v, max_val)
            rect = Rectangle((v, ind),
                             1, 1,
                             color=cmap(ind / (len(fos_sets) - 1)))

            ax.add_patch(rect)

    ax.set_xlim([0, max_val + 1])
    ax.set_ylim([0, len(fos_sets)])
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(os.path.join(sys.argv[1], "fos.png"), bbox_inches="tight")

elif sys.argv[2] == "animation":
    frame_dir = os.path.join(sys.argv[1], "fos_frames")
    os.system(f"rm -rf {frame_dir} && mkdir -p {frame_dir}")

    images = []
    for n in range(num_generations + 1):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        cmap = plt.colormaps.get_cmap('viridis')
        fos_sets = get_fos(n)

        max_val = 0
        for ind, f in enumerate(fos_sets):
            for v in f:
                max_val = max(v, max_val)
                rect = Rectangle((v, ind),
                                 1, 1,
                                 color=cmap(ind / (len(fos_sets) - 1)))

                ax.add_patch(rect)

        ax.set_xlim([0, max_val + 1])
        ax.set_ylim([0, len(fos_sets)])
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(f"Generation {n:03}")

        filename = os.path.join(frame_dir, f"{n}.png")
        plt.savefig(filename, bbox_inches="tight")
        images.append(filename)
        plt.clf()
        plt.close(fig)

    clip = ImageSequenceClip(images, fps=5)

    clip.write_gif(os.path.join(sys.argv[1], "fos_progression.gif"), fps=5)
